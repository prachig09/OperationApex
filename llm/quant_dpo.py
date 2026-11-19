"""4-bit quantization inspector/loader and Human-in-the-Loop -> DPO preference pipeline.

Part 1 - Quantization
    * ``QuantizedModelLoader`` connects to a local quantized model served by Ollama (default
      ``qwen2.5:3b-instruct``, Q4_K_M GGUF) and reports measured disk/RAM footprint versus the FP16 size.
    * If Ollama is unreachable or the model is not pulled, it falls back to ``RuleTemplateEngine``: a
      deterministic extractor that copies verbatim sentences (so it also passes the RAG Triad gate).
      Both backends expose the same ``chat_json`` / ``ping`` / ``cfg.model_name`` surface as Stage 2's
      ``LocalLLMClient`` and can be passed to ``LocalRiskExtractor`` unchanged.
    * ``quantize_demo`` runs real block-wise 4-bit quantization on a random weight matrix (NumPy only).

Part 2 - HITL -> DPO
    * ``build_pair_from_resolution`` turns a reviewer decision into ``{"prompt","chosen","rejected"}``.
    * ``DPOStore`` writes every pair to ``apex_data/dpo_preferences.jsonl`` (exactly those 3 keys) and to SQLite.
    * ``run_dpo_step`` computes exact DPO loss, implicit rewards and gradients in NumPy on CPU in well under
      2 s, using a small hashed-unigram policy. It demonstrates the objective; it does NOT update Qwen weights.
      With a GPU, ``trl`` + ``peft`` and a local HF base model (``APEX_DPO_BASE_MODEL``) it runs a real LoRA
      DPO step via ``train_with_trl``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import re
import sqlite3
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import llm.extractor as _ex
from llm.extractor import LLMConfig, LLMUnavailableError, LocalLLMClient
from pipeline.ingest import _SENTENCE_SPLIT

LOG = logging.getLogger("apex.quant_dpo")

TARGET_MODEL = os.getenv("APEX_LLM_MODEL") or "qwen2.5:3b-instruct"
REFERENCE = {"params_b": 3.09, "fp16_gb": 6.2, "q4km_gb": 2.1}  # nominal figures quoted for Qwen2.5-3B
_WORD = re.compile(r"[a-z0-9]+")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_[0-9A-F]{6}\]")
_OVERRIDE_FIELDS = {"category", "severity", "title", "summary", "evidence_quote", "financial_exposure", "recommended_action"}

Q4KM_EXPLAINER = (
    "Q4_K_M is a llama.cpp 'k-quant': weights are stored as 4-bit integers in super-blocks of 256 weights "
    "(8 sub-blocks of 32) with 6-bit per-sub-block scales and minimums, and the most quality-sensitive tensors "
    "(half of attention.wv and ffn_down) are kept at 6-bit (Q6_K). Effective cost is about 4.8 bits per weight "
    "versus 16 for FP16, so roughly a 3x smaller file at a small perplexity cost."
)


# ============================================================== quantization inspector
@dataclass
class QuantReport:
    """Footprint report for a quantized model (``measured`` is False for reference figures)."""

    backend: str
    model: str
    quantization: str
    parameter_count_b: float
    disk_gb: float
    resident_gb: Optional[float]
    fp16_gb: float
    reduction_x: float
    reduction_pct: float
    bits_per_weight: float
    measured: bool
    notes: List[str] = field(default_factory=list)
    explainer: str = Q4KM_EXPLAINER

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view."""
        return asdict(self)


def _parse_params_b(text: Any) -> Optional[float]:
    m = re.match(r"\s*([\d.]+)\s*([BMK]?)", str(text or ""), re.I)
    if not m:
        return None
    scale = {"B": 1.0, "M": 1e-3, "K": 1e-6, "": 1e-9}[m.group(2).upper()]
    return float(m.group(1)) * scale


def _build_report(backend: str, model: str, quant: str, params_b: float, disk_gb: float,
                  resident: Optional[float], measured: bool, notes: List[str]) -> QuantReport:
    fp16 = params_b * 2.0  # 2 bytes per weight
    return QuantReport(
        backend=backend, model=model, quantization=quant, parameter_count_b=round(params_b, 3),
        disk_gb=round(disk_gb, 2), resident_gb=None if resident is None else round(resident, 2),
        fp16_gb=round(fp16, 2), reduction_x=round(fp16 / disk_gb, 2), reduction_pct=round(100 * (1 - disk_gb / fp16), 1),
        bits_per_weight=round(disk_gb * 8 / params_b, 2), measured=measured, notes=notes,
    )


def inspect_ollama(client: LocalLLMClient, model: str) -> Optional[QuantReport]:
    """Read measured quantization/size facts for ``model`` from a local Ollama server (None if not pulled)."""
    tags = client._request("GET", "/api/tags").get("models", [])  # noqa: SLF001  (guarded local request)
    entry = next((m for m in tags if m.get("name") in (model, f"{model}:latest") or m.get("model") == model), None)
    if entry is None:
        return None
    show: Dict[str, Any] = {}
    for key in ("model", "name"):  # the field was renamed across Ollama versions
        try:
            show = client._request("POST", "/api/show", {key: entry["name"]})  # noqa: SLF001
            break
        except LLMUnavailableError:
            continue
    details = show.get("details") or entry.get("details") or {}
    info = show.get("model_info") or {}
    params_b = (info.get("general.parameter_count") or 0) / 1e9 or _parse_params_b(details.get("parameter_size")) or REFERENCE["params_b"]
    resident = None
    try:
        loaded = client._request("GET", "/api/ps").get("models", [])  # noqa: SLF001
        hit = next((m for m in loaded if m.get("name") == entry["name"]), None)
        resident = hit["size"] / 1e9 if hit else None
    except LLMUnavailableError:
        pass
    notes = ["Measured from the local Ollama server (/api/tags, /api/show, /api/ps).",
             "resident_gb is None until the model has been loaded by a first request."]
    return _build_report("ollama-live", entry["name"], details.get("quantization_level", "unknown"), params_b,
                         entry["size"] / 1e9, resident, True, notes)


# ===================================================================== rule fallback
_CATEGORY_RULES: Dict[str, Tuple[re.Pattern, str, str]] = {
    "CHANGE_OF_CONTROL": (re.compile(r"change\s+(?:of|in)\s+control", re.I), "Change of Control trigger",
                          "Seek waivers or amendments from the affected counterparties before signing."),
    "TERMINATION_RIGHTS": (re.compile(r"without\s+(?:any\s+)?(?:penalty|liability|termination\s+fee)|terminat\w+[^.]{0,120}(?:notice|breach)", re.I),
                           "Termination right", "Map every exit right and its notice window in the disclosure schedule."),
    "CUSTOMER_CONCENTRATION": (re.compile(r"(?:accounted\s+for|represent\w*|approximately)[^.]{0,80}\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?\s?%[^.]{0,60}(?:revenue|customers)", re.I),
                               "Customer concentration", "Model downside revenue scenarios for the top customers."),
    "LIABILITY_INDEMNITY": (re.compile(r"indemnif|aggregate\s+liability|shall\s+not\s+exceed|insurance", re.I),
                            "Liability and indemnity term", "Confirm caps and carve-outs against deal risk appetite."),
    "FINANCIAL_COVENANT": (re.compile(r"revolving\s+credit|covenant|borrowings|minimum\s+purchase|credit\s+facility", re.I),
                           "Financial commitment", "Verify facility terms and change-of-control prepayment triggers."),
}
_QUESTION_HINTS = {
    "CHANGE_OF_CONTROL": r"change of control|acquisition|acquir|takeover",
    "TERMINATION_RIGHTS": r"terminat|cancel|penalt",
    "CUSTOMER_CONCENTRATION": r"concentrat|largest customers|share of revenue|percent",
    "LIABILITY_INDEMNITY": r"liabilit|indemn|insurance",
    "FINANCIAL_COVENANT": r"debt|covenant|commitment|obligation|credit|facilit",
}
_NO_PENALTY = re.compile(r"without\s+(?:any\s+)?(?:penalty|liability|termination\s+fee)", re.I)
_EXCERPT = re.compile(
    r"\[(\d+)\]\s+document=(.*?)\s+\|\s+section=(.*?)\s+\|\s+pages=([^\n]*)\n(?:TEXT:\s*)?(.*?)"
    r"(?=\n\n\[\d+\]\s+document=|\n\nReturn the JSON|\Z)", re.S)
_EXPOSURE = re.compile(r"\d+(?:\.\d+)?\s?%|(?:US\$|USD|[$€£₹])\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|m|b)\b)?", re.I)


def _sentences(text: str) -> List[str]:
    lines = text.split("\n")
    if len(lines) > 1 and not lines[0].rstrip().endswith((".", ";", ":")):
        lines = lines[1:]  # drop the section-header prefix line added by the chunker
    return [s.strip() for line in lines for s in _SENTENCE_SPLIT.split(line) if s.strip()]


def _window(sentence: str, pattern: re.Pattern, limit: int = 40) -> str:
    words = sentence.split()
    if len(words) <= limit:
        return " ".join(words)
    m = pattern.search(sentence)
    idx = len(sentence[:m.start()].split()) if m else 0
    start = max(0, min(idx - 10, len(words) - limit))
    return " ".join(words[start:start + limit])


class RuleTemplateEngine:
    """Deterministic CPU-only stand-in for the quantized LLM: copies verbatim sentences into the JSON schema."""

    def __init__(self) -> None:
        """Expose the attributes Stage 2 expects of an LLM client."""
        self.cfg = SimpleNamespace(model_name="rule-template-engine-v1", backend="rules", url="local://rule-engine")

    def ping(self) -> bool:
        """Always available."""
        return True

    def chat_json(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Return the extraction JSON for a Stage 2 prompt (QUESTION + numbered EXCERPTS)."""
        q = re.search(r"QUESTION:\s*(.*?)\s*EXCERPTS:", user, re.S)
        question = (q.group(1) if q else "").casefold()
        wanted = [c for c, hint in _QUESTION_HINTS.items() if re.search(hint, question)] or list(_CATEGORY_RULES)
        risks: List[Dict[str, Any]] = []
        used: set = set()
        for category in wanted:
            pattern, title, action = _CATEGORY_RULES[category]
            taken = 0
            for ref, doc, section, pages, text in ((int(m[0]), m[1], m[2], m[3], m[4]) for m in _EXCERPT.findall(user)):
                for sentence in _sentences(text):
                    if taken >= 2 or not pattern.search(sentence):
                        continue
                    quote = _window(sentence, pattern)
                    if quote in used:
                        continue
                    used.add(quote)
                    taken += 1
                    pct = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s?%", quote)]
                    severity = ("HIGH" if category in ("CHANGE_OF_CONTROL", "TERMINATION_RIGHTS") and _NO_PENALTY.search(quote)
                                else "HIGH" if pct and max(pct) >= 15 else "MEDIUM" if category != "FINANCIAL_COVENANT" else "LOW")
                    exposure = _EXPOSURE.search(quote)
                    risks.append({
                        "category": category, "severity": severity, "title": f"{title} in {section[:60] or 'document'}",
                        "summary": quote, "evidence_quote": quote, "source_ids": [ref], "source_document": doc.strip(),
                        "source_section": section.strip(), "source_pages": pages.strip(),
                        "financial_exposure": exposure.group(0) if exposure else None, "recommended_action": action,
                    })
        return json.dumps({"risks": risks})


# ============================================================================ loader
@dataclass
class LoadedModel:
    """Backend chosen by the loader plus its footprint report."""

    backend: Any
    report: QuantReport
    live: bool


class QuantizedModelLoader:
    """Connect to a local Q4_K_M model via Ollama, else fall back to the rule/template engine."""

    def __init__(self, model: str = TARGET_MODEL, allowed_hosts: Sequence[str] = (), force_fallback: bool = False) -> None:
        """Configure the target tag and whether to skip Ollama entirely."""
        self.model, self.allowed_hosts, self.force_fallback = model, tuple(allowed_hosts), force_fallback

    def load(self) -> LoadedModel:
        """Return a usable backend; never raises for an unreachable server."""
        cfg = dataclasses.replace(LLMConfig(), model=self.model)
        notes: List[str] = []
        if self.force_fallback:
            notes.append("fallback forced by caller")
        elif cfg.backend != "ollama":
            notes.append(f"backend '{cfg.backend}' is not inspectable here (GGUF inspection is Ollama-only)")
        else:
            client = LocalLLMClient(cfg, self.allowed_hosts)
            if not client.ping():
                notes.append(f"Ollama not reachable at {cfg.url}")
            else:
                try:
                    report = inspect_ollama(client, self.model)
                except LLMUnavailableError as exc:
                    report, notes = None, notes + [f"inspection failed: {exc}"]
                if report is not None:
                    LOG.info("quantized model ready: %s (%s, %.2f GB on disk)", report.model, report.quantization, report.disk_gb)
                    return LoadedModel(client, report, True)
                notes.append(f"model '{self.model}' is not pulled: run `ollama pull {self.model}`")
        LOG.warning("using deterministic rule/template fallback: %s", "; ".join(notes))
        notes.append("Footprint figures below are REFERENCE values for Qwen2.5-3B, not measurements.")
        notes.append("The rule engine runs in a few MB of RAM and extracts verbatim sentences only.")
        report = _build_report("rule-engine-fallback", self.model, "Q4_K_M (reference)", REFERENCE["params_b"],
                               REFERENCE["q4km_gb"], None, False, notes)
        report.fp16_gb = REFERENCE["fp16_gb"]
        report.reduction_x = round(REFERENCE["fp16_gb"] / REFERENCE["q4km_gb"], 2)
        report.reduction_pct = round(100 * (1 - REFERENCE["q4km_gb"] / REFERENCE["fp16_gb"]), 1)
        return LoadedModel(RuleTemplateEngine(), report, False)


def quantize_demo(rows: int = 256, cols: int = 256, block: int = 32, seed: int = 0) -> Dict[str, Any]:
    """Block-wise 4-bit (min/scale) quantization of a random weight matrix with nibble packing, NumPy only."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.02, (rows, cols)).astype(np.float32)
    blocks = w.reshape(-1, block)
    lo, hi = blocks.min(axis=1, keepdims=True), blocks.max(axis=1, keepdims=True)
    scale = np.maximum((hi - lo) / 15.0, 1e-12)
    q = np.clip(np.round((blocks - lo) / scale), 0, 15).astype(np.uint8)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)  # two 4-bit weights per byte
    deq = q.astype(np.float32) * scale + lo
    err = deq - blocks
    fp16_bytes = w.size * 2
    q_bytes = packed.nbytes + blocks.shape[0] * 4  # + fp16 scale and fp16 min per block
    return {
        "weights": int(w.size), "block_size": block, "fp16_bytes": int(fp16_bytes), "quantized_bytes": int(q_bytes),
        "compression_x": round(fp16_bytes / q_bytes, 2), "bits_per_weight": round(q_bytes * 8 / w.size, 2),
        "rmse": float(np.sqrt(np.mean(err ** 2))), "snr_db": round(float(10 * np.log10(np.mean(blocks ** 2) / np.mean(err ** 2))), 1),
        "note": "Per-block fp16 min+scale costs 1 extra bit/weight here; Q4_K's 6-bit scales bring the real cost to about 4.5 bits.",
    }


# ======================================================================= DPO pair builder
def _norm(text: str) -> str:
    return "".join(_WORD.findall(str(text).casefold()))


def _bump_numbers(text: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group(1):
            return m.group(1)  # leave placeholders such as [CURRENCY_1A2B3C] intact
        s = m.group(2)
        return f"{float(s) * 1.5:.{len(s.split('.')[1])}f}" if "." in s else str(int(int(s) * 1.5) + 1)

    return re.sub(rf"({_PLACEHOLDER.pattern})|(\d+(?:\.\d+)?)", repl, text)


def synthesize_rejected(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Hard negative for an approved finding: wrong numbers if any exist, else an ellipsis-truncated paraphrase."""
    bad = dict(item)
    blob = f"{item.get('summary', '')} {item.get('evidence_quote', '')}"
    if re.search(r"\d", _PLACEHOLDER.sub("", blob)):
        for key in ("summary", "evidence_quote", "financial_exposure"):
            if bad.get(key):
                bad[key] = _bump_numbers(str(bad[key]))
        return bad, "synthetic_number_perturbation"
    words = str(item.get("evidence_quote", "")).split()
    bad["evidence_quote"] = " ".join(words[:max(3, len(words) // 2)]) + " ..."
    bad["summary"] = "Possibly relevant: " + str(item.get("summary", ""))
    return bad, "synthetic_ellipsis_truncation"


def _as_output(item: Dict[str, Any]) -> str:
    return json.dumps({"risks": [item]}, ensure_ascii=False)


def build_pair_from_resolution(
    finding: Dict[str, Any], action: str, override: Optional[Dict[str, str]] = None,
    rejected_text: Optional[str] = None, proxy: Any = None,
) -> Dict[str, str]:
    """Format a reviewer decision as a DPO pair.

    approve  -> chosen = the model's finding; rejected = supplied text or a synthetic hard negative
    override -> chosen = the finding patched with the reviewer's fields; rejected = the model's finding
    reject   -> chosen = abstention ``{"risks": []}``; rejected = the model's finding

    Raises:
        ValueError: on unknown actions/fields or an override quote that is not verbatim in the prompt.
    """
    if action not in ("approve", "override", "reject"):
        raise ValueError("action must be approve, override or reject")
    anon = (lambda s: proxy.anonymize(s)) if proxy is not None else (lambda s: s)
    prompt = finding["prompt"]
    item = json.loads(finding["model_output"])["risks"][0]
    model_out = _as_output(item)
    if action == "approve":
        if rejected_text:
            return {"prompt": prompt, "chosen": model_out, "rejected": anon(rejected_text), "rejected_source": "human_supplied", "human_output": ""}
        bad, source = synthesize_rejected(item)
        return {"prompt": prompt, "chosen": model_out, "rejected": _as_output(bad), "rejected_source": source, "human_output": ""}
    if action == "reject":
        return {"prompt": prompt, "chosen": json.dumps({"risks": []}), "rejected": model_out,
                "rejected_source": "model_output_rejected_by_human", "human_output": ""}
    if not override:
        raise ValueError("override action requires an 'override' object")
    unknown = set(override) - _OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"unsupported override fields: {sorted(unknown)}")
    fixed = dict(item)
    for key, value in override.items():
        fixed[key] = anon(str(value)) if key not in ("category", "severity") else str(value).strip().upper()
    if fixed["category"] not in _ex.RISK_CATEGORIES or fixed["severity"] not in _ex.SEVERITIES:
        raise ValueError("invalid category or severity")
    if _norm(fixed["evidence_quote"]) not in _norm(prompt):
        raise ValueError("override evidence_quote must be a verbatim span of the excerpts in the prompt")
    return {"prompt": prompt, "chosen": _as_output(fixed), "rejected": model_out,
            "rejected_source": "model_output_overridden_by_human", "human_output": _as_output(fixed)}


class DPOStore:
    """Append-only preference store: JSONL (exact 3 keys) plus SQLite (with provenance metadata)."""

    SCHEMA = ("CREATE TABLE IF NOT EXISTS dpo_pairs (pair_id TEXT PRIMARY KEY, created_at TEXT, finding_id TEXT, action TEXT, "
              "reviewer TEXT, note TEXT, rejected_source TEXT, prompt TEXT, chosen TEXT, rejected TEXT)")

    def __init__(self, jsonl_path: Path, db_path: Path) -> None:
        """Open/create both stores."""
        self.jsonl_path, self._lock = Path(jsonl_path), threading.RLock()
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(self.SCHEMA)
            self._db.commit()

    def add_pair(self, prompt: str, chosen: str, rejected: str, *, finding_id: str = "", action: str = "",
                 reviewer: str = "", note: str = "", rejected_source: str = "") -> Tuple[bool, str]:
        """Insert a pair; returns ``(inserted, pair_id)`` and skips exact duplicates."""
        if not (prompt and chosen and rejected) or chosen == rejected:
            raise ValueError("prompt, chosen and rejected must be non-empty and chosen != rejected")
        pair_id = hashlib.sha256("\x1f".join((prompt, chosen, rejected)).encode()).hexdigest()[:16]
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO dpo_pairs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pair_id, time.strftime("%Y-%m-%dT%H:%M:%S%z"), finding_id, action, reviewer, note, rejected_source, prompt, chosen, rejected))
            self._db.commit()
            inserted = cur.rowcount == 1
            if inserted:
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
        return inserted, pair_id

    def fetch_pairs(self, limit: int = 256) -> List[Dict[str, str]]:
        """Most recent pairs (prompt/chosen/rejected only)."""
        with self._lock:
            rows = self._db.execute("SELECT prompt, chosen, rejected FROM dpo_pairs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        """Counts by action and rejected-source, plus file location."""
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM dpo_pairs").fetchone()[0]
            by_action = dict(self._db.execute("SELECT action, COUNT(*) FROM dpo_pairs GROUP BY action").fetchall())
            by_src = dict(self._db.execute("SELECT rejected_source, COUNT(*) FROM dpo_pairs GROUP BY rejected_source").fetchall())
        return {"pairs": total, "by_action": by_action, "by_rejected_source": by_src, "jsonl": str(self.jsonl_path)}


# ============================================================== exact DPO (NumPy, CPU)
VOCAB = 4096
_TOK = re.compile(r"\w+|[^\w\s]")


def _counts(text: str) -> Tuple[np.ndarray, int]:
    ids = [zlib.crc32(t.lower().encode()) % VOCAB for t in _TOK.findall(text)]
    return np.bincount(ids, minlength=VOCAB).astype(np.float64), len(ids)


def _log_softmax(theta: np.ndarray) -> np.ndarray:
    m = theta.max()
    return theta - (m + np.log(np.exp(theta - m).sum()))


def _dpo_loss_grad(theta: np.ndarray, ref_logp: np.ndarray, cw: np.ndarray, nw: np.ndarray,
                   cl: np.ndarray, nl: np.ndarray, beta: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """Mean DPO loss -log sigmoid(beta*[(logpi_w - logref_w) - (logpi_l - logref_l)]), its gradient and margins z."""
    logp = _log_softmax(theta)
    z = beta * ((cw @ logp - cw @ ref_logp) - (cl @ logp - cl @ ref_logp))
    loss = float(np.mean(np.logaddexp(0.0, -z)))
    p = np.exp(logp)
    coef = -beta * np.exp(-np.logaddexp(0.0, z))  # dL/dz = -sigmoid(-z)
    g = (cw - cl) - (nw - nl)[:, None] * p[None, :]  # grad of logpi(w) - logpi(l) w.r.t. theta
    return loss, (coef[:, None] * g).mean(axis=0), z


def grad_check(seed: int = 0) -> float:
    """Max abs difference between the analytic gradient and central finite differences."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.3, VOCAB)
    ref = _log_softmax(rng.normal(0, 0.3, VOCAB))
    cw, nw = _counts("The three largest customers may terminate without penalty after a change of control.")
    cl, nl = _counts("Customers might consider leaving at some point.")
    args = (ref, cw[None], np.array([nw], dtype=float), cl[None], np.array([nl], dtype=float), 0.1)
    _, grad, _ = _dpo_loss_grad(theta, *args)
    worst = 0.0
    for i in rng.choice(np.flatnonzero((cw + cl) > 0), size=6, replace=False):
        e = np.zeros(VOCAB)
        e[i] = 1e-6
        numeric = (_dpo_loss_grad(theta + e, *args)[0] - _dpo_loss_grad(theta - e, *args)[0]) / 2e-6
        worst = max(worst, abs(numeric - grad[i]))
    return worst


def demo_pairs() -> List[Dict[str, str]]:
    """Built-in pairs for a live demo when no human pairs exist yet (never written to the store)."""
    samples = [
        ("CHANGE_OF_CONTROL", "HIGH", "Three customers representing approximately 40% of total revenue may terminate without penalty following a change of control."),
        ("TERMINATION_RIGHTS", "HIGH", "Customer may terminate this Agreement within 180 days after a Change of Control without penalty or termination fee."),
        ("CUSTOMER_CONCENTRATION", "HIGH", "The three largest customers accounted for 17%, 13% and 10% of total revenue."),
        ("LIABILITY_INDEMNITY", "MEDIUM", "Neither party's aggregate liability shall exceed the fees paid in the twelve months preceding the claim."),
        ("FINANCIAL_COVENANT", "LOW", "We had no borrowings outstanding under our revolving credit facility."),
        ("LIABILITY_INDEMNITY", "MEDIUM", "Supplier shall maintain cyber liability insurance of not less than 10,000,000 per occurrence."),
    ]
    out = []
    for i, (cat, sev, quote) in enumerate(samples, start=1):
        item = {"category": cat, "severity": sev, "title": cat.title(), "summary": quote, "evidence_quote": quote, "source_ids": [i],
                "source_document": "MSA_demo.pdf", "source_section": "demo", "source_pages": "1", "financial_exposure": None,
                "recommended_action": "Review."}
        bad, _ = synthesize_rejected(item)
        out.append({"prompt": f"QUESTION: demo {i}\nEXCERPTS:\n[{i}] {quote}", "chosen": _as_output(item), "rejected": _as_output(bad)})
    return out


def _trl_ready() -> bool:
    try:
        import torch  # noqa: F401
        import peft, trl  # noqa: F401, E401
        return bool(torch.cuda.is_available() and os.getenv("APEX_DPO_BASE_MODEL"))
    except Exception:  # noqa: BLE001
        return False


def lora_trainer_config(base_model: str = "", use_4bit: bool = False) -> Dict[str, Any]:
    """LoRA + DPO hyper-parameters sized for <4 GB VRAM (default base: Qwen2.5-0.5B-Instruct in fp16)."""
    return {
        "base_model": base_model or os.getenv("APEX_DPO_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "quantization": {"load_in_4bit": use_4bit, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                         "bnb_4bit_compute_dtype": "float16"},
        "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "dpo": {"beta": 0.1, "learning_rate": 5e-6, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
                "max_steps": 1, "max_length": 768, "gradient_checkpointing": True, "logging_steps": 1, "report_to": "none"},
        "vram_note": "0.5B in fp16 fits comfortably under 4 GB. A 3B base in NF4 (use_4bit=True) is borderline at this sequence length. "
                     "Train on HF weights, then merge the adapter and convert to GGUF (llama.cpp) for Ollama.",
    }


def train_with_trl(pairs: Sequence[Dict[str, str]], output_dir: str = "apex_data/dpo_lora", base_model: str = "",
                   beta: float = 0.1, max_steps: int = 1, use_4bit: bool = False) -> Dict[str, Any]:
    """Real LoRA DPO via ``trl.DPOTrainer`` + ``peft`` (local files only; needs a CUDA GPU)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = lora_trainer_config(base_model, use_4bit)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: Dict[str, Any] = {"local_files_only": True, "torch_dtype": torch.float16}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    dataset = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    allowed = {f.name for f in dataclasses.fields(DPOConfig)}  # TRL renames args between releases
    args = DPOConfig(**{k: v for k, v in {**cfg["dpo"], "beta": beta, "max_steps": max_steps, "output_dir": output_dir}.items() if k in allowed})
    trainer_kwargs: Dict[str, Any] = dict(model=model, args=args, train_dataset=dataset, peft_config=LoraConfig(**cfg["lora"]))
    trainer_kwargs["processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else "tokenizer"] = tok
    trainer = DPOTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(output_dir)
    return {"backend": "trl+peft", "base_model": cfg["base_model"], "pairs": len(pairs), "steps": max_steps,
            "train_loss": float(result.training_loss), "adapter_dir": output_dir, "config": cfg}


def run_dpo_step(pairs: Optional[Sequence[Dict[str, str]]] = None, *, steps: int = 60, beta: float = 0.1, lr: float = 0.5,
                 backend: str = "auto", allow_demo_seed: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Run DPO on the given pairs: TRL+LoRA if a GPU and ``APEX_DPO_BASE_MODEL`` are available, else exact NumPy DPO.

    The NumPy path optimises a prompt-independent hashed-unigram policy against a frozen reference (a smoothed
    unigram model of the pairs). At initialisation policy == reference, so the loss equals ln 2 = 0.6931.
    """
    seeded = False
    if not pairs:
        if not allow_demo_seed:
            raise ValueError("no preference pairs available")
        pairs, seeded = demo_pairs(), True
    if backend == "trl" or (backend == "auto" and _trl_ready()):
        out = train_with_trl(pairs, beta=beta, max_steps=max(1, min(steps, 5)))
        out["demo_seeded"] = seeded
        return out
    started = time.perf_counter()
    cw, nw = zip(*(_counts(p["chosen"]) for p in pairs))
    cl, nl = zip(*(_counts(p["rejected"]) for p in pairs))
    cw, cl, nw, nl = np.array(cw), np.array(cl), np.array(nw, dtype=float), np.array(nl, dtype=float)
    total = cw.sum(axis=0) + cl.sum(axis=0) + 1.0
    ref_logp = np.log(total / total.sum())  # frozen reference policy
    theta = ref_logp.copy()
    curve: List[float] = []
    first: Optional[Tuple[float, np.ndarray]] = None
    for _ in range(steps):
        loss, grad, z = _dpo_loss_grad(theta, ref_logp, cw, nw, cl, nl, beta)
        if first is None:
            first = (loss, z)
        curve.append(round(loss, 5))
        theta = theta - lr * grad
    loss_end, _, z_end = _dpo_loss_grad(theta, ref_logp, cw, nw, cl, nl, beta)
    z0 = first[1]
    return {
        "backend": "numpy-cpu-exact-dpo", "pairs": len(pairs), "demo_seeded": seeded, "steps": steps, "beta": beta, "lr": lr,
        "loss_initial": round(first[0], 5), "loss_final": round(loss_end, 5), "ln2_check": round(float(np.log(2)), 5),
        "reward_accuracy_before": float(np.mean(z0 > 0)), "reward_accuracy_after": float(np.mean(z_end > 0)),
        "mean_reward_margin_before": round(float(z0.mean()), 5), "mean_reward_margin_after": round(float(z_end.mean()), 5),
        "loss_curve": curve[:: max(1, steps // 20)], "elapsed_s": round(time.perf_counter() - started, 4),
        "note": "Exact DPO objective on a toy unigram policy; Qwen/Ollama weights are unchanged. "
                "Set APEX_DPO_BASE_MODEL on a GPU host for real LoRA training.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``--selftest`` | ``--inspect``."""
    parser = argparse.ArgumentParser(description="Apex quantization + DPO utilities.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.inspect:
        print(json.dumps(QuantizedModelLoader().load().report.to_dict(), indent=2))
        return 0
    if args.selftest:
        print(json.dumps({"quantize_demo": quantize_demo(), "gradient_check_max_abs_err": grad_check(),
                          "dpo": run_dpo_step(backend="numpy")}, indent=2))
        return 0
    parser.error("choose --selftest or --inspect")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())