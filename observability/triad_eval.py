"""Real-time RAG Triad evaluator (TruLens/Ragas pattern, fully local).

Scores, per retrieved snippet and in aggregate:
    * Context Relevance : calibrated cosine(query, snippet) from the local BGE embedder.
    * Groundedness      : per claim, verbatim-quote containment + numeric verification +
                          embedding support + optional local-LLM entailment judge, evaluated
                          against the snippets the claim cites. No valid citation scores 0.
    * Answer Relevance  : calibrated cosine(query, claims) blended with a local-LLM judge.
...
``gate`` evaluates every claim individually. A claim is retained when its groundedness is at least
``claim_floor`` (0.50) AND its evidence quote is verified in the cited excerpt (fuzzy, normalized match).
Aggregate groundedness below ``groundedness_threshold`` (0.75) marks the answer PARTIAL instead of blocking it;
``LowConfidenceRetractionException`` is raised only when findings exist and none is retained.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from llm.extractor import LocalLLMClient, RetrievedContext, RiskFinding
from pipeline.ingest import BGE_QUERY_PREFIX
from security.network_guard import NetworkViolation, local_processing_guarantee

LOG = logging.getLogger("apex.triad")
_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


class Embedder(Protocol):
    """Embedding contract (satisfied by Stage 1 ``LocalEmbedder``)."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return L2-normalized embeddings."""


class LowConfidenceRetractionException(RuntimeError):
    """Raised when groundedness is too low; the answer must not be released."""

    def __init__(self, message: str, report: "TriadReport") -> None:
        """Carry the score report (scores only, never the blocked answer text)."""
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class TriadConfig:
    """Evaluator settings."""

    groundedness_threshold: float = 0.75   # aggregate "clean pass" mark; below it the answer is PARTIAL
    claim_floor: float = 0.50              # a claim scoring below this is dropped
    retain_min_quote: float = 0.50         # verified-quote requirement for retention (0.0 = score-only)
    citation_recovery_min: float = 0.60    # fuzzy quote match needed to recover/correct a citation
    sim_floor: float = 0.55                # BGE cosine similarities are compressed into ~[0.55, 1]
    sim_ceiling: float = 0.85
    use_llm_judge: bool = True
    w_lexical_judged: float = 0.45
    w_embed_judged: float = 0.15
    w_judge: float = 0.40
    w_lexical_plain: float = 0.70
    w_embed_plain: float = 0.30
    quote_weight: float = 0.75             # share of lexical score from the verbatim quote (rest: numbers)
    relevance_warn_below: float = 0.50
    metrics_path: Optional[Path] = None


# ---------------------------------------------------------------- score helpers
def calibrate(cosine: float, floor: float, ceiling: float) -> float:
    """Map a raw cosine similarity onto [0, 1] using the embedder's practical range."""
    return float(min(1.0, max(0.0, (cosine - floor) / (ceiling - floor))))


_CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u00a0": " ", "\u00ad": "",
})


def _tokens(text: str) -> List[str]:
    """Normalize (NFKC, smart quotes/dashes, case) and split into alphanumeric tokens; whitespace/punctuation-insensitive."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    return _WORD.findall(text.casefold())


def fuzzy_quote_score(quote: str, context: str) -> float:
    """How much of ``quote`` appears, in order, in ``context`` (0..1), tolerant of spacing, case and punctuation.

    1.0 when the normalized quote is a substring (also after removing all spaces, which absorbs PDF line-break
    artefacts). Otherwise the share of quote tokens covered by in-order ``SequenceMatcher`` blocks of >= 2 tokens,
    so scattered stop-words cannot inflate the score.
    """
    q, c = _tokens(quote), _tokens(context)
    if not q or not c:
        return 0.0
    if "".join(q) in "".join(c):
        return 1.0
    blocks = SequenceMatcher(None, q, c, autojunk=False).get_matching_blocks()
    covered = sum(b.size for b in blocks if b.size >= min(2, len(q)))
    return min(1.0, covered / len(q))


quote_containment = fuzzy_quote_score  # backward-compatible name


def resolve_citations(
    finding: RiskFinding, contexts: Sequence[RetrievedContext], index_of: Dict[int, int], min_support: float = 0.60,
) -> Tuple[List[int], Optional[str]]:
    """Return context indices backing a finding, recovering or correcting wrong ``source_ids`` via the quote."""
    cited = [index_of[r] for r in finding.source_ids if r in index_of]
    quote = (finding.evidence_quote or "").strip()
    if not quote:
        return cited, (None if cited else "no valid citation")
    support = {j: fuzzy_quote_score(quote, c.text) for j, c in enumerate(contexts)}
    if cited and max(support[j] for j in cited) >= min_support:
        return cited, None
    best = max(support, key=support.get)
    if support[best] >= min_support:
        return [best], ("citation corrected from quote match" if cited else "citation recovered from quote match")
    return cited, (None if cited else "no valid citation")


def _numbers(text: str) -> set:
    return {m.replace(",", "") for m in _NUM.findall(text)}


def numeric_support(claim_text: str, context_text: str) -> float:
    """Fraction of numbers asserted in the claim that literally appear in the context (1.0 if none)."""
    claimed = _numbers(claim_text)
    if not claimed:
        return 1.0
    return len(claimed & _numbers(context_text)) / len(claimed)


# ------------------------------------------------------------------ data model
@dataclass
class SnippetScore:
    """Triad scores for one retrieved snippet (groundedness is None when no claim cites it)."""

    ref: int
    chunk_id: str
    context_relevance: float
    groundedness: Optional[float]
    answer_relevance: float


@dataclass
class ClaimScore:
    """Groundedness breakdown for one extracted finding."""

    index: int
    cited_refs: List[int]
    lexical: float
    embedding: float
    judge: Optional[float]
    score: float
    issue: Optional[str] = None
    quote_support: float = 0.0


@dataclass
class TriadReport:
    """Aggregate and per-snippet triad scores plus the granular claim-retention outcome."""

    query_id: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    threshold: float
    passed: bool          # True when the answer may be released (vacuous, or >= 1 claim retained)
    vacuous: bool
    judge_used: bool
    snippets: List[SnippetScore] = field(default_factory=list)
    claims: List[ClaimScore] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    retained: List[int] = field(default_factory=list)   # finding indices that survive
    dropped: List[int] = field(default_factory=list)
    partial: bool = False                               # released with dropped claims or aggregate < threshold
    retained_groundedness: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view (contains scores only)."""
        return asdict(self)


# ------------------------------------------------------------------- evaluator
class TriadEvaluator:
    """Scores each answer in real time and enforces the groundedness gate."""

    def __init__(self, embedder: Embedder, llm: Optional[LocalLLMClient] = None, cfg: Optional[TriadConfig] = None) -> None:
        """Bind the local embedder and (optionally) the local LLM used as an entailment judge."""
        self.embedder, self.llm, self.cfg = embedder, llm, cfg or TriadConfig()

    # --- judges (local LLM, loopback only) ---
    def _judge(self, system: str, user: str) -> Optional[float]:
        if not (self.cfg.use_llm_judge and self.llm):
            return None
        try:
            data = json.loads(self.llm.chat_json(system, user, max_tokens=120, temperature=0.0))
            return float(min(1.0, max(0.0, float(data["score"]))))
        except NetworkViolation:
            raise
        except Exception as exc:  # noqa: BLE001  (judge failure degrades to deterministic scoring)
            LOG.warning("LLM judge unavailable (%s); using deterministic scoring for this item", exc)
            return None

    def _judge_claim(self, finding: RiskFinding, excerpts: str) -> Optional[float]:
        system = (
            "You are a strict fact-checker. Decide whether the CLAIM is fully supported by the EXCERPTS. "
            "Reply with JSON only: {\"score\": <0.0-1.0>, \"reason\": \"<short>\"}. "
            "1.0 = fully supported, 0.5 = partly supported, 0.0 = unsupported or contradicted."
        )
        user = f"EXCERPTS:\n{excerpts[:6000]}\n\nCLAIM:\n{finding.title}. {finding.summary}\nExposure: {finding.financial_exposure or 'n/a'}"
        return self._judge(system, user)

    def _judge_answer(self, query: str, findings: Sequence[RiskFinding]) -> Optional[float]:
        system = (
            "You grade answer relevance. Decide whether the FINDINGS directly address the QUESTION. "
            "Reply with JSON only: {\"score\": <0.0-1.0>, \"reason\": \"<short>\"}."
        )
        body = "\n".join(f"- {f.title}: {f.summary}" for f in findings)
        return self._judge(system, f"QUESTION:\n{query}\n\nFINDINGS:\n{body[:4000]}")

    # --- core scoring ---
    def evaluate(self, query: str, findings: Sequence[RiskFinding], contexts: Sequence[RetrievedContext]) -> TriadReport:
        """Compute the RAG Triad plus per-claim retention without enforcing the gate."""
        cfg = self.cfg
        if not contexts:
            raise ValueError("cannot evaluate an answer without retrieved contexts")
        claim_texts = [f"{f.title}. {f.summary}".strip() for f in findings]
        texts = [BGE_QUERY_PREFIX + query] + [c.text for c in contexts] + claim_texts
        with local_processing_guarantee(allow_loopback=False):
            vecs = np.asarray(self.embedder.embed(texts), dtype=np.float64)
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        q_vec, ctx_vecs, claim_vecs = vecs[0], vecs[1:1 + len(contexts)], vecs[1 + len(contexts):]

        ctx_rel = [calibrate(float(v @ q_vec), cfg.sim_floor, cfg.sim_ceiling) for v in ctx_vecs]
        index_of = {c.ref: i for i, c in enumerate(contexts)}

        claims: List[ClaimScore] = []
        judge_used = False
        for i, finding in enumerate(findings):
            cited, issue = resolve_citations(finding, contexts, index_of, cfg.citation_recovery_min)
            if not cited:
                claims.append(ClaimScore(i, [], 0.0, 0.0, None, 0.0, issue or "no valid citation", 0.0))
                continue
            joined = "\n".join(contexts[j].text for j in cited)
            quote = max(fuzzy_quote_score(finding.evidence_quote, contexts[j].text) for j in cited)
            numbers = numeric_support(f"{finding.summary} {finding.financial_exposure or ''}", joined)
            lexical = cfg.quote_weight * quote + (1.0 - cfg.quote_weight) * numbers
            emb = max(calibrate(float(claim_vecs[i] @ ctx_vecs[j]), cfg.sim_floor, cfg.sim_ceiling) for j in cited)
            judge = self._judge_claim(finding, joined)
            if judge is not None:
                judge_used = True
                score = cfg.w_lexical_judged * lexical + cfg.w_embed_judged * emb + cfg.w_judge * judge
            else:
                score = cfg.w_lexical_plain * lexical + cfg.w_embed_plain * emb
            if issue is None and quote < 0.6:
                issue = "evidence quote not found in cited excerpt"
            claims.append(ClaimScore(i, [contexts[j].ref for j in cited], lexical, emb, judge, float(score), issue, quote))

        # --- granular retention: each claim stands or falls on its own evidence ---
        vacuous = not findings
        retained = [c.index for c in claims if c.score >= cfg.claim_floor and c.quote_support >= cfg.retain_min_quote]
        kept = set(retained)
        dropped = [c.index for c in claims if c.index not in kept]
        groundedness = 1.0 if vacuous else float(np.mean([c.score for c in claims]))
        kept_scores = [c.score for c in claims if c.index in kept]
        retained_g = float(np.mean(kept_scores)) if kept_scores else None

        if vacuous:
            answer_rel = 1.0
        elif not retained:
            answer_rel = 0.0
        else:
            emb_rel = float(np.mean([calibrate(float(claim_vecs[i] @ q_vec), cfg.sim_floor, cfg.sim_ceiling) for i in retained]))
            judged = self._judge_answer(query, [findings[i] for i in retained])
            answer_rel = emb_rel if judged is None else 0.5 * emb_rel + 0.5 * judged

        per_ref: Dict[int, List[float]] = {}
        for claim in claims:
            for ref in claim.cited_refs:
                per_ref.setdefault(ref, []).append(claim.score)
        snippets = [
            SnippetScore(c.ref, c.chunk_id, ctx_rel[i], float(np.mean(per_ref[c.ref])) if c.ref in per_ref else None, answer_rel)
            for i, c in enumerate(contexts)
        ]
        mean_ctx = float(np.mean(ctx_rel))

        reasons: List[str] = []
        if groundedness < cfg.groundedness_threshold:
            reasons.append(f"aggregate groundedness {groundedness:.2f} < {cfg.groundedness_threshold:.2f}")
        by_index = {c.index: c for c in claims}
        for i in dropped:
            c = by_index[i]
            reasons.append(f"claim {i} dropped: score {c.score:.2f}, quote support {c.quote_support:.2f}"
                           + (f" ({c.issue})" if c.issue else ""))
        warnings: List[str] = []
        if mean_ctx < cfg.relevance_warn_below:
            warnings.append(f"low mean context relevance {mean_ctx:.2f}")
        if answer_rel < cfg.relevance_warn_below:
            warnings.append(f"low answer relevance {answer_rel:.2f}")
        if retained and groundedness < cfg.groundedness_threshold:
            warnings.append("released below the aggregate threshold on the strength of individually verified claims")
        return TriadReport(
            query_id=hashlib.sha256(query.encode()).hexdigest()[:12], context_relevance=mean_ctx,
            groundedness=groundedness, answer_relevance=answer_rel, threshold=cfg.groundedness_threshold,
            passed=vacuous or bool(retained), vacuous=vacuous, judge_used=judge_used, snippets=snippets,
            claims=claims, warnings=warnings, reasons=reasons, retained=retained, dropped=dropped,
            partial=bool(dropped) or groundedness < cfg.groundedness_threshold, retained_groundedness=retained_g,
        )

    def gate(self, query: str, findings: Sequence[RiskFinding], contexts: Sequence[RetrievedContext]) -> TriadReport:
        """Evaluate, log, and block only if findings exist and none survives verification.

        Callers must release only ``report.retained`` findings (``LocalRiskExtractor.extract`` does this).
        """
        report = self.evaluate(query, findings, contexts)
        self._log(report)
        if findings and not report.retained:
            raise LowConfidenceRetractionException(
                "Output retracted by RAG Triad gate (no claim reached groundedness "
                f">= {self.cfg.claim_floor:.2f} with a verified quote): " + "; ".join(report.reasons), report
            )
        return report

    def _log(self, report: TriadReport) -> None:
        LOG.info(
            "triad q=%s ctx_rel=%.2f grounded=%.2f ans_rel=%.2f retained=%d dropped=%d partial=%s",
            report.query_id, report.context_relevance, report.groundedness, report.answer_relevance,
            len(report.retained), len(report.dropped), report.partial,
        )
        path = self.cfg.metrics_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "query_id": report.query_id,
                "context_relevance": report.context_relevance, "groundedness": report.groundedness,
                "answer_relevance": report.answer_relevance, "passed": report.passed, "partial": report.partial,
                "retained": len(report.retained), "dropped": len(report.dropped), "claims": len(report.claims),
                "claim_scores": [round(c.score, 3) for c in report.claims],
                "quote_support": [round(c.quote_support, 3) for c in report.claims],
                "issues": [c.issue for c in report.claims], "judge_used": report.judge_used,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError as exc:
            LOG.warning("could not write triad metrics: %s", exc)