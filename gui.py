#!/usr/bin/env python3
"""AegisDeal / Operation Apex, Stage 4: Gradio executive dashboard.

Tab 1  Data Room & Air-Gap Security Setup   (generate/upload PDFs, ingest, network audit, PII inspector)
Tab 2  RAG Triad & Data Drift Monitor        (local RAG audit, triad scores, drift health bars)
Tab 3  Multi-Agent Consensus & Final Memo    (agent dialogue, revised bid, executive memo)

Wraps Stages 1-3 directly: data.synthetic_gen, pipeline.ingest, security.proxy, security.network_guard,
llm.extractor, observability.*, agents.*, report.synthesizer and main.drift_gate.
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no phoning home from an air-gapped demo

import security.network_guard as ng  # noqa: E402  (sets offline env; must precede ML imports)
from security.network_guard import NetworkViolation, local_processing_guarantee, verify_guard_blocks_egress  # noqa: E402

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter, deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple  # noqa: E402

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from agents.core import (  # noqa: E402
    AgentError, DealTerms, EgressLedger, MockWebSearchTool, build_brief, stage1_canaries, stage1_known_entities,
)
from agents.negotiation import final_offer, open_negotiation, run_round  # noqa: E402
from agents.nodes import AuditLegalAgent, FinancialAuditAgent, MarketIntelligenceAgent  # noqa: E402
from config import ApexConfig  # noqa: E402
from data.synthetic_gen import generate_all  # noqa: E402
from llm.extractor import (  # noqa: E402
    AUDIT_QUERIES, HybridRetriever, LLMConfig, LocalLLMClient, LocalRiskExtractor,
)
from main import drift_gate, load_or_create_proxy_key  # noqa: E402
from observability.triad_eval import LowConfidenceRetractionException, TriadConfig, TriadEvaluator  # noqa: E402
from pipeline.ingest import IngestPipeline, infer_doc_type  # noqa: E402
from report.synthesizer import MemoInputs, build_memo  # noqa: E402
from security.proxy import AirGappedProxy  # noqa: E402

LOG = logging.getLogger("apex.gui")
RUN_LOCK = threading.Lock()
_PLACEHOLDER = re.compile(r"\[([A-Z_]+)_[0-9A-F]{6}\]")

SAMPLE_STATEMENT = (
    "Meridian Helix Technologies, Inc. reported fiscal 2025 revenue of $412.6 million. Dana R. Whitfield, our CEO, "
    "confirmed that Northgate Logistics Group, Inc. (Contract No. MSA-2023-004817, $70,100,000 in annual fees) may "
    "terminate without penalty after a change of control. Contact priya.v@meridianhelix.example or (512) 555-0147."
)


# ======================================================================= shared state
class LogBus:
    """Thread-safe rolling console fed by the apex.* loggers and GUI messages."""

    def __init__(self) -> None:
        """Create an empty bus."""
        self.lines: List[str] = []
        self.lock = threading.Lock()

    def say(self, message: str) -> None:
        """Append a timestamped line."""
        with self.lock:
            self.lines.append(f"{time.strftime('%H:%M:%S')}  {message}")

    def tail(self, n: int = 300) -> str:
        """Return the last ``n`` lines."""
        with self.lock:
            return "\n".join(self.lines[-n:])


class NetRecorder:
    """Records every guarded network operation (allowed loopback or blocked), tagged by stage."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.events: deque = deque(maxlen=3000)
        self.totals: Counter = Counter()
        self.stage = "idle"
        self.lock = threading.Lock()

    def add(self, kind: str, host: Any, verdict: str) -> None:
        """Record one operation."""
        with self.lock:
            self.totals[(self.stage, verdict)] += 1
            self.events.append({"Time": time.strftime("%H:%M:%S"), "Stage": self.stage, "Operation": kind,
                                "Target": str(host), "Verdict": verdict})

    def counts(self, exclude: Tuple[str, ...] = ("selftest",)) -> Tuple[int, int]:
        """Return ``(allowed, blocked)`` excluding intentional self-test probes."""
        with self.lock:
            allowed = sum(v for (s, verdict), v in self.totals.items() if verdict == "ALLOWED" and s not in exclude)
            blocked = sum(v for (s, verdict), v in self.totals.items() if verdict == "BLOCKED" and s not in exclude)
        return allowed, blocked


class Session:
    """Everything the three tabs share (single-user demo state)."""

    def __init__(self) -> None:
        """Initialise with default config and empty results."""
        self.cfg: ApexConfig = ApexConfig()
        self.proxy: Optional[AirGappedProxy] = None
        self.pipeline: Optional[IngestPipeline] = None
        self.pdfs: List[Path] = []
        self.ingest_report: Any = None
        self.results: List[Any] = []
        self.triad_rows: List[Dict[str, Any]] = []
        self.drift: Optional[Dict[str, Any]] = None
        self.selftest: Optional[bool] = None
        self.passage_i = 0


BUS, NET, STATE = LogBus(), NetRecorder(), Session()


class _BusHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        BUS.say(f"{record.name.split('.')[-1]:<9} {record.levelname:<7} {record.getMessage()}")


def _install_instrumentation() -> None:
    """Hook the apex.* loggers and the network guard so the UI can display both."""
    apex_logger = logging.getLogger("apex")
    if not getattr(apex_logger, "_gui_hooked", False):
        apex_logger.addHandler(_BusHandler())
        apex_logger.setLevel(logging.INFO)
        apex_logger._gui_hooked = True  # type: ignore[attr-defined]
    if not getattr(ng, "_gui_patched", False):
        original = ng._check

        def recording_check(kind: str, host: Any) -> None:
            active = bool(ng._STACK)
            try:
                original(kind, host)
            except NetworkViolation:
                NET.add(kind, host, "BLOCKED")
                raise
            if active:
                NET.add(kind, host, "ALLOWED")

        ng._check = recording_check
        ng._gui_patched = True  # type: ignore[attr-defined]


@contextlib.contextmanager
def stage(name: str):
    """Tag network events with the pipeline stage that produced them."""
    previous, NET.stage = NET.stage, name
    try:
        yield
    finally:
        NET.stage = previous


def get_proxy() -> AirGappedProxy:
    """Create the Stage 1 anonymization proxy once (stable HMAC key persisted in the workdir)."""
    if STATE.proxy is None:
        BUS.say("Loading anonymization proxy (spaCy NER + regex rules + HMAC placeholders)...")
        with stage("proxy-load"):
            STATE.proxy = AirGappedProxy(
                STATE.cfg.spacy_model, secret_key=load_or_create_proxy_key(STATE.cfg.data_dir),
                known_entities=stage1_known_entities(),
            )
        BUS.say("Proxy ready. Re-identification vault lives in process memory only.")
    return STATE.proxy


def stream_job(work: Callable[[], Any], render: Callable[[], Any]) -> Generator[Any, None, Dict[str, Any]]:
    """Run ``work`` in a thread while yielding ``render()`` so the UI stays live; returns ``{out|err}``."""
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["out"] = work()
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        yield render()
        thread.join(timeout=0.6)
    return box


# ======================================================================= Tab 1 views
def ingestion_metrics() -> pd.DataFrame:
    """Per-document chunk statistics read from the Stage 1 Chroma store."""
    cols = ["Document", "Type", "Chunks", "Mean tokens", "Max tokens", "CoC chunks"]
    pipe = STATE.pipeline
    if pipe is None or pipe.chroma is None:
        return pd.DataFrame(columns=cols)
    metas = pipe.chroma.collection.get(include=["metadatas"])["metadatas"] or []
    docs: Dict[str, Dict[str, Any]] = {}
    for m in metas:
        d = docs.setdefault(m["doc_id"], {"name": m["filename"], "type": m["doc_type"], "tok": [], "coc": 0})
        d["tok"].append(int(m["token_count"]))
        d["coc"] += 1 if m.get("has_change_of_control") else 0
    rows = [[d["name"], d["type"], len(d["tok"]), round(sum(d["tok"]) / len(d["tok"]), 1), max(d["tok"]), d["coc"]]
            for d in docs.values()]
    if rows:
        allt = [t for d in docs.values() for t in d["tok"]]
        rows.append(["TOTAL", "", len(allt), round(sum(allt) / len(allt), 1), max(allt), sum(d["coc"] for d in docs.values())])
    return pd.DataFrame(rows, columns=cols)


def security_banner() -> str:
    """Markdown verdict summarising the recorded network activity."""
    allowed, blocked = NET.counts()
    probes = NET.totals.get(("selftest", "BLOCKED"), 0)
    head = ("🟢 **AIR-GAP INTACT**: no external connection attempt was observed" if blocked == 0
            else f"🟠 **{blocked} EXTERNAL ATTEMPT(S) INTERCEPTED AND BLOCKED** (nothing left the process)")
    if STATE.selftest is None:
        test = "Egress self-test: *not run yet* (use the button)."
    elif STATE.selftest:
        test = (f"Egress self-test: ✅ probes to `203.0.113.10:443` and `example.invalid:80` were blocked before any "
                f"packet was sent ({probes} probe events).")
    else:
        test = "Egress self-test: ⛔ **FAILED**: do not trust this run."
    return (f"{head}\n\n- Guarded operations observed: **{allowed + blocked}** "
            f"(loopback/allow-listed: **{allowed}**, blocked: **{blocked}**)\n- {test}\n"
            "- Parsing, chunking, embedding and scrubbing run with **zero** sockets; only the local DB and model servers use loopback.")


def security_table() -> pd.DataFrame:
    """Newest-first intercepted-call log."""
    cols = ["Time", "Stage", "Operation", "Target", "Verdict"]
    with NET.lock:
        events = list(NET.events)[-150:][::-1]
    rows = []
    for e in events:
        verdict = ("⛔ BLOCKED (self-test probe)" if e["Stage"] == "selftest" and e["Verdict"] == "BLOCKED"
                   else "⛔ BLOCKED" if e["Verdict"] == "BLOCKED" else "✅ loopback / allow-listed")
        rows.append([e["Time"], e["Stage"], e["Operation"], e["Target"], verdict])
    return pd.DataFrame(rows, columns=cols)


def render_tab1(with_metrics: bool = True) -> Tuple[Any, ...]:
    """Console, ingestion metrics, security banner, security table."""
    return (BUS.tail(), ingestion_metrics() if with_metrics else gr.update(), security_banner(), security_table())


def tab1_job(title: str, work: Callable[[], Any]) -> Generator[Tuple[Any, ...], None, None]:
    """Run a Tab 1 job exclusively, streaming console and security views."""
    if not RUN_LOCK.acquire(blocking=False):
        BUS.say("⚠ another job is running; wait for it to finish")
        yield render_tab1()
        return
    try:
        BUS.say(f"▶ {title}")
        box = yield from stream_job(work, lambda: render_tab1(with_metrics=False))
        if "err" in box:
            BUS.say(f"✖ {title} failed: {str(box['err']).splitlines()[0] if str(box['err']) else type(box['err']).__name__}")
        else:
            BUS.say(f"✔ {title} done")
        yield render_tab1()
    finally:
        RUN_LOCK.release()


# ----------------------------------------------------------------------- Tab 1 jobs
def work_generate() -> None:
    """Generate the three synthetic PDFs (zero-socket)."""
    with stage("generate"), local_processing_guarantee(allow_loopback=False):
        paths = generate_all(STATE.cfg.pdf_dir)
    for p in paths:
        BUS.say(f"wrote {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    BUS.say("ground_truth.json written; hidden Change-of-Control clause verified in every PDF via PyPDF2")


def work_ingest(use_graph: bool) -> None:
    """Scrub + parse + chunk + embed + store (Stage 1 pipeline)."""
    cfg = STATE.cfg
    pdfs = sorted(cfg.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("Data room is empty: generate or upload PDFs first.")
    BUS.say(f"data room: {len(pdfs)} PDFs | chunk={cfg.chunk_tokens} tokens, overlap={cfg.chunk_overlap}")
    proxy = get_proxy()
    if STATE.pipeline is not None:
        STATE.pipeline.close()
    pipeline = IngestPipeline(cfg, proxy, use_graph=use_graph)
    try:
        with stage("ingest"):
            report = pipeline.run(pdfs)
    except BaseException:
        pipeline.close()
        raise
    STATE.pipeline, STATE.pdfs, STATE.ingest_report = pipeline, pdfs, report
    STATE.results, STATE.triad_rows, STATE.drift = [], [], None
    BUS.say(f"documents={report.documents} chunks={report.chunks} coc_clauses={report.clauses} "
            f"max_chunk_tokens={report.max_chunk_tokens} (budget {cfg.chunk_tokens})")
    BUS.say(f"proxy replacements: {proxy.stats()}")
    BUS.say(f"ingest-stage network audit: {report.audit}")


def on_generate():
    """Button: generate the synthetic data room."""
    yield from tab1_job("Generate synthetic data room", work_generate)


def on_ingest(use_graph: bool):
    """Button: ingest the data room."""
    yield from tab1_job("Ingest data room (scrub, chunk, embed, store)", lambda: work_ingest(use_graph))


def on_upload(files: Any):
    """Copy uploaded PDFs into the data room."""
    STATE.cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        src = Path(getattr(f, "name", f))
        if src.suffix.lower() == ".pdf":
            shutil.copy2(src, STATE.cfg.pdf_dir / src.name)
            BUS.say(f"uploaded {src.name} -> data room (click Ingest to process)")
    return render_tab1()


def on_selftest():
    """Button: prove the guard blocks external IP and DNS egress."""
    if not RUN_LOCK.acquire(blocking=False):
        gr.Warning("Another job is running.")
        return render_tab1()
    try:
        with stage("selftest"):
            STATE.selftest = bool(verify_guard_blocks_egress())
        BUS.say(f"egress self-test {'PASSED' if STATE.selftest else 'FAILED'}")
    finally:
        RUN_LOCK.release()
    return render_tab1()


def on_load_passage():
    """Pull a Change-of-Control chunk from the ingested store into the inspector."""
    pipe = STATE.pipeline
    if pipe is None or pipe.chroma is None:
        gr.Warning("Ingest the data room first.")
        return gr.update()
    col = pipe.chroma.collection
    docs = col.get(where={"has_change_of_control": True}, include=["documents"]).get("documents") or []
    if not docs:
        docs = col.get(limit=5, include=["documents"]).get("documents") or []
    if not docs:
        return gr.update()
    STATE.passage_i += 1
    return docs[STATE.passage_i % len(docs)][:1500]


def on_scrub(text: str):
    """Scrub text with the Stage 1 proxy and show raw vs abstract signal side by side."""
    empty = pd.DataFrame(columns=["Entity type", "Raw value (local only)", "Placeholder sent onward"])
    if not (text or "").strip():
        return [], [], "Enter or load a statement first.", empty
    if not RUN_LOCK.acquire(blocking=False):
        gr.Warning("Another job is running.")
        return [], [], "Busy: try again in a moment.", empty
    try:
        proxy = get_proxy()
        with stage("scrub"), local_processing_guarantee(allow_loopback=False):
            scrubbed, spans = proxy.scrub(text)
        try:
            proxy.assert_clean(scrubbed)
            verdict = "✅ **Egress gate passed**: no residual PII, currency or contract IDs detected in the scrubbed signal."
        except Exception as exc:  # noqa: BLE001
            verdict = f"⛔ **Egress gate BLOCKED this text**: {exc}"
        raw_seg, cur = [], 0
        for s in spans:
            if s.start > cur:
                raw_seg.append((text[cur:s.start], None))
            raw_seg.append((text[s.start:s.end], s.label))
            cur = s.end
        if cur < len(text):
            raw_seg.append((text[cur:], None))
        scrub_seg, cur = [], 0
        for m in _PLACEHOLDER.finditer(scrubbed):
            if m.start() > cur:
                scrub_seg.append((scrubbed[cur:m.start()], None))
            scrub_seg.append((m.group(0), m.group(1)))
            cur = m.end()
        if cur < len(scrubbed):
            scrub_seg.append((scrubbed[cur:], None))
        table = pd.DataFrame([[s.label, s.text.strip(), proxy.placeholder_for(s.label, s.text)] for s in spans],
                             columns=empty.columns)
        return raw_seg, scrub_seg, verdict, table
    except Exception as exc:  # noqa: BLE001
        return [], [], f"⛔ Scrub failed: {exc}", empty
    finally:
        RUN_LOCK.release()


# ======================================================================= Tab 2
T2_COLS = ["Query", "Status", "Context Relevance", "Groundedness", "Answer Relevance",
           "Claims kept / total", "Finding (anonymized view)", "Notes"]
STATUS_LABEL = {"PASS": "✅ PASS", "PARTIAL": "🟡 PARTIAL", "RETRACTED": "🔴 RETRACTED",
                "NO_CONTEXT": "⚪ NO CONTEXT", "ERROR": "⚠️ ERROR"}


def _meter(label: str, value: float, limit: float, note: str) -> str:
    ratio = value / limit if limit > 0 else 0.0
    width = min(ratio / 1.5, 1.0) * 100
    color = "#2e9e5b" if ratio < 0.7 else "#d9a400" if ratio < 1.0 else "#d64545"
    return (
        "<div style='margin:12px 0'>"
        f"<div style='display:flex;justify-content:space-between;font-size:14px'><b>{label}</b>"
        f"<span>{value:.4f} &nbsp;/&nbsp; limit {limit:.4f} &nbsp;({ratio:.0%} of limit)</span></div>"
        "<div style='position:relative;height:16px;border-radius:8px;background:rgba(128,128,128,.25);overflow:hidden'>"
        f"<div style='width:{width:.1f}%;height:100%;background:{color}'></div>"
        "<div style='position:absolute;left:66.7%;top:0;bottom:0;width:2px;background:#888' title='threshold'></div></div>"
        f"<div style='font-size:12px;opacity:.7;margin-top:2px'>{note}</div></div>"
    )


def drift_html(d: Optional[Dict[str, Any]]) -> str:
    """Drift health bars (cosine distance and KL divergence vs calibrated limits)."""
    if not d:
        return "<i>Drift check has not run yet.</i>"
    if d.get("status") == "SKIPPED":
        return f"<b>Drift check skipped:</b> {html.escape(str(d.get('reason', '')))}"
    ok = d["status"] == "STABLE"
    pill = (f"<span style='padding:2px 10px;border-radius:12px;color:white;background:{'#2e9e5b' if ok else '#d64545'}'>"
            f"{'STABLE' if ok else 'DRIFT WARNING'}</span>")
    return (
        f"<div><b>Semantic drift vs baseline centroid</b> &nbsp;{pill}<br>"
        f"<span style='font-size:12px;opacity:.75'>held-out document <code>{html.escape(str(d['holdout']))}</code> "
        f"({d['n_new']} chunks) vs baseline of {d['n_baseline']} chunks from the other documents</span></div>"
        + _meter("Cosine distance (new centroid vs baseline centroid)", d["centroid_cosine_distance"], d["cosine_threshold"],
                 f"mean per-chunk distance {d['mean_doc_cosine_distance']:.4f} vs baseline {d['baseline_mean_doc_distance']:.4f}")
        + _meter("KL divergence KL(new ‖ baseline)", d["kl_divergence"], d["kl_threshold"],
                 f"radial KL {d['kl_radial']:.4f}; mean over {len(d['kl_components'])} principal components")
        + "<div style='font-size:12px;opacity:.7'>Limits are bootstrap-calibrated from the baseline. "
          "A small corpus gives wide limits, and a warning is advisory when the held-out file is a different document class.</div>"
    )


def tab2_view(status: str, rows: List[List[Any]], drift: Optional[Dict[str, Any]] = None, details: Any = None):
    """Assemble Tab 2 outputs: status, table, drift bars, drift JSON, triad detail JSON."""
    df = pd.DataFrame(rows, columns=T2_COLS) if rows else pd.DataFrame(columns=T2_COLS)
    return status, df, drift_html(drift), json.loads(json.dumps(drift or {}, default=str)), \
        json.loads(json.dumps(details or [], default=str))


def on_audit(backend: str, model: str, url: str, threshold: float):
    """Button: run local RAG extraction, the triad gate and the drift check."""
    if STATE.pipeline is None or STATE.pipeline.embedder is None or STATE.pipeline.chroma is None:
        yield tab2_view("⚠️ Ingest the data room first (Tab 1).", [])
        return
    if not RUN_LOCK.acquire(blocking=False):
        yield tab2_view("⚠️ Another job is running.", [])
        return
    rows: List[List[Any]] = []
    try:
        cfg, pipe, proxy = STATE.cfg, STATE.pipeline, get_proxy()
        overrides = {k: v for k, v in (("backend", backend), ("model", (model or "").strip()),
                                       ("base_url", (url or "").strip())) if v}
        llm = LocalLLMClient(dataclasses.replace(LLMConfig(), **overrides), cfg.allowed_hosts)
        with stage("llm-health"):
            alive = llm.ping()
        if not alive:
            yield tab2_view(f"⛔ No local model server at `{llm.cfg.url}` ({llm.cfg.backend}). Start Ollama/vLLM and retry.", [])
            return
        evaluator = TriadEvaluator(pipe.embedder, llm, TriadConfig(
            groundedness_threshold=float(threshold), metrics_path=cfg.data_dir / "observability" / "triad_metrics.jsonl"))
        extractor = LocalRiskExtractor(HybridRetriever(pipe.embedder, pipe.chroma, pipe.graph, False, cfg.allowed_hosts),
                                       llm, evaluator)
        results, trows, details = [], [], []
        for i, q in enumerate(AUDIT_QUERIES, start=1):
            yield tab2_view(f"⏳ Query {i}/{len(AUDIT_QUERIES)}: `{q.name}` on `{llm.cfg.model_name}`...", rows, None, details)
            report, res, note, status = None, None, "", "PASS"
            try:
                with stage("rag-llm"):
                    res = extractor.extract(q.question, q.clause_types)
                report = res.triad
            except LowConfidenceRetractionException as exc:
                report, status, note = exc.report, "RETRACTED", "; ".join(exc.report.reasons)[:200]
            except NetworkViolation:
                raise
            except Exception as exc:  # noqa: BLE001
                status, note = "ERROR", f"{type(exc).__name__}: {str(exc)[:160]}"
            if status == "PASS":
                if report is None:
                    status = "NO_CONTEXT"
                elif getattr(report, "partial", False) or (not report.vacuous and report.groundedness < report.threshold):
                    status = "PARTIAL"
                    note = "; ".join(report.reasons)[:200]
            kept = len(res.findings) if res else 0
            total = len(report.claims) if report else 0
            titles = " | ".join(f.title for f in res.findings[:2]) if res and res.findings else ""
            r2 = lambda v: round(float(v), 2) if report else 0.0  # noqa: E731
            rows.append([q.name, STATUS_LABEL[status], r2(report.context_relevance if report else 0),
                         r2(report.groundedness if report else 0), r2(report.answer_relevance if report else 0),
                         f"{kept} / {total}", proxy.anonymize(titles)[:140], note])
            trows.append({"name": q.name, "status": status,
                          "context_relevance": float(report.context_relevance) if report else 0.0,
                          "groundedness": float(report.groundedness) if report else 0.0,
                          "answer_relevance": float(report.answer_relevance) if report else 0.0,
                          "reasons": list(report.reasons) if report else ([note] if note else [])})
            if res is not None and (res.findings or status != "NO_CONTEXT"):
                results.append(res)
            if report is not None:
                details.append({"query": q.name, "status": status, **report.to_dict()})
        STATE.results, STATE.triad_rows = results, trows
        yield tab2_view("⏳ Running semantic drift check...", rows, None, details)
        try:
            with stage("drift"), local_processing_guarantee(cfg.allowed_hosts):
                STATE.drift = drift_gate(pipe, STATE.pdfs)
        except Exception as exc:  # noqa: BLE001
            STATE.drift = {"status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}
        has_coc = any(f.category == "CHANGE_OF_CONTROL" for r in results for f in r.findings)
        counts = Counter(r["status"] for r in trows)
        summary = (f"✅ Audit complete: {counts['PASS']} PASS, {counts['PARTIAL']} PARTIAL, {counts['RETRACTED']} RETRACTED"
                   f"{', ' + str(counts['ERROR']) + ' ERROR' if counts['ERROR'] else ''}. "
                   + ("Verified **Change-of-Control** finding available, so Tab 3 can run."
                      if has_coc else "⛔ **No verified Change-of-Control finding**: Tab 3 cannot price the deal."))
        yield tab2_view(summary, rows, STATE.drift, details)
    except NetworkViolation as exc:
        yield tab2_view(f"⛔ **Air-gap violation blocked:** {exc}", rows)
    finally:
        RUN_LOCK.release()


# ======================================================================= Tab 3
SPEAKERS = {
    "audit_legal": ("⚖️ Legal Agent", "assistant"),
    "financial_audit": ("💰 Financial Agent", "user"),
    "market_intel": ("📈 Market Agent", "assistant"),
    "mediator": ("🤝 Consensus Coordinator", "assistant"),
}


def compute_negotiation(terms: DealTerms) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Run the three agents and the debate loop (same node order as agents/graph.py), zero sockets."""
    cfg, proxy = STATE.cfg, get_proxy()
    ledger = EgressLedger(proxy)
    script: List[Tuple[str, str]] = []
    with stage("agents"), local_processing_guarantee(cfg.allowed_hosts, allow_loopback=False):
        brief = build_brief(STATE.results, proxy, ledger)
        script.append(("mediator", f"Anonymized brief sealed: **{len(brief.findings)}** findings, sector terms "
                                   f"`{', '.join(brief.sector_terms)}`. Every hand-off is proxy-gated and logged."))
        legal = AuditLegalAgent(terms).run(brief)
        script.append(("audit_legal", f"Verified Change-of-Control findings put **{legal.exposure_pct:.0f}%** of revenue behind "
                       f"no-penalty exit rights (per-customer shares reconcile: {legal.reconciled}). Gross exposure "
                       f"**${legal.gross_penalty_musd:.1f}M** on a ${terms.base_price_musd:.1f}M base. Rating **{legal.risk_rating}**; "
                       f"earn-out ask ${legal.earnout_ask_musd:.1f}M."))
        market = MarketIntelligenceAgent(MockWebSearchTool(), ledger).run(brief)
        script.append(("market_intel", f"External public query (released only after the proxy gate): `{market.query}`\n\n"
                       f"Sector multiples moved **{market.sector_move_pct:+.1f}%** (peer EV/Revenue corroboration: "
                       f"{market.corroborated}). Tool `{market.tool}` is a labelled mock, so no sockets were opened."))
        fin = FinancialAuditAgent(terms).run(legal, market)
        script.append(("financial_audit", f"Baseline = EBITDA ${fin.ebitda_musd:.2f}M × **{fin.base_multiple:.2f}x**. Sector-adjusted "
                       f"multiple **{fin.adjusted_multiple:.2f}x** → full markdown ${fin.full_markdown_musd:.1f}M. Risk-weighted churn at "
                       f"{fin.exercise_probability:.1%} = ${fin.risk_weighted_churn_musd:.1f}M; earn-out cap ${fin.earnout_cap_musd:.1f}M."))
        neg = open_negotiation(terms, legal, market, fin)
        seen = 0
        while not neg.all_settled:
            run_round(neg, terms)
            for t in neg.transcript[seen:]:
                value = f" · **${t['value']:,.1f}M**" if t["value"] is not None else ""
                script.append((t["agent"] if t["agent"] in SPEAKERS else "mediator",
                               f"*Round {t['round']}* · `{t['lever']}` · {t['action']}{value}\n\n{t['message']}"))
            seen = len(neg.transcript)
        offer = final_offer(neg, terms)
        script.append(("mediator", f"**Consensus reached.** Upfront **${offer.upfront_musd:.1f}M** + earn-out "
                       f"**${offer.earnout_musd:.1f}M** (max ${offer.headline_musd:.1f}M). Lever status: {offer.statuses}."))
    return script, {"offer": offer, "neg": neg, "legal": legal, "market": market, "fin": fin, "brief": brief, "ledger": ledger}


def citation_table(proxy: AirGappedProxy) -> str:
    """Anonymized citation sources for the retained findings (no filenames: type + chunk hash only)."""
    rows, seen = [], set()
    for res in STATE.results:
        refs = {c.ref: c for c in res.contexts}
        for f in res.findings:
            for ref in f.source_ids:
                c = refs.get(ref)
                if c is None or (c.chunk_id, f.category) in seen:
                    continue
                seen.add((c.chunk_id, f.category))
                pages = str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
                section = proxy.anonymize(c.section or "n/a").replace("|", "/").replace("\n", " ")[:70]
                rows.append(f"| {f.category} | {infer_doc_type(c.filename)} · `{c.chunk_id[:13]}` | {section} | {pages} |")
    if not rows:
        return ""
    return ("\n## 9. Citation sources (anonymized)\n\n| Risk category | Source (doc type · chunk id) | Section | Pages |\n|---|---|---|---|\n"
            + "\n".join(rows) + "\n\nMarket sources are listed in section 5.\n")


def on_negotiate(base: float, ebitda: float, prob: float):
    """Button: run the multi-agent negotiation and synthesize the memo."""
    keep = gr.update()
    if not any(f.category == "CHANGE_OF_CONTROL" for r in STATE.results for f in r.findings):
        yield ([{"role": "assistant", "content": "⛔ No verified Change-of-Control finding. Run Tab 2 first."}],
               keep, keep, keep, keep, keep)
        return
    if not RUN_LOCK.acquire(blocking=False):
        yield ([{"role": "assistant", "content": "⚠️ Another job is running."}], keep, keep, keep, keep, keep)
        return
    try:
        yield ([{"role": "assistant", "content": "⏳ Convening the agents..."}], keep, keep, keep, keep, keep)
        if STATE.selftest is None:
            with stage("selftest"):
                STATE.selftest = bool(verify_guard_blocks_egress())
        terms = DealTerms(base_price_musd=float(base), ebitda_musd=float(ebitda), exercise_probability=float(prob))
        try:
            script, ctx = compute_negotiation(terms)
        except (AgentError, NetworkViolation, Exception) as exc:  # noqa: BLE001
            label = "Air-gap violation" if isinstance(exc, NetworkViolation) else "Negotiation aborted"
            yield ([{"role": "assistant", "content": f"⛔ **{label}:** {exc}"}], keep, keep, keep, keep, keep)
            return
        chat: List[Dict[str, str]] = []
        for key, text in script:
            name, role = SPEAKERS[key]
            chat.append({"role": role, "content": f"**{name}**\n\n{text}"})
            yield (list(chat), keep, keep, keep, keep, keep)
            time.sleep(0.45)

        proxy, offer, ledger = get_proxy(), ctx["offer"], ctx["ledger"]
        allowed, blocked = NET.counts()
        evidence = {
            "ledger": ledger.audit(stage1_canaries()),
            "network_summary": f"permitted(loopback/allow-listed)={allowed} blocked={blocked} clean={blocked == 0} (all stages, process-wide recorder)",
            "network_blocked": blocked,
            "ingest_audit": STATE.ingest_report.audit if STATE.ingest_report else "n/a",
            "proxy_stats": proxy.stats(),
            "guard_selftest": bool(STATE.selftest),
        }
        drift = STATE.drift or {"status": "SKIPPED", "reason": "not run"}
        memo = build_memo(MemoInputs(offer, ctx["neg"], ctx["legal"], ctx["market"], ctx["fin"], ctx["brief"],
                                     STATE.triad_rows, drift, evidence, "gradio-stepwise (agents/graph.py node order)"))
        memo += citation_table(proxy)
        out_dir = STATE.cfg.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        memo_path = out_dir / "Investment_Recommendation_Memo.md"
        memo_path.write_text(memo, encoding="utf-8")
        (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")

        met = abs(offer.upfront_musd - 34.5) < 0.05 and abs(offer.earnout_musd - 10.0) < 0.05
        bid = (f"## Revised bid: **${offer.upfront_musd:.1f}M** upfront + **${offer.earnout_musd:.1f}M** earn-out\n"
               f"Maximum headline **${offer.headline_musd:.1f}M** vs **${offer.base_price_musd:.1f}M** opening base "
               f"(sector markdown −${offer.sector_deduction_musd:.1f}M, risk-weighted churn −${offer.churn_deduction_musd:.1f}M). "
               f"Mandate target ($34.5M + $10.0M): {'✅ met' if met else '⚠️ not met with the current deal-term inputs'}.")
        led = evidence["ledger"]
        passed = led["clean"] and blocked == 0 and evidence["guard_selftest"]
        airgap = (f"### {'🟢' if passed else '🔴'} Air-gap integrity checks: {'PASS' if passed else 'FAIL'}\n"
                  f"- {led['events']} boundary releases ({led['internal_events']} internal, {led['external_events']} external), all proxy-gated\n"
                  f"- Residual-PII scan: **{len(led['leaks'])}** leaks · canary scan ({led['canaries_checked']} identifiers): "
                  f"**{len(led['canary_hits'])}** hits · external network attempts: **{blocked}**\n"
                  "- Evidence for these checks, not a formal proof; see memo section 8 for scope.")
        yield (list(chat), bid, airgap, memo, str(memo_path), json.loads(json.dumps(evidence, default=str)))
    finally:
        RUN_LOCK.release()


# ======================================================================= UI
def _chatbot() -> Any:
    kwargs = dict(label="Agent negotiation log", height=520)
    try:
        return gr.Chatbot(type="messages", **kwargs)
    except TypeError:
        return gr.Chatbot(**kwargs)


def build_ui() -> gr.Blocks:
    """Construct the three-tab dashboard."""
    with gr.Blocks(title="AegisDeal | Operation Apex", analytics_enabled=False) as demo:
        gr.Markdown("# 🛡️ AegisDeal: Operation Apex\n"
                    "Air-gapped M&A due diligence: local PII scrubbing → local RAG with groundedness gating → multi-agent price negotiation.")
        with gr.Tabs():
            # ---------------------------------------------------------------- Tab 1
            with gr.Tab("Data Room & Air-Gap Security Setup"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload = gr.File(label="Upload target PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
                        use_graph = gr.Checkbox(value=False, label="Also write to Neo4j (optional; needs a Neo4j server)")                        
                        b_gen = gr.Button("Generate Synthetic Data Room", variant="primary")
                        b_ing = gr.Button("Ingest Data Room (scrub → chunk → embed → store)")
                        b_test = gr.Button("Run Egress Self-Test")
                    console = gr.Textbox(label="Ingestion & chunking console", lines=17, max_lines=17,
                                         interactive=False, autoscroll=True, scale=2)
                metrics = gr.Dataframe(label="Ingestion metrics (per document)", interactive=False, wrap=True)
                with gr.Accordion("Security Audit Box: intercepted network call log", open=True):
                    banner = gr.Markdown()
                    sec_table = gr.Dataframe(label="Guarded network operations (newest first)", interactive=False, wrap=True)
                gr.Markdown("### PII Scrubbing Inspector\nThe left panel is the private text and stays local. "
                            "Only the right panel (the abstract signal) may cross a boundary.")
                raw_in = gr.Textbox(label="Raw private statement (edit freely)", value=SAMPLE_STATEMENT, lines=4)
                with gr.Row():
                    b_load = gr.Button("Load a Change-of-Control passage from the ingested data room")
                    b_scrub = gr.Button("Scrub", variant="primary")
                with gr.Row():
                    raw_view = gr.HighlightedText(label="Raw private statement (detected entities highlighted)", combine_adjacent=False)
                    scrub_view = gr.HighlightedText(label="Scrubbed abstract signal", combine_adjacent=False)
                gate_md = gr.Markdown()
                span_table = gr.Dataframe(label="Replacement map (local vault)", interactive=False, wrap=True)
                t1_out = [console, metrics, banner, sec_table]
                b_gen.click(on_generate, None, t1_out)
                b_ing.click(on_ingest, [use_graph], t1_out)
                b_test.click(on_selftest, None, t1_out)
                upload.upload(on_upload, [upload], t1_out)
                b_load.click(on_load_passage, None, [raw_in])
                b_scrub.click(on_scrub, [raw_in], [raw_view, scrub_view, gate_md, span_table])
                demo.load(render_tab1, None, t1_out)

            # ---------------------------------------------------------------- Tab 2
            with gr.Tab("RAG Triad & Data Drift Observability Monitor"):
                with gr.Accordion("Local LLM settings", open=False):
                    with gr.Row():
                        backend = gr.Dropdown(["ollama", "vllm"], value=os.getenv("APEX_LLM_BACKEND", "ollama"), label="Backend")
                        model = gr.Textbox(value=os.getenv("APEX_LLM_MODEL", "qwen2.5:3b-instruct"), label="Model tag")
                        url = gr.Textbox(value=os.getenv("APEX_LLM_URL", ""), label="Server URL (blank = loopback default)")
                        thr = gr.Slider(0.5, 0.95, value=0.75, step=0.01, label="Groundedness threshold")
                b_audit = gr.Button("Run Local RAG Extraction & Audit", variant="primary")
                t2_status = gr.Markdown("Ingest the data room (Tab 1), then run the audit.")
                t2_df = gr.Dataframe(value=pd.DataFrame(columns=T2_COLS), label="Query results & RAG Triad scores",
                                     interactive=False, wrap=True)
                gr.Markdown("### Data drift health")
                t2_drift = gr.HTML()
                with gr.Accordion("Drift report (JSON)", open=False):
                    t2_drift_json = gr.JSON()
                with gr.Accordion("Per-claim triad detail (JSON, scores only)", open=False):
                    t2_detail = gr.JSON()
                b_audit.click(on_audit, [backend, model, url, thr], [t2_status, t2_df, t2_drift, t2_drift_json, t2_detail])

            # ---------------------------------------------------------------- Tab 3
            with gr.Tab("Multi-Agent Consensus & Final Investment Memo"):
                with gr.Accordion("Deal terms (mandate inputs)", open=False):
                    with gr.Row():
                        s_base = gr.Slider(10, 200, value=50.0, step=0.5, label="Opening base price (USD M)")
                        s_ebitda = gr.Slider(1, 30, value=6.25, step=0.25, label="Baseline EBITDA (USD M)")
                        s_prob = gr.Slider(0, 1, value=0.325, step=0.005,
                                           label="Exercise probability (calibration input, not a derived estimate)")
                b_neg = gr.Button("Run Multi-Agent Negotiation Loop", variant="primary")
                bid_md = gr.Markdown()
                chat = _chatbot()
                airgap_md = gr.Markdown()
                gr.Markdown("### Investment Recommendation Memo")
                memo_md = gr.Markdown()
                memo_file = gr.File(label="Download memo (.md)", interactive=False)
                with gr.Accordion("Air-gap evidence (JSON)", open=False):
                    evidence_json = gr.JSON()
                b_neg.click(on_negotiate, [s_base, s_ebitda, s_prob], [chat, bid_md, airgap_md, memo_md, memo_file, evidence_json])
    return demo


def main() -> None:
    """CLI entry point: ``python gui.py``."""
    parser = argparse.ArgumentParser(description="AegisDeal / Operation Apex dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--workdir", type=Path, default=ApexConfig().data_dir)
    args = parser.parse_args()
    STATE.cfg = dataclasses.replace(ApexConfig(), data_dir=args.workdir)
    _install_instrumentation()
    demo = build_ui()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()