#!/usr/bin/env python3
"""AegisDeal FastAPI backend: SQLite + ChromaDB, CPU-friendly, Neo4j optional (off by default).

Run:  python -m backend.server [--port 8000] [--workdir ./apex_data]      Docs: /docs
"""
from __future__ import annotations

import security.network_guard as _ng  # noqa: F401  (sets offline env before any ML import)

import argparse
import dataclasses
import hashlib
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from agents.copilot import AgenticCopilot, EgressMonitor, FindingsStore, ensure_demo_financials
from agents.core import (AgentError, AnonymizedBrief, DealTerms, EgressLedger, MockWebSearchTool, SECTOR_VOCAB,
                         stage1_canaries, stage1_known_entities)
from agents.graph import run_consensus
from agents.nodes import AuditLegalAgent, FinancialAuditAgent, MarketIntelligenceAgent
from backend.db import Database
from config import ApexConfig
from data.synthetic_gen import generate_all, validate_financials_csv
from llm.extractor import AUDIT_QUERIES, HybridRetriever, LocalRiskExtractor
from llm.quant_dpo import DPOStore, QuantizedModelLoader, build_pair_from_resolution, quantize_demo, run_dpo_step
from main import drift_gate, load_or_create_proxy_key
from observability.triad_eval import LowConfidenceRetractionException, TriadConfig, TriadEvaluator
from pipeline.csv_ingest import ingest_financial_csv
from pipeline.ingest import ChromaStore, IngestPipeline, LocalEmbedder, infer_doc_type
from report.synthesizer import MemoInputs, build_memo
from security.network_guard import NetworkViolation, local_processing_guarantee
from security.proxy import AirGappedProxy, LeakageError

LOG = logging.getLogger("apex.server")
DATA_DIR: Path = ApexConfig().data_dir
RT: Optional["Runtime"] = None
MAX_UPLOAD = 20 * 1024 * 1024
DOMAIN_IGNORE = ("change of control", "change-of-control", "earn-out", "earn out", "rag triad", "air gap", "air-gap",
                 "copilot", "triad", "hitl", "dpo", "qwen", "ollama", "chromadb", "neo4j", "legal agent",
                 "financial agent", "market agent", "consensus coordinator", "sqlite")


# ======================================================================= runtime
class Runtime:
    """Process-wide state: proxy, quantized model backend, SQLite, lazily opened ChromaDB/embedder, copilot."""

    def __init__(self, data_dir: Path) -> None:
        """Build everything cheap; the embedder and Chroma open lazily."""
        self.cfg = dataclasses.replace(ApexConfig(), data_dir=data_dir)
        cfg = self.cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cfg.data_dir / "aegisdeal.db"
        self.db = Database(self.db_path)  # creates all five tables first
        self.monitor = EgressMonitor()
        self.monitor.install()
        with self.monitor.stage("proxy-load"):
            self.proxy = AirGappedProxy(cfg.spacy_model, secret_key=load_or_create_proxy_key(cfg.data_dir),
                                        known_entities=stage1_known_entities(), ignore=DOMAIN_IGNORE)
        with self.monitor.stage("llm-health"):
            self.model = QuantizedModelLoader(allowed_hosts=cfg.allowed_hosts).load()
        self.findings = FindingsStore(self.db_path)
        self.dpo = DPOStore(cfg.data_dir / "dpo_preferences.jsonl", self.db_path)
        self.csv_path = ensure_demo_financials(cfg.data_dir / "demo_financials.csv")
        self.embedder: Any = None
        self.chroma: Optional[ChromaStore] = None
        self.graph: Any = None
        self.job_lock = threading.Lock()
        self._store_lock = threading.RLock()
        self.ingest_audit = "n/a (no ingest in this session)"
        self.triad_rows: List[Dict[str, Any]] = []
        self.drift: Dict[str, Any] = {"status": "SKIPPED", "reason": "audit not run in this session"}
        self.copilot = AgenticCopilot(self.proxy, self.findings, self.csv_path, self.monitor, store_provider=self.stores,
                                      planner_llm=self.model.backend if self.model.live else None)
        self.monitor.run_selftest()

    def adopt(self, pipeline: IngestPipeline) -> None:
        """Reuse the stores/embedder opened by an ingestion run."""
        with self._store_lock:
            self.embedder, self.chroma, self.graph = pipeline.embedder, pipeline.chroma, pipeline.graph

    def chroma_store(self) -> Optional[ChromaStore]:
        """Open the persisted ChromaDB lazily (no embedder needed)."""
        with self._store_lock:
            if self.chroma is None and self.cfg.chroma_dir.exists():
                try:
                    with local_processing_guarantee(self.cfg.allowed_hosts):
                        self.chroma = ChromaStore(self.cfg.chroma_dir, self.cfg.chroma_collection)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("chroma unavailable: %s", exc)
            return self.chroma

    def stores(self) -> Optional[Tuple[ChromaStore, Any]]:
        """Return ``(ChromaStore, embedder)``, loading the local embedder on first use."""
        chroma = self.chroma_store()
        if chroma is None:
            return None
        with self._store_lock:
            if self.embedder is None:
                try:
                    with local_processing_guarantee(allow_loopback=False):
                        self.embedder = LocalEmbedder(self.cfg.embedding_model_path, self.cfg.embed_batch_size)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("embedder unavailable: %s", exc)
                    return None
            return chroma, self.embedder

    def register_documents(self, files: Sequence[Path]) -> int:
        """Write per-document chunk statistics (read from Chroma) into the SQLite ``documents`` table."""
        chroma = self.chroma_store()
        if chroma is None:
            return 0
        stats: Dict[str, Dict[str, Any]] = {}
        for m in chroma.collection.get(include=["metadatas"])["metadatas"] or []:
            d = stats.setdefault(m["doc_id"], {"type": m["doc_type"], "tok": [], "coc": 0})
            d["tok"].append(int(m["token_count"]))
            d["coc"] += 1 if m.get("has_change_of_control") else 0
        registered = 0
        for path in files:
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            d = stats.get(sha[:16])
            if not d:
                continue
            self.db.upsert_document(
                doc_id=sha[:16], display_name=f"{d['type']}_{sha[:8]}", doc_type=d["type"], kind=path.suffix[1:].lower(),
                sha256=sha, size_bytes=len(raw), chunks=len(d["tok"]), mean_tokens=round(sum(d["tok"]) / len(d["tok"]), 1),
                max_tokens=max(d["tok"]), coc_chunks=d["coc"])
            registered += 1
        return registered


def rt() -> Runtime:
    """Return the runtime or 503 while starting."""
    if RT is None:
        raise HTTPException(503, "server is starting")
    return RT


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the runtime once at startup."""
    global RT
    RT = Runtime(DATA_DIR)
    yield


app = FastAPI(title="AegisDeal API", version="6.0", lifespan=lifespan)


# ======================================================================= schemas
class IngestRequest(BaseModel):
    """Ingest options (Neo4j is opt-in)."""

    use_graph: bool = False


class AuditRequest(BaseModel):
    """Audit options."""

    groundedness_threshold: float = Field(0.75, ge=0.5, le=0.95)


class HitlRequest(BaseModel):
    """Reviewer decision. ``edit`` is an alias of ``override``."""

    finding_id: str
    action: Literal["approve", "edit", "override", "reject"]
    override: Optional[Dict[str, str]] = Field(None, description="fields to correct, e.g. {'summary': '...', 'severity': 'HIGH'}")
    rejected_text: Optional[str] = None
    reviewer: str = Field("analyst", max_length=64)
    note: str = Field("", max_length=1000)


class NegotiateRequest(BaseModel):
    """Deal terms for the consensus loop."""

    base_price_musd: float = Field(50.0, gt=0, le=1000)
    ebitda_musd: float = Field(6.25, gt=0, le=500)
    exercise_probability: float = Field(0.325, ge=0, le=1)
    require_review_for_partial: bool = Field(False, description="exclude PARTIAL findings until a human approves/edits them")


class DpoStepRequest(BaseModel):
    """DPO step options."""

    steps: int = Field(60, ge=1, le=500)
    beta: float = Field(0.1, gt=0, le=1)
    lr: float = Field(0.5, gt=0, le=5)
    backend: Literal["auto", "numpy", "trl"] = "auto"
    allow_demo_seed: bool = True


class ChatRequest(BaseModel):
    """Chat message."""

    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field("default", max_length=64)


# ======================================================================= helpers
def effective_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """The finding as the agents should see it: the human's edit if there is one, else the model output."""
    raw = row["human_output"] if row["hitl_status"] == "overridden" and row["human_output"] else row["model_output"]
    return json.loads(raw)["risks"][0]


def select_findings(rows: Sequence[Dict[str, Any]], require_review_for_partial: bool) -> List[Dict[str, Any]]:
    """Drop rejected findings (and unreviewed PARTIAL ones when review is required)."""
    return [r for r in rows if r["hitl_status"] != "rejected"
            and not (require_review_for_partial and r["status"] == "PARTIAL" and r["hitl_status"] == "pending")]


def brief_from_findings(rows: Sequence[Dict[str, Any]], proxy: AirGappedProxy, ledger: EgressLedger) -> AnonymizedBrief:
    """Build the agents' brief from stored (already anonymized, human-reviewed) findings; every hand-off is gated."""
    findings: List[Dict[str, Any]] = []
    for row in rows:
        item = effective_item(row)
        findings.append(ledger.release("internal", "legal_agent_input", {
            "query_text": row["query_name"], "category": item["category"], "severity": item["severity"], "title": item["title"],
            "summary": item["summary"], "evidence_quote": item["evidence_quote"],
            "financial_exposure": item.get("financial_exposure"), "recommended_action": item.get("recommended_action", "")}))
    blob = " ".join(f"{f['title']} {f['summary']}" for f in findings).casefold()
    terms = [t for t in SECTOR_VOCAB if t in blob] or ["enterprise software"]
    grounded = [r["groundedness"] for r in rows if r["groundedness"] is not None]
    signals = ledger.release("internal", "vector_signals", {
        "findings": str(len(rows)), "mean_groundedness": f"{sum(grounded) / len(grounded):.3f}" if grounded else "n/a",
        "coc_findings": str(sum(1 for r in rows if r["category"] == "CHANGE_OF_CONTROL")), "sector_terms": terms})
    return AnonymizedBrief(findings=findings, signals=signals, sector_terms=terms)


def derive_triad_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-query triad summary reconstructed from stored findings (used when no audit ran in this session)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["query_name"], []).append(row)
    out = []
    for name, items in groups.items():
        avg = lambda k: sum((i[k] or 0.0) for i in items) / len(items)  # noqa: E731
        out.append({"name": name, "status": "PARTIAL" if any(i["status"] == "PARTIAL" for i in items) else "PASS",
                    "context_relevance": avg("context_relevance"), "groundedness": avg("groundedness"),
                    "answer_relevance": avg("answer_relevance")})
    return out


# ======================================================================= endpoints
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Air-gap status, active quantized model and DB row counts."""
    r = rt()
    allowed, blocked = r.monitor.counts()
    selftest = r.monitor.selftest
    status = "UNVERIFIED" if not selftest else "ATTEMPTS_BLOCKED" if blocked else "INTACT"
    chroma = r.chroma_store()
    rep = r.model.report
    return {
        "status": "ok",
        "airgap": {"status": status, "egress_selftest": selftest, "loopback_ops": allowed, "blocked_ops": blocked},
        "model": {"backend": rep.backend, "name": rep.model, "quantization": rep.quantization, "live": r.model.live,
                  "disk_gb": rep.disk_gb, "fp16_gb": rep.fp16_gb, "reduction_x": rep.reduction_x, "measured": rep.measured},
        "db": r.db.counts(), "chroma_chunks": chroma.collection.count() if chroma else 0,
        "graph": "neo4j" if r.graph else "disabled (SQLite + ChromaDB)",
    }


@app.get("/api/quant")
def quant(demo: bool = False) -> Dict[str, Any]:
    """Quantization footprint report (measured via Ollama, else labelled reference values)."""
    r = rt()
    out: Dict[str, Any] = {"report": r.model.report.to_dict()}
    if demo:
        out["quantize_demo"] = quantize_demo()
    return out


@app.post("/api/generate-demo")
def generate_demo() -> Dict[str, Any]:
    """Generate the 3 synthetic PDFs and ``demo_financials.csv`` (zero sockets)."""
    r = rt()
    if not r.job_lock.acquire(blocking=False):
        raise HTTPException(409, "another job is running")
    try:
        started = time.perf_counter()
        with r.monitor.stage("generate"), local_processing_guarantee(allow_loopback=False):
            pdfs = generate_all(r.cfg.pdf_dir, csv_path=r.csv_path)
            rows = validate_financials_csv(r.csv_path)
        r.copilot.csv_path = r.csv_path
        return {"pdfs": [{"doc_type": infer_doc_type(p.name), "size_kb": round(p.stat().st_size / 1024)} for p in pdfs],
                "csv": {"file": "demo_financials.csv", "rows": len(rows), "download": "/api/demo-financials", "preview": "/api/financials"},
                "elapsed_s": round(time.perf_counter() - started, 2), "next": "POST /api/ingest"}
    finally:
        r.job_lock.release()


@app.get("/api/financials")
def financials() -> List[Dict[str, str]]:
    """Rows of the active financials CSV (for in-UI inspection)."""
    try:
        return validate_financials_csv(rt().copilot.csv_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(404, f"no valid financials CSV: {exc}") from exc


@app.get("/api/demo-financials")
def download_financials() -> FileResponse:
    """Download the active financials CSV."""
    path = rt().copilot.csv_path
    if not path.exists():
        raise HTTPException(404, "call POST /api/generate-demo first")
    return FileResponse(path, media_type="text/csv", filename="demo_financials.csv")


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a PDF or a financials CSV (same schema as the demo CSV). Names are replaced by content hashes."""
    r = rt()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".csv"):
        raise HTTPException(415, "only .pdf and .csv uploads are accepted")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file larger than 20 MB")
    digest = hashlib.sha256(data).hexdigest()[:12]
    if suffix == ".pdf":
        if not data.startswith(b"%PDF"):
            raise HTTPException(422, "not a PDF file")
        dest = r.cfg.pdf_dir / f"upload_{digest}.pdf"
    else:
        dest = r.cfg.data_dir / "csv" / f"upload_{digest}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    if suffix == ".csv":
        try:
            validate_financials_csv(dest)
        except ValueError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, f"CSV rejected: {exc}") from exc
        r.copilot.csv_path = dest
    return {"saved_as": dest.name, "kind": suffix[1:], "next": "POST /api/ingest"}


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """PII-scrub, chunk (250 tokens / 30 overlap) and embed PDFs + financial CSVs into ChromaDB; register them in SQLite."""
    r = rt()
    pdfs = sorted(r.cfg.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(409, "data room is empty: call POST /api/generate-demo or POST /api/upload first")
    if not r.job_lock.acquire(blocking=False):
        raise HTTPException(409, "another job is running")
    try:
        csvs = [p for p in [r.csv_path] + sorted((r.cfg.data_dir / "csv").glob("*.csv")) if p.exists()]
        pipeline = IngestPipeline(r.cfg, r.proxy, use_graph=req.use_graph)
        pipeline.embedder = r.embedder  # reuse the loaded model when there is one
        csv_reports: List[Dict[str, Any]] = []
        with r.monitor.stage("ingest"):
            report = pipeline.run(pdfs)
            with local_processing_guarantee(r.cfg.allowed_hosts):
                for path in csvs:
                    try:
                        csv_reports.append(ingest_financial_csv(path, pipeline))
                    except ValueError as exc:
                        csv_reports.append({"file": "(skipped)", "error": str(exc)})
        r.adopt(pipeline)
        r.ingest_audit = report.audit
        registered = r.register_documents(list(pdfs) + csvs)
        return {"pdf_documents": report.documents, "pdf_chunks": report.chunks, "coc_clauses": report.clauses,
                "max_chunk_tokens": report.max_chunk_tokens, "csv": csv_reports, "documents_registered": registered,
                "graph": "neo4j" if pipeline.graph else "disabled", "network_audit": report.audit}
    except HTTPException:
        raise
    except NetworkViolation as exc:
        raise HTTPException(403, f"air-gap violation blocked: {exc}") from exc
    except Exception as exc:  # noqa: BLE001  (e.g. Neo4j requested but not running)
        raise HTTPException(500, f"ingest failed: {type(exc).__name__}: {exc}") from exc
    finally:
        r.job_lock.release()


@app.get("/api/documents")
def documents() -> List[Dict[str, Any]]:
    """Registered documents (pseudonymous display names)."""
    return rt().db.list_documents()


@app.post("/api/audit/run")
def audit(req: AuditRequest) -> Dict[str, Any]:
    """Run the RAG audit on the quantized model (or rule fallback), gate it with the RAG Triad, persist retained findings."""
    r = rt()
    pair = r.stores()
    if not pair or pair[0].collection.count() == 0:
        raise HTTPException(409, "no ingested documents: call POST /api/ingest first")
    if not r.job_lock.acquire(blocking=False):
        raise HTTPException(409, "another job is running")
    chroma, embedder = pair
    try:
        llm = r.model.backend
        evaluator = TriadEvaluator(embedder, llm if r.model.live else None, TriadConfig(
            groundedness_threshold=req.groundedness_threshold, metrics_path=r.cfg.data_dir / "observability" / "triad_metrics.jsonl"))
        extractor = LocalRiskExtractor(HybridRetriever(embedder, chroma, r.graph, False, r.cfg.allowed_hosts), llm, evaluator)
        named, api_rows, triad_rows = [], [], []
        for q in AUDIT_QUERIES:
            try:
                with r.monitor.stage("rag-llm"):
                    res = extractor.extract(q.question, q.clause_types)
            except LowConfidenceRetractionException as exc:
                t = exc.report
                api_rows.append({"query": q.name, "status": "RETRACTED", "groundedness": round(t.groundedness, 3), "reasons": t.reasons[:3]})
                triad_rows.append({"name": q.name, "status": "RETRACTED", "context_relevance": t.context_relevance,
                                   "groundedness": t.groundedness, "answer_relevance": t.answer_relevance, "reasons": list(t.reasons)[:3]})
                continue
            t = res.triad
            status = "NO_CONTEXT" if t is None else "PARTIAL" if getattr(t, "partial", False) else "PASS"
            api_rows.append({"query": q.name, "status": status, "findings": len(res.findings),
                             "groundedness": round(t.groundedness, 3) if t else None})
            triad_rows.append({"name": q.name, "status": status, "context_relevance": getattr(t, "context_relevance", 0.0),
                               "groundedness": getattr(t, "groundedness", 0.0), "answer_relevance": getattr(t, "answer_relevance", 0.0)})
            named.append((q.name, res))
        stored = r.findings.persist(named, r.proxy)
        r.triad_rows = triad_rows
        try:
            with r.monitor.stage("drift"), local_processing_guarantee(r.cfg.allowed_hosts):
                r.drift = drift_gate(SimpleNamespace(chroma=chroma), sorted(r.cfg.pdf_dir.glob("*.pdf")))
        except Exception as exc:  # noqa: BLE001
            r.drift = {"status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}
        return {"model": r.model.report.model, "backend": r.model.report.backend, "queries": api_rows,
                "findings_stored": stored, "drift": r.drift.get("status")}
    except NetworkViolation as exc:
        raise HTTPException(403, f"air-gap violation blocked: {exc}") from exc
    finally:
        r.job_lock.release()


@app.get("/api/findings")
def list_findings(hitl_status: Optional[str] = Query(None, pattern="^(pending|approved|overridden|rejected)$")) -> List[Dict[str, Any]]:
    """Stored anonymized findings for review (prompts omitted)."""
    keep = ("finding_id", "category", "severity", "title", "summary", "evidence_quote", "status", "groundedness", "claim_score",
            "doc_type", "chunk_id", "pages", "hitl_status")
    return [{k: row[k] for k in keep} for row in rt().findings.list(hitl_status)]


@app.post("/api/hitl/resolve")
def hitl_resolve(req: HitlRequest) -> Dict[str, Any]:
    """Approve, edit or reject a finding; logs a DPO pair (JSONL + SQLite) and an audit-trail row."""
    r = rt()
    finding = r.findings.get(req.finding_id)
    if finding is None:
        raise HTTPException(404, "unknown finding_id")
    action = "override" if req.action in ("edit", "override") else req.action
    try:
        pair = build_pair_from_resolution(finding, action, req.override, req.rejected_text, r.proxy)
        note = r.proxy.anonymize(req.note)
        inserted, pair_id = r.dpo.add_pair(pair["prompt"], pair["chosen"], pair["rejected"], finding_id=req.finding_id, action=action,
                                           reviewer=req.reviewer, note=note, rejected_source=pair["rejected_source"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    status = {"approve": "approved", "override": "overridden", "reject": "rejected"}[action]
    r.findings.mark_resolved(req.finding_id, status, pair["human_output"])
    after = {"approve": finding["model_output"], "override": pair["human_output"], "reject": json.dumps({"risks": []})}[action]
    r.db.record_hitl(finding_id=req.finding_id, action=req.action, reviewer=req.reviewer, note=note,
                     before_json=finding["model_output"], after_json=after, pair_id=pair_id)
    return {"status": "recorded", "hitl_status": status, "pair_id": pair_id, "duplicate": not inserted,
            "rejected_source": pair["rejected_source"], "dpo": r.dpo.stats()}


@app.post("/api/negotiate")
def negotiate(req: NegotiateRequest) -> Dict[str, Any]:
    """Run the multi-agent consensus loop on the reviewed findings; store the memo in SQLite and on disk."""
    r = rt()
    usable = select_findings(r.findings.list(), req.require_review_for_partial)
    if not any(effective_item(f)["category"] == "CHANGE_OF_CONTROL" for f in usable):
        raise HTTPException(409, "no usable CHANGE_OF_CONTROL finding: run POST /api/audit/run (or approve/edit one via /api/hitl/resolve)")
    if not r.job_lock.acquire(blocking=False):
        raise HTTPException(409, "another job is running")
    try:
        terms = DealTerms(base_price_musd=req.base_price_musd, ebitda_musd=req.ebitda_musd, exercise_probability=req.exercise_probability)
        ledger = EgressLedger(r.proxy)
        try:
            with r.monitor.stage("agents"), local_processing_guarantee(r.cfg.allowed_hosts, allow_loopback=False):
                brief = brief_from_findings(usable, r.proxy, ledger)
                state, engine = run_consensus(brief, AuditLegalAgent(terms), MarketIntelligenceAgent(MockWebSearchTool(), ledger),
                                              FinancialAuditAgent(terms), terms)
        except AgentError as exc:
            raise HTTPException(422, f"agents could not price the deal: {exc}") from exc
        except (LeakageError, NetworkViolation) as exc:
            raise HTTPException(403, f"air-gap violation blocked: {exc}") from exc
        offer, neg = state["offer"], state["negotiation"]
        allowed, blocked = r.monitor.counts()
        evidence = {"ledger": ledger.audit(stage1_canaries()),
                    "network_summary": f"permitted(loopback/allow-listed)={allowed} blocked={blocked} clean={blocked == 0} (all stages, process-wide recorder)",
                    "network_blocked": blocked, "ingest_audit": r.ingest_audit, "proxy_stats": r.proxy.stats(),
                    "guard_selftest": bool(r.monitor.selftest)}
        memo = build_memo(MemoInputs(offer, neg, state["legal"], state["market"], state["financial"], brief,
                                     r.triad_rows or derive_triad_rows(usable), r.drift, evidence, engine))
        run_id = r.db.record_negotiation(
            base_price_musd=terms.base_price_musd, ebitda_musd=terms.ebitda_musd, exercise_probability=terms.exercise_probability,
            upfront_musd=offer.upfront_musd, earnout_musd=offer.earnout_musd, headline_musd=offer.headline_musd, engine=engine,
            statuses=offer.statuses, transcript=neg.transcript, memo_md=memo, evidence=evidence)
        out_dir = r.cfg.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "Investment_Recommendation_Memo.md").write_text(memo, encoding="utf-8")
        r.copilot.terms, r.copilot.sector_move_pct = terms, state["market"].sector_move_pct  # chat explains THIS run
        led = evidence["ledger"]
        return {"run_id": run_id, "engine": engine, "rounds": neg.round, "offer": dataclasses.asdict(offer),
                "mandate_target_met": abs(offer.upfront_musd - 34.5) < 0.05 and abs(offer.earnout_musd - 10.0) < 0.05,
                "findings_used": len(usable), "findings_excluded": len(r.findings.list()) - len(usable),
                "air_gap": {"clean": led["clean"] and blocked == 0, "leaks": len(led["leaks"]), "canary_hits": len(led["canary_hits"]),
                            "blocked_ops": blocked},
                "memo_markdown": memo}
    finally:
        r.job_lock.release()


@app.get("/api/negotiations/latest")
def latest_negotiation() -> Dict[str, Any]:
    """The most recent stored negotiation run with its memo."""
    row = rt().db.get_negotiation()
    if row is None:
        raise HTTPException(404, "no negotiation has run yet")
    return row


@app.get("/api/dpo/stats")
def dpo_stats() -> Dict[str, Any]:
    """Preference-pair counts."""
    return rt().dpo.stats()


@app.post("/api/dpo/step")
def dpo_step(req: DpoStepRequest) -> Dict[str, Any]:
    """Run DPO on the logged pairs (exact NumPy loss on CPU, or TRL + LoRA when a GPU is configured)."""
    try:
        return run_dpo_step(rt().dpo.fetch_pairs(), steps=req.steps, beta=req.beta, lr=req.lr, backend=req.backend,
                            allow_demo_seed=req.allow_demo_seed)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    """Ask the Agentic Copilot; the response is scrubbed and cites its source tools."""
    return rt().copilot.chat(req.message, req.session_id)


PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AegisDeal</title>
<style>body{font:14px system-ui;margin:0;padding:16px;background:#f6f7f9;color:#1b1f24;display:grid;grid-template-columns:1fr 1fr;gap:16px}
section{background:#fff;border:1px solid #d8dee4;border-radius:8px;padding:12px;min-width:0}h2{margin:0 0 8px;font-size:15px}
pre{white-space:pre-wrap;margin:0;font:12px ui-monospace,monospace}#out,#log{height:340px;overflow:auto;border:1px solid #d8dee4;padding:8px;border-radius:6px;margin-top:8px}
button{margin:2px 4px 2px 0}input[type=text]{width:68%}.f{border-top:1px solid #eaeef2;padding:6px 0}.m{margin:6px 0}.u{color:#0b5cad}
#tbl{border-collapse:collapse;font-size:11px;display:block;overflow:auto;max-height:200px}#tbl td,#tbl th{border:1px solid #d8dee4;padding:2px 6px}
#badge{font-weight:600}@media(max-width:900px){body{grid-template-columns:1fr}}</style>
<section><h2>AegisDeal <span id=badge></span></h2>
<button onclick="go('/api/generate-demo',{})">1 Generate demo data</button>
<button onclick="fin()">Inspect CSV</button><a href="/api/demo-financials">download CSV</a>
<input type=file id=up accept=".pdf,.csv" onchange="upl(this)">
<br><button onclick="go('/api/ingest',{})">2 Ingest</button><button onclick="go('/api/audit/run',{})">3 Run audit</button>
<button onclick="go('/api/negotiate',{})">5 Negotiate</button><button onclick="go('/api/health')">Health</button>
<button onclick="go('/api/dpo/step',{})">DPO step</button><button onclick="go('/api/quant?demo=true')">Quantization</button>
<table id=tbl></table><pre id=out></pre></section>
<section><h2>Findings (human-in-the-loop) &nbsp;<button onclick=load()>Refresh</button></h2><div id=fs></div>
<h2 style="margin-top:12px">Copilot</h2><div id=log></div><p><input type=text id=q placeholder="Why is the bid 34.5 + 10 earn-out?"><button onclick=ask()>Send</button></p></section>
<script>
const j=async(u,o)=>{const r=await fetch(u,o);const t=await r.json();if(!r.ok)throw new Error(typeof t.detail==='string'?t.detail:JSON.stringify(t.detail));return t};
const post=(u,b)=>j(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const out=t=>{document.getElementById('out').textContent=typeof t==='string'?t:JSON.stringify(t,null,2)};
async function go(u,b){out('working…');try{const r=b===undefined?await j(u):await post(u,b);
 if(r.memo_markdown){const {memo_markdown,...rest}=r;out(JSON.stringify(rest,null,2)+'\n\n'+memo_markdown)}else out(r);load();badge()}catch(e){out('Error: '+e.message)}}
async function badge(){try{const h=await j('/api/health');document.getElementById('badge').textContent='| air-gap: '+h.airgap.status+' | model: '+h.model.name+' ('+(h.model.live?'live':'rule fallback')+') | rows: '+Object.values(h.db).join('/')}catch(e){}}
async function upl(i){const f=i.files[0];if(!f)return;const fd=new FormData();fd.append('file',f);try{out(await j('/api/upload',{method:'POST',body:fd}))}catch(e){out('Error: '+e.message)}}
async function fin(){try{const rows=await j('/api/financials');const t=document.getElementById('tbl');t.textContent='';const cols=Object.keys(rows[0]||{});
 const h=t.insertRow();cols.forEach(c=>{const th=document.createElement('th');th.textContent=c;h.appendChild(th)});
 rows.forEach(r=>{const tr=t.insertRow();cols.forEach(c=>tr.insertCell().textContent=r[c])})}catch(e){out('Error: '+e.message)}}
async function load(){const fs=document.getElementById('fs');fs.textContent='';
 try{(await j('/api/findings')).forEach(f=>{const d=document.createElement('div');d.className='f';const p=document.createElement('div');
  p.textContent='['+f.severity+'] '+f.title+' — '+f.status+' / '+f.hitl_status+' (id '+f.finding_id+')';d.appendChild(p);
  [['approve','Approve'],['edit','Edit'],['reject','Reject']].forEach(([a,l])=>{const b=document.createElement('button');b.textContent=l;b.onclick=()=>resolve(f,a);d.appendChild(b)});fs.appendChild(d)})}catch(e){}}
async function resolve(f,a){const body={finding_id:f.finding_id,action:a};
 if(a==='edit'){const s=prompt('Corrected summary (the evidence quote is kept):',f.summary);if(!s)return;body.override={summary:s}}
 try{out(await post('/api/hitl/resolve',body));load()}catch(e){out('Error: '+e.message)}}
const log=document.getElementById('log');
function add(c,t){const d=document.createElement('div');d.className='m '+c;const p=document.createElement('pre');p.textContent=t;d.appendChild(p);log.appendChild(d);log.scrollTop=1e9}
async function ask(){const q=document.getElementById('q');const m=q.value.trim();if(!m)return;q.value='';add('u','You: '+m);
 try{const r=await post('/api/chat',{message:m});add('a',r.answer+'\n\n[tools: '+r.tools_called.join(', ')+' | redactions: '+JSON.stringify(r.redactions)+']')}catch(e){add('a','Error: '+e.message)}}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
load();badge();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Single-page demo UI."""
    return PAGE


def main() -> None:
    """CLI entry point: ``python -m backend.server``."""
    global DATA_DIR
    parser = argparse.ArgumentParser(description="AegisDeal FastAPI server (local only).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workdir", type=Path, default=ApexConfig().data_dir)
    args = parser.parse_args()
    DATA_DIR = args.workdir
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()