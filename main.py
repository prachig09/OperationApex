"""Operation Apex: end-to-end pipeline.

PDF generation -> scrubbing/ingest (Stage 1) -> local RAG + triad gate (Stage 2) ->
drift check -> agent consensus (Stage 3) -> executive memo.
"""
from __future__ import annotations

from security.network_guard import (  # must be imported before any ML library
    NetworkViolation, local_processing_guarantee, verify_guard_blocks_egress,
)

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agents.core import (
    AgentError, DealTerms, EgressLedger, MockWebSearchTool, build_brief, stage1_canaries, stage1_known_entities,
)
from agents.graph import run_consensus
from agents.nodes import AuditLegalAgent, FinancialAuditAgent, MarketIntelligenceAgent
from config import ApexConfig
from data.synthetic_gen import generate_all
from llm.extractor import (
    AUDIT_QUERIES, HybridRetriever, LLMConfig, LLMUnavailableError, LocalLLMClient, LocalRiskExtractor,
)
from observability.drift_monitor import DataDriftWarning, SemanticDriftMonitor
from observability.triad_eval import LowConfidenceRetractionException, TriadConfig, TriadEvaluator
from pipeline.ingest import IngestPipeline
from report.synthesizer import MemoInputs, build_memo
from security.proxy import AirGappedProxy, LeakageError

LOG = logging.getLogger("apex.main")


def step(number: int, title: str) -> None:
    """Print a progress banner."""
    print(f"\n[{number}/8] {title}", flush=True)


def load_or_create_proxy_key(workdir: Path) -> bytes:
    """Return a stable proxy key (env var, else a 0600 file in the workdir) so placeholders persist across runs."""
    env = os.environ.get("APEX_PROXY_KEY")
    if env:
        return env.encode()
    path = workdir / ".proxy_key"
    if path.exists():
        return path.read_bytes().strip()
    workdir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32).encode()
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def drift_gate(pipeline: IngestPipeline, pdfs: Sequence[Path]) -> Dict[str, Any]:
    """Leave-one-document-out drift check: baseline from the other documents, test the last one."""
    holdout = pdfs[-1]
    doc_id = hashlib.sha256(holdout.read_bytes()).hexdigest()[:16]
    try:
        monitor = SemanticDriftMonitor.from_chroma(pipeline.chroma, exclude_doc_ids=[doc_id])
    except ValueError as exc:
        return {"status": "SKIPPED", "reason": str(exc)}
    got = pipeline.chroma.collection.get(where={"doc_id": doc_id}, include=["embeddings"])
    embeddings = got.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return {"status": "SKIPPED", "reason": "held-out document has no stored embeddings"}
    vectors = np.asarray(embeddings, dtype=np.float64)
    try:
        report, status = monitor.check(vectors, mode="raise"), "STABLE"
    except DataDriftWarning as warning:
        report, status = warning.report, "DRIFT_WARNING"
    return {"status": status, "holdout": holdout.name, **report.to_dict()}


def run_audit(extractor: LocalRiskExtractor) -> tuple:
    """Run the audit battery query by query; retracted answers are excluded and recorded."""
    results, rows = [], []
    for q in AUDIT_QUERIES:
        try:
            res = extractor.extract(q.question, q.clause_types)
        except LowConfidenceRetractionException as exc:
            r = exc.report
            rows.append({"name": q.name, "status": "RETRACTED", "context_relevance": r.context_relevance,
                         "groundedness": r.groundedness, "answer_relevance": r.answer_relevance, "reasons": r.reasons})
            LOG.warning("query '%s' retracted: %s", q.name, exc)
            continue
        results.append(res)
        t = res.triad
        status = "NO_CONTEXT" if not t else ("PARTIAL" if t.partial else "PASSED")
        rows.append({"name": q.name, "status": status,
                     "context_relevance": t.context_relevance if t else 0.0,
                     "groundedness": t.groundedness if t else 0.0, "answer_relevance": t.answer_relevance if t else 0.0})
    return results, rows


def execute(args: argparse.Namespace) -> int:
    """Run the full pipeline and return a process exit code."""
    cfg = dataclasses.replace(ApexConfig(), data_dir=args.workdir)
    terms = DealTerms(base_price_musd=args.base_price, ebitda_musd=args.ebitda, exercise_probability=args.exercise_probability)

    step(1, "Local Processing Guarantee self-test")
    if not verify_guard_blocks_egress():
        print("FATAL: egress guard self-test failed; refusing to run.", file=sys.stderr)
        return 2

    step(2, "Generating synthetic data room")
    pdf_dir = cfg.pdf_dir
    pdfs = sorted(pdf_dir.glob("*.pdf")) if args.skip_generate else generate_all(pdf_dir)
    pdfs = sorted(pdfs)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1
    print(f"  {len(pdfs)} PDFs in {pdf_dir}")

    pipeline: Optional[IngestPipeline] = None
    try:
        with local_processing_guarantee(cfg.allowed_hosts, allow_loopback=True) as net:
            step(3, "Anonymization proxy + ETL ingest (parse -> chunk -> embed -> Chroma/Neo4j)")
            proxy = AirGappedProxy(cfg.spacy_model, secret_key=load_or_create_proxy_key(cfg.data_dir),
                                   known_entities=stage1_known_entities())
            pipeline = IngestPipeline(cfg, proxy, use_graph=args.graph or cfg.use_graph)   
            ingest = pipeline.run(pdfs)
            print(f"  documents={ingest.documents} chunks={ingest.chunks} coc_clauses={ingest.clauses} max_tokens={ingest.max_chunk_tokens}")

            step(4, "Semantic drift gate")
            drift = drift_gate(pipeline, pdfs)
            print(f"  drift status: {drift['status']}")

            step(5, "Local RAG extraction + RAG Triad gate")
            overrides = {k: v for k, v in (("backend", args.llm_backend), ("model", args.llm_model), ("base_url", args.llm_url)) if v}
            llm = LocalLLMClient(dataclasses.replace(LLMConfig(), **overrides), cfg.allowed_hosts)
            if not llm.ping():
                raise LLMUnavailableError(f"No local model server at {llm.cfg.url} ({llm.cfg.backend}).")
            evaluator = TriadEvaluator(pipeline.embedder, llm, TriadConfig(
                groundedness_threshold=args.groundedness_threshold,
                metrics_path=cfg.data_dir / "observability" / "triad_metrics.jsonl"))
            retriever = HybridRetriever(pipeline.embedder, pipeline.chroma, pipeline.graph, False, cfg.allowed_hosts)
            extractor = LocalRiskExtractor(retriever, llm, evaluator)
            results, triad_rows = run_audit(extractor)
            for row in triad_rows:
                print(f"  {row['name']:<24} {row['status']:<9} grounded={row['groundedness']:.2f}")
            if not any(f.category == "CHANGE_OF_CONTROL" for r in results for f in r.findings):
                print("FATAL: no verified Change of Control finding survived the triad gate.", file=sys.stderr)
                return 5

            step(6, "Scrubbing findings through the proxy for the agents")
            ledger = EgressLedger(proxy)
            brief = build_brief(results, proxy, ledger)
            print(f"  {len(brief.findings)} anonymized findings; sector terms: {brief.sector_terms}")

            step(7, "Multi-agent consensus loop")
            state, engine = run_consensus(
                brief, AuditLegalAgent(terms), MarketIntelligenceAgent(MockWebSearchTool(), ledger),
                FinancialAuditAgent(terms), terms)
            offer = state["offer"]
            print(f"  engine={engine} rounds={state['negotiation'].round} statuses={offer.statuses}")

            step(8, "Synthesizing executive memo")
            evidence = {
                "ledger": ledger.audit(stage1_canaries()), "network_summary": net.summary(), "network_blocked": len(net.blocked),
                "ingest_audit": ingest.audit, "proxy_stats": proxy.stats(), "guard_selftest": True,
            }
            memo = build_memo(MemoInputs(offer, state["negotiation"], state["legal"], state["market"], state["financial"],
                                         brief, triad_rows, drift, evidence, engine))
            out_dir = cfg.data_dir / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            memo_path = out_dir / "Investment_Recommendation_Memo.md"
            memo_path.write_text(memo, encoding="utf-8")
            (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
            if args.restore:
                (out_dir / "memo_restored.md").write_text(proxy.restore(memo), encoding="utf-8")

        print(f"\nFinal bid: ${offer.upfront_musd:.1f}M upfront + ${offer.earnout_musd:.1f}M earn-out (max ${offer.headline_musd:.1f}M)")
        print(f"Mandate target (34.5 + 10.0) met: {abs(offer.upfront_musd - 34.5) < 0.05 and abs(offer.earnout_musd - 10.0) < 0.05}")
        print(f"Air-gap evidence: {'PASS' if evidence['ledger']['clean'] and evidence['network_blocked'] == 0 else 'FAIL'}")
        print(f"Memo: {memo_path}")
        return 0
    except (LeakageError, NetworkViolation) as exc:
        print(f"FATAL: air-gap violation: {exc}", file=sys.stderr)
        return 3
    except AgentError as exc:
        print(f"FATAL: agent input unusable: {exc}", file=sys.stderr)
        return 5
    except LLMUnavailableError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 6
    finally:
        if pipeline is not None:
            pipeline.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Operation Apex end-to-end pipeline.")
    parser.add_argument("--workdir", type=Path, default=ApexConfig().data_dir)
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing PDFs in <workdir>/pdfs.")
    parser.add_argument("--graph", action="store_true", help="Also write to Neo4j (off by default).")   # replaces --no-graph
    parser.add_argument("--restore",action="store_true",default=False,help="De-anonymize PII placeholders in the final executive memo.",)
    parser.add_argument("--llm-backend", choices=["ollama", "vllm"])
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-url")
    parser.add_argument("--groundedness-threshold", type=float, default=0.75)
    parser.add_argument("--base-price", type=float, default=50.0, help="Opening base price, USD M.")
    parser.add_argument("--ebitda", type=float, default=6.25, help="Baseline EBITDA, USD M.")
    parser.add_argument("--exercise-probability", type=float, default=0.325,
                        help="Probability weight on the gross churn exposure (calibration input).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())