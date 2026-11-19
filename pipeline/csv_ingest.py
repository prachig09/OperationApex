"""Financial-CSV ingestion: one dense chunk per quarter -> proxy scrub -> local embedding -> ChromaDB (+ Neo4j if enabled).

Reuses the Stage 1 ``Chunk`` dataclass, ``IngestPipeline._enrich`` (proxy scrubbing) and the Chroma/Neo4j stores.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data.synthetic_gen import validate_financials_csv
from pipeline.ingest import Chunk, IngestPipeline
from security.network_guard import local_processing_guarantee

DOC_TYPE = "FINANCIALS"


def quarter_text(row: Dict[str, str]) -> Tuple[str, str]:
    """Return ``(section_title, body)`` describing one quarter in plain sentences."""
    f = lambda col: float(row[col])  # noqa: E731
    section = f"Quarterly financials {row['quarter']}"
    body = (
        f"{row['quarter']} (period ending {row['period_end']}): total revenue USD {f('total_revenue_musd'):,.1f}M; "
        f"ARR USD {f('arr_musd'):,.1f}M; net new ARR USD {f('net_new_arr_musd'):,.1f}M; "
        f"EBITDA USD {f('ebitda_musd'):,.1f}M ({f('ebitda_margin_pct'):.1f}% margin). "
        f"Customer concentration: Northgate Logistics {f('northgate_share_pct'):.1f}% of revenue, "
        f"Solara Retail Partners {f('solara_share_pct'):.1f}%, Halcyon Pharma Services {f('halcyon_share_pct'):.1f}%; "
        f"top three customers {f('top3_share_pct'):.1f}% of revenue. "
        f"Gross revenue churn {f('gross_churn_pct'):.2f}% in the quarter; net revenue retention "
        f"{f('net_revenue_retention_pct'):.1f}%; {int(float(row['logo_churn_count']))} logos churned out of "
        f"{int(float(row['customers_total']))} customers."
    )
    return section, body


def ingest_financial_csv(path: Path, pipeline: IngestPipeline) -> Dict[str, Any]:
    """Chunk, scrub, embed and store a financials CSV.

    Call after ``pipeline.run`` so ``pipeline.embedder`` and ``pipeline.chroma`` exist.

    Raises:
        ValueError: if the CSV does not match ``FINANCIAL_COLUMNS``.
    """
    rows = validate_financials_csv(path)
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    doc_id = sha[:16]
    filename = f"{DOC_TYPE}_{doc_id[:8]}.csv"  # pseudonymous: uploaded names are never stored
    tokenizer = pipeline.embedder.tokenizer
    chunks: List[Chunk] = []
    with local_processing_guarantee(allow_loopback=False):
        for i, row in enumerate(rows):
            section, body = quarter_text(row)
            chunk = Chunk(chunk_id=f"{doc_id}-{i:04d}", doc_id=doc_id, filename=filename, index=i,
                          page_start=i + 1, page_end=i + 1, section=section, text=f"{section}\n{body}")
            chunk.token_count = len(tokenizer(chunk.text, add_special_tokens=False, verbose=False)["input_ids"])
            pipeline._enrich(chunk)  # noqa: SLF001  (proxy scrub -> anon_text + entities)
            chunks.append(chunk)
        embeddings = pipeline.embedder.embed([c.text for c in chunks])
    with local_processing_guarantee(pipeline.cfg.allowed_hosts):
        pipeline.chroma.delete_document(doc_id)
        pipeline.chroma.upsert(chunks, embeddings, DOC_TYPE)
        if pipeline.graph:
            pipeline.graph.delete_document(doc_id)
            pipeline.graph.write_document({"doc_id": doc_id, "filename": filename, "doc_type": DOC_TYPE, "sha256": sha}, chunks)
    tokens = [c.token_count for c in chunks]
    return {"file": filename, "doc_id": doc_id, "chunks": len(chunks), "max_tokens": max(tokens),
            "mean_tokens": round(sum(tokens) / len(tokens), 1)}