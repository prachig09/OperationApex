# AegisDeal Refactoring Log

Three architectural refactorings made after the first end-to-end runs. Each entry gives the problem, the change, the trade-off and a way to verify it.

## 1. Chunking 500 -> 250 tokens (overlap 50 -> 30) and top_k = 3

**Problem.** With a 3B-7B local model, answers failed the RAG Triad gate (groundedness 0.00 on several queries). Large
chunks (500 tokens) plus 5-10 retrieved excerpts gave the small model long, noisy contexts. It paraphrased instead of
quoting, and the verbatim-quote check then scored 0.

**Change.**
- `config.py`: `chunk_tokens=250`, `chunk_overlap=30`, `retrieval_top_k=3`, `max_context_chunks=5`.
- `LLMConfig.num_ctx` 12288 -> 4096.
- A strict verbatim-extraction prompt.
- Fuzzy quote matching in `triad_eval.py` with per-claim retention (claims scoring >= 0.50 with a verified quote survive).
- `ChromaStore.delete_document` / `Neo4jStore.delete_document`, so re-ingesting re-chunked files leaves no stale vectors or `Clause` nodes.

**Trade-offs.** More chunks per document. Clauses that span a chunk boundary are only partly covered by the 30-token overlap.
Existing stores and `baseline.npz` must be rebuilt.

**Verify.**
- `python -m pipeline.ingest --verify` prints `max_chunk_tokens` (must be <= 250).
- `apex_data/observability/triad_metrics.jsonl` records `quote_support` and `issues` per claim.

## 2. Neo4j made optional: SQLite + ChromaDB as the zero-setup datastore

**Problem.** A mandatory Neo4j container made the system hard to deploy on CPU-only or low-resource hosts. The graph was used only
for clause lookup, neighbour expansion and entity mentions.

**Change.**
- `use_graph` is off by default (`--graph` or `APEX_USE_GRAPH=1` enables it). The `neo4j` package is imported lazily.
- Clause-tagged chunk lookup moved to a ChromaDB metadata filter (`has_change_of_control`, in `HybridRetriever._chroma_clauses`).
  The Neo4j lookup remains when a graph is attached.
- Neighbour expansion stays Neo4j-only and is disabled by default (`expand_neighbors=0`).
- Financial-CSV chunks (one per quarter) are ingested into ChromaDB by the same pipeline.

**Trade-offs.** `NEXT`-chain neighbours and `MENTIONS` entity traversal need `--graph`. Clause filtering in Chroma currently
covers only Change of Control.

**Not claimed.** The existing Cypher uses comma-separated `MATCH` clauses, each anchored on a unique-constraint
index (one row per side). This is not a Cartesian-product performance problem, so we do not present it as one.

**Verify.** `GET /api/health` reports `"graph": "disabled (SQLite + ChromaDB)"`. Running the audit without Neo4j still
retrieves the Change-of-Control chunks.

## 3. SQLite state persistence (`apex_data/aegisdeal.db`)

**Problem.** Pipeline state lived in process memory or in separate files, so a restart lost findings, reviewer
decisions and negotiation results.

**Change.** One SQLite file (WAL mode, busy timeout) with five tables: `documents`, `audit_findings`, `hitl_decisions`,
`dpo_pairs`, `negotiation_runs`. `audit_findings` and `dpo_pairs` keep the DDL owned by `FindingsStore` and `DPOStore`
(imported, not copied). The human-in-the-loop endpoint writes the audit-trail row and the DPO preference pair, and the
negotiation reads human-reviewed findings (rejected ones are excluded; edited ones use the human's text). Everything
stored is proxy-anonymized, and documents use pseudonymous display names.

**Trade-offs.** Process-global single-user state. The old `aegis.sqlite` is not migrated. SQLite allows one writer at a time,
which is fine for a single-user demo.

**Verify.**
```bash
curl -s localhost:8000/api/health            # "db" row counts per table
sqlite3 apex_data/aegisdeal.db "SELECT action, COUNT(*) FROM hitl_decisions GROUP BY action;"
# restart the server: the counts are unchanged
```