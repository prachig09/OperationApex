"""Stage 1 ETL: PDF -> semantic chunks -> local embeddings -> ChromaDB + Neo4j.

Parsing, chunking and embedding run under a zero-socket guarantee. Only the
database stage may touch loopback (or explicitly allow-listed) hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import ApexConfig
from security.network_guard import (  # noqa: I001  (must import before transformers)
    NetworkViolation,
    local_processing_guarantee,
    verify_guard_blocks_egress,
)
from security.proxy import AirGappedProxy

LOG = logging.getLogger("apex.ingest")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_NOISE = re.compile(r"^\s*APEX DATA ROOM\b.*$", re.I)
_NUMBERED_HEADING = re.compile(r"^(?:ITEM\s+\d+[A-C]?\.|(?:ARTICLE|SECTION)\s+\d+|\d+(?:\.\d+)*\.?)\s+\S", re.I)
_COC = re.compile(r"change\s+(?:of|in)\s+control", re.I)
_NO_PENALTY = re.compile(r"terminat\w*[^.]{0,300}?without\s+(?:any\s+)?(?:penalty|liability|termination\s+fee)", re.I | re.S)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[“\"])")


# ------------------------------------------------------------------ data model
@dataclass
class ParsedElement:
    """A parsed document element."""

    text: str
    category: str
    page: int


@dataclass
class Section:
    """A heading-delimited group of narrative elements."""

    title: str
    body: List[str]
    page_start: int
    page_end: int


@dataclass
class Chunk:
    """A retrieval chunk with provenance, anonymized text, entities and flags."""

    chunk_id: str
    doc_id: str
    filename: str
    index: int
    page_start: int
    page_end: int
    section: str
    text: str
    token_count: int = 0
    anon_text: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    clauses: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IngestReport:
    """Summary of one pipeline run."""

    documents: int = 0
    chunks: int = 0
    clauses: int = 0
    max_chunk_tokens: int = 0
    audit: str = ""


# --------------------------------------------------------------------- parsing
def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80 or s[-1] in ".;,:":
        return False
    if s.isupper() and len(s) > 3:
        return True
    if _NUMBERED_HEADING.match(s) and ". " not in s and len(s.split()) <= 12:
        return True
    words = [w for w in s.split() if w[0].isalpha()]
    return bool(words) and len(s.split()) <= 12 and sum(w[0].isupper() for w in words) / len(words) >= 0.7


def _parse_with_unstructured(path: Path) -> List[ParsedElement]:
    from unstructured.partition.pdf import partition_pdf

    out: List[ParsedElement] = []
    for el in partition_pdf(filename=str(path), strategy="fast"):
        text = (el.text or "").strip()
        if not text or _NOISE.match(text) or el.category in ("Footer", "PageBreak"):
            continue
        page = int(getattr(el.metadata, "page_number", 1) or 1)
        out.append(ParsedElement(text, el.category, page))
    return out


def _parse_with_pypdf2(path: Path) -> List[ParsedElement]:
    from PyPDF2 import PdfReader

    elements: List[ParsedElement] = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        buffer: List[str] = []
        for line in (page.extract_text() or "").splitlines():
            if not line.strip() or _NOISE.match(line):
                continue
            if _looks_like_heading(line):
                if buffer:
                    elements.append(ParsedElement(" ".join(buffer), "NarrativeText", page_no))
                    buffer = []
                elements.append(ParsedElement(line.strip(), "Title", page_no))
            else:
                buffer.append(line.strip())
        if buffer:
            elements.append(ParsedElement(" ".join(buffer), "NarrativeText", page_no))
    return elements


def parse_document(path: Path) -> List[ParsedElement]:
    """Parse a PDF under a zero-socket guarantee (Unstructured, else PyPDF2).

    If Unstructured tries to reach the network (e.g. to download NLTK data), the
    attempt is blocked and the PyPDF2 fallback is used instead.
    """
    try:
        with local_processing_guarantee(allow_loopback=False):
            elements = _parse_with_unstructured(path)
        if elements:
            return elements
    except ImportError:
        LOG.info("unstructured not installed; using PyPDF2 parser")
    except NetworkViolation as exc:
        LOG.warning("unstructured attempted network access (blocked): %s; falling back to PyPDF2", exc)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("unstructured failed on %s (%s); falling back to PyPDF2", path.name, exc)
    with local_processing_guarantee(allow_loopback=False):
        return _parse_with_pypdf2(path)


def group_sections(elements: Sequence[ParsedElement], fallback_title: str) -> List[Section]:
    """Group elements into heading-delimited sections (nested headings are joined with ' | ')."""
    sections: List[Section] = []
    current = Section(fallback_title, [], 1, 1)
    for el in elements:
        if el.category in ("Title", "Header"):
            if current.body:
                sections.append(current)
                current = Section(el.text, [], el.page, el.page)
            elif current.title == fallback_title:
                current = Section(el.text, [], el.page, el.page)
            else:
                current = Section(f"{current.title} | {el.text}", [], current.page_start, el.page)
        else:
            current.body.append(el.text)
            current.page_end = el.page
    if current.body:
        sections.append(current)
    return sections


# -------------------------------------------------------------------- chunking
class SemanticChunker:
    """Sentence-aware chunker: <= max_tokens per chunk, ~overlap tokens carried over."""

    def __init__(
        self, tokenizer: Any, max_tokens: int = ApexConfig.chunk_tokens, overlap: int = ApexConfig.chunk_overlap
    ) -> None:
        """Create a chunker bound to the embedding model's own tokenizer (defaults: 250 tokens, 30 overlap)."""
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.tok, self.max_tokens, self.overlap = tokenizer, max_tokens, overlap

    def _ids(self, texts: List[str]) -> List[List[int]]:
        if not texts:
            return []
        return self.tok(texts, add_special_tokens=False, verbose=False)["input_ids"]

    def _decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _tail(self, text: str) -> Tuple[str, int]:
        ids = self._ids([text])[0][-self.overlap:]
        return self._decode(ids), len(ids)

    def split_text(self, text: str, budget: int) -> List[str]:
        """Split ``text`` into chunks of at most ``budget`` tokens with overlap."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        units: List[Tuple[str, int]] = []
        for sentence, ids in zip(sentences, self._ids(sentences)):
            if len(ids) <= budget:
                units.append((sentence, len(ids)))
                continue
            stride = max(1, budget - self.overlap)
            for start in range(0, len(ids), stride):
                window = ids[start:start + budget]
                units.append((self._decode(window), len(window)))
                if start + budget >= len(ids):
                    break

        chunks: List[str] = []
        current: List[Tuple[str, int]] = []
        current_n, fresh = 0, False
        for unit_text, n in units:
            if current and current_n + n > budget:
                chunks.append(" ".join(t for t, _ in current))
                keep: List[Tuple[str, int]] = []
                keep_n = 0
                for t, m in reversed(current):
                    if keep_n + m > self.overlap:
                        break
                    keep.insert(0, (t, m))
                    keep_n += m
                if not keep:
                    tail_text, tail_n = self._tail(current[-1][0])
                    keep, keep_n = [(tail_text, tail_n)], tail_n
                while keep and keep_n + n > budget:
                    _, dropped = keep.pop(0)
                    keep_n -= dropped
                current, current_n, fresh = keep, keep_n, False
            current.append((unit_text, n))
            current_n += n
            fresh = True
        if current and fresh:
            chunks.append(" ".join(t for t, _ in current))
        return chunks

    def chunk_document(self, doc_id: str, filename: str, sections: Sequence[Section]) -> List[Chunk]:
        """Chunk each section, prefixing every chunk with its header for context."""
        chunks: List[Chunk] = []
        for sec in sections:
            header = sec.title[:200].strip()
            header_n = len(self._ids([header])[0]) if header else 0
            budget = max(64, self.max_tokens - header_n - 1)
            for body in self.split_text(" ".join(sec.body), budget):
                text = f"{header}\n{body}" if header else body
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}-{len(chunks):04d}", doc_id=doc_id, filename=filename,
                    index=len(chunks), page_start=sec.page_start, page_end=sec.page_end,
                    section=header, text=text,
                ))
        for chunk, ids in zip(chunks, self._ids([c.text for c in chunks])):
            chunk.token_count = len(ids)
        return chunks


# ------------------------------------------------------------------- embedding
class LocalEmbedder:
    """Local HuggingFace embedder (bge-large-en-v1.5). Never contacts the Hub."""

    def __init__(self, model_path: str, batch_size: int = 16, device: Optional[str] = None) -> None:
        """Load the model strictly from a local directory."""
        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"Embedding model not found at '{model_path}'. Pre-download BAAI/bge-large-en-v1.5 "
                "on a staging host and copy it here (air-gapped hosts never fetch models)."
            )
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    @property
    def tokenizer(self) -> Any:
        """The model's own tokenizer (used for exact chunk sizing)."""
        return self.model.tokenizer

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return L2-normalized embeddings."""
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


# ---------------------------------------------------------------------- stores
class ChromaStore:
    """Local persistent ChromaDB collection."""

    def __init__(self, persist_dir: Path, collection: str) -> None:
        """Open (or create) the persistent collection with telemetry disabled."""
        import chromadb
        from chromadb.config import Settings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], doc_type: str) -> None:
        """Upsert chunks in batches; ``anon_text`` is the only egress-eligible text."""
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=[list(e) for e in embeddings[start:start + 64]],
                documents=[c.text for c in batch],
                metadatas=[{
                    "doc_id": c.doc_id, "filename": c.filename, "doc_type": doc_type,
                    "chunk_index": c.index, "page_start": c.page_start, "page_end": c.page_end,
                    "section": c.section, "token_count": c.token_count,
                    "has_change_of_control": bool(c.clauses),
                    "entity_labels": ",".join(sorted({e["label"] for e in c.entities})),
                    "anon_text": c.anon_text,
                } for c in batch],
            )
    def delete_document(self, doc_id: str) -> None:
        """Remove every stored chunk of a document so re-chunked re-ingestion leaves no stale vectors."""
        self.collection.delete(where={"doc_id": doc_id}
                               )
        


class Neo4jStore:
    """Self-hosted Neo4j knowledge graph writer.

    Schema:
        (Document)-[:HAS_CHUNK]->(Chunk)-[:NEXT]->(Chunk)
        (Chunk)-[:MENTIONS {count}]->(Entity {alias, label, name})
        (Chunk)-[:CONTAINS_CLAUSE]->(Clause)<-[:HAS_CLAUSE]-(Document)
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Connect and verify the Bolt endpoint."""
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def init_schema(self) -> None:
        """Create uniqueness constraints (idempotent)."""
        statements = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_alias IF NOT EXISTS FOR (e:Entity) REQUIRE e.alias IS UNIQUE",
            "CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (c:Clause) REQUIRE c.clause_id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)

    def write_document(self, doc: Dict[str, Any], chunks: Sequence[Chunk]) -> None:
        """Write one document, its chunks, entities and detected clauses in one transaction."""
        chunk_rows = [{
            "chunk_id": c.chunk_id, "idx": c.index, "page_start": c.page_start, "page_end": c.page_end,
            "section": c.section, "token_count": c.token_count, "text": c.text, "anon_text": c.anon_text,
        } for c in chunks]
        links = [{"a": a.chunk_id, "b": b.chunk_id} for a, b in zip(chunks, chunks[1:])]
        mentions = [{"chunk_id": c.chunk_id, **e} for c in chunks for e in c.entities]
        clauses = [{"chunk_id": c.chunk_id, **cl} for c in chunks for cl in c.clauses]

        def work(tx: Any) -> None:
            tx.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.filename = $filename, d.doc_type = $doc_type, d.sha256 = $sha256, d.ingested_at = datetime()
                WITH d UNWIND $chunks AS c
                MERGE (k:Chunk {chunk_id: c.chunk_id})
                SET k.idx = c.idx, k.page_start = c.page_start, k.page_end = c.page_end, k.section = c.section,
                    k.token_count = c.token_count, k.text = c.text, k.anon_text = c.anon_text
                MERGE (d)-[:HAS_CHUNK]->(k)
                """,
                chunks=chunk_rows, **doc,
            )
            tx.run(
                "UNWIND $links AS l MATCH (a:Chunk {chunk_id: l.a}), (b:Chunk {chunk_id: l.b}) MERGE (a)-[:NEXT]->(b)",
                links=links,
            )
            tx.run(
                """
                UNWIND $mentions AS m
                MATCH (k:Chunk {chunk_id: m.chunk_id})
                MERGE (e:Entity {alias: m.alias})
                ON CREATE SET e.name = m.name, e.label = m.label
                MERGE (k)-[r:MENTIONS]->(e) SET r.count = m.count
                """,
                mentions=mentions,
            )
            tx.run(
                """
                UNWIND $clauses AS cl
                MATCH (k:Chunk {chunk_id: cl.chunk_id}), (d:Document {doc_id: $doc_id})
                MERGE (c:Clause {clause_id: cl.clause_id})
                SET c.type = cl.type, c.severity = cl.severity, c.excerpt = cl.excerpt, c.excerpt_anon = cl.excerpt_anon
                MERGE (k)-[:CONTAINS_CLAUSE]->(c)
                MERGE (d)-[:HAS_CLAUSE]->(c)
                """,
                clauses=clauses, doc_id=doc["doc_id"],
            )

        with self.driver.session() as session:
            session.execute_write(work)

    def count_clauses(self, clause_type: str = "CHANGE_OF_CONTROL") -> int:
        """Return the number of stored clauses of a given type."""
        with self.driver.session() as session:
            return session.run("MATCH (c:Clause {type: $t}) RETURN count(c) AS n", t=clause_type).single()["n"]

    def close(self) -> None:
        """Close the driver."""
        self.driver.close()

    def delete_document(self, doc_id: str) -> None:
        """Remove a document with its chunks and clauses (shared Entity nodes are kept)."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(k:Chunk)
                OPTIONAL MATCH (d)-[:HAS_CLAUSE]->(c:Clause)
                DETACH DELETE c, k, d
                """,
                doc_id=doc_id,
            )


# -------------------------------------------------------------------- pipeline
def detect_clauses(text: str, chunk_id: str, proxy: AirGappedProxy) -> List[Dict[str, Any]]:
    """Deterministically tag Change-of-Control language in a chunk."""
    match = _COC.search(text)
    if not match:
        return []
    excerpt = " ".join(text[max(0, match.start() - 200): match.end() + 400].split())
    return [{
        "clause_id": f"{chunk_id}:CHANGE_OF_CONTROL",
        "type": "CHANGE_OF_CONTROL",
        "severity": "HIGH" if _NO_PENALTY.search(text) else "MEDIUM",
        "excerpt": excerpt,
        "excerpt_anon": proxy.anonymize(excerpt),
    }]


def infer_doc_type(filename: str) -> str:
    """Infer a coarse document type from the file name."""
    name = filename.upper()
    if "10K" in name or "10-K" in name:
        return "10-K"
    if "MSA" in name:
        return "MSA"
    if "SCHEDULE" in name or "AMENDMENT" in name:
        return "SCHEDULE"
    return "CONTRACT"


class IngestPipeline:
    """Orchestrates parse -> chunk -> enrich -> embed -> Chroma + Neo4j."""

    def __init__(self, cfg: ApexConfig, proxy: AirGappedProxy, use_graph: bool = True) -> None:
        """Bind configuration, proxy and graph toggle."""
        self.cfg, self.proxy, self.use_graph = cfg, proxy, use_graph
        self.embedder = LocalEmbedder(self.cfg.embedding_model_path, self.cfg.embed_batch_size)        
        self.chroma: Optional[ChromaStore] = None
        self.graph: Optional[Neo4jStore] = None

    def _enrich(self, chunk: Chunk) -> None:
        scrubbed, spans = self.proxy.scrub(chunk.text)
        chunk.anon_text = scrubbed
        entities: Dict[str, Dict[str, Any]] = {}
        for span in spans:
            alias = self.proxy.placeholder_for(span.label, span.text)
            row = entities.setdefault(alias, {"alias": alias, "label": span.label, "name": span.text.strip(), "count": 0})
            row["count"] += 1
        chunk.entities = list(entities.values())
        chunk.clauses = detect_clauses(chunk.text, chunk.chunk_id, self.proxy)

    def run(self, pdf_paths: Sequence[Path]) -> IngestReport:
        """Ingest PDFs. Parsing/embedding use zero sockets; DB writes use loopback only.

        Each document is purged from both stores before writing, so changing the chunk size is safe.
        """
        report = IngestReport()
        with local_processing_guarantee(self.cfg.allowed_hosts) as audit:
            with local_processing_guarantee(allow_loopback=False):
                self.embedder = LocalEmbedder(self.cfg.embedding_model_path, self.cfg.embed_batch_size)
                chunker = SemanticChunker(self.embedder.tokenizer, self.cfg.chunk_tokens, self.cfg.chunk_overlap)
            self.chroma = ChromaStore(self.cfg.chroma_dir, self.cfg.chroma_collection)
            if self.use_graph:
                self.graph = Neo4jStore(self.cfg.neo4j_uri, self.cfg.neo4j_user, self.cfg.neo4j_password)
                self.graph.init_schema()

            for path in pdf_paths:
                raw = path.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                doc_id = sha[:16]
                doc_type = infer_doc_type(path.name)
                elements = parse_document(path)
                with local_processing_guarantee(allow_loopback=False):
                    sections = group_sections(elements, fallback_title=path.stem)
                    chunks = chunker.chunk_document(doc_id, path.name, sections)
                    for chunk in chunks:
                        self._enrich(chunk)
                    embeddings = self.embedder.embed([c.text for c in chunks])
                self.chroma.delete_document(doc_id)
                self.chroma.upsert(chunks, embeddings, doc_type)
                if self.graph:
                    self.graph.delete_document(doc_id)
                    self.graph.write_document(
                        {"doc_id": doc_id, "filename": path.name, "doc_type": doc_type, "sha256": sha}, chunks
                    )
                report.documents += 1
                report.chunks += len(chunks)
                report.clauses += sum(len(c.clauses) for c in chunks)
                report.max_chunk_tokens = max([report.max_chunk_tokens] + [c.token_count for c in chunks])
                LOG.info(
                    "ingested %s: %d chunks (<=%d tok, overlap %d), %d CoC clauses", path.name, len(chunks),
                    self.cfg.chunk_tokens, self.cfg.chunk_overlap, sum(len(c.clauses) for c in chunks),
                )
        report.audit = audit.summary()
        return report

    def sanity_check(self) -> None:
        """Run a retrieval probe (top_k from config) and print anonymized results only."""
        assert self.embedder and self.chroma
        query = "Which customers may terminate upon a change of control without penalty?"
        with local_processing_guarantee(allow_loopback=False):
            vector = self.embedder.embed([BGE_QUERY_PREFIX + query])
        result = self.chroma.collection.query(
            query_embeddings=vector, n_results=self.cfg.retrieval_top_k, include=["metadatas", "distances"]
        )
        print(f"Top-{self.cfg.retrieval_top_k} retrieval hits (anonymized):")
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            print(f"  d={dist:.3f} {meta['filename']} | CoC={meta['has_change_of_control']} | {meta['anon_text'][:140]!r}")
        if self.graph:
            print(f"Neo4j CHANGE_OF_CONTROL clauses: {self.graph.count_clauses()}")

    def close(self) -> None:
        """Release external handles."""
        if self.graph:
            self.graph.close()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Apex Stage 1 ingestion pipeline.")
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--graph", action="store_true", help="Also write to Neo4j (off by default).")
    pipeline = IngestPipeline(cfg, proxy, use_graph=args.graph or cfg.use_graph)
    parser.add_argument("--verify", action="store_true", help="Run retrieval/graph sanity check.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = ApexConfig()
    if not verify_guard_blocks_egress():
        LOG.error("Local Processing Guarantee self-test FAILED; refusing to run.")
        return 2
    pdfs = sorted((args.pdf_dir or cfg.pdf_dir).glob("*.pdf"))
    if not pdfs:
        LOG.error("No PDFs found. Run: python -m data.synthetic_gen")
        return 1

    proxy = AirGappedProxy(
        cfg.spacy_model,
        known_entities={
            "COMPANY": ["Meridian Helix Technologies, Inc.", "Northgate Logistics Group, Inc.", "Solara Retail Partners LLC", "Halcyon Pharma Services plc"],
            "EXEC": ["Dana R. Whitfield", "Marcus Okonkwo-Reyes", "Priya Venkataraman", "Thomas J. Albrecht", "Elena M. Castellanos", "Rajiv S. Menon"],
        },
    )
    pipeline = IngestPipeline(cfg, proxy, use_graph=args.graph or cfg.use_graph)
    try:
        report = pipeline.run(pdfs)
        print(f"documents={report.documents} chunks={report.chunks} coc_clauses={report.clauses} "
              f"max_chunk_tokens={report.max_chunk_tokens}")
        print(f"network audit: {report.audit}")
        print(f"proxy replacements: {proxy.stats()}")
        if args.verify:
            pipeline.sanity_check()
    except NetworkViolation as exc:
        LOG.error("LOCAL PROCESSING GUARANTEE VIOLATED: %s", exc)
        return 3
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())