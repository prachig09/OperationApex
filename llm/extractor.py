"""Offline RAG risk extractor (Stage 2).

Hybrid retrieval over the Stage 1 stores (ChromaDB vectors + Neo4j knowledge
graph) feeds a local Mistral-7B-Instruct-v0.3 served by Ollama or vLLM on
loopback. Every network-capable step runs under the Local Processing
Guarantee. When an evaluator is attached, the findings are released only if
the RAG Triad gate passes; otherwise ``LowConfidenceRetractionException`` is
raised and no answer is returned.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlparse

from config import ApexConfig
from pipeline.ingest import BGE_QUERY_PREFIX, ChromaStore, Chunk, LocalEmbedder, Neo4jStore
from security.network_guard import local_processing_guarantee

LOG = logging.getLogger("apex.extractor")

RISK_CATEGORIES = (
    "CHANGE_OF_CONTROL", "TERMINATION_RIGHTS", "CUSTOMER_CONCENTRATION", "LIABILITY_INDEMNITY",
    "ASSIGNMENT_RESTRICTION", "EXCLUSIVITY_NON_COMPETE", "FINANCIAL_COVENANT", "LITIGATION",
    "IP_OWNERSHIP", "REGULATORY", "OTHER",
)
SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_[0-9A-F]{6}\]")

EXTRACTION_PROMPT = (
    "You are a due-diligence extraction engine. You COPY text; you never write, summarize or rephrase it.\n"
    "Input: a QUESTION and numbered EXCERPTS. Each excerpt begins with a header line: "
    "[N] document=... | section=... | pages=...  followed by its TEXT.\n"
    "RULES (any violation makes your answer unusable):\n"
    "1. `evidence_quote` MUST be an exact, character-for-character copy of ONE contiguous span taken from ONE "
    "excerpt's TEXT. Do not paraphrase, reorder, abbreviate, use '...', fix typos or join separate sentences. "
    "Maximum 40 words.\n"
    "2. `source_ids` is a list holding the single number N of the excerpt the quote was copied from, e.g. [2].\n"
    "3. `source_document`, `source_section` and `source_pages` MUST be copied exactly from that excerpt's header line.\n"
    "4. `summary` is ONE short sentence restating only facts that appear inside `evidence_quote`. "
    "Add no names, numbers, causes, opinions or outside knowledge.\n"
    "5. `financial_exposure` holds figures exactly as written in the quote (e.g. \"40%\"), otherwise null.\n"
    "6. Ignore excerpts that do not directly answer the QUESTION. If none do, return {\"risks\": []}. Never guess.\n"
    "7. Output exactly ONE JSON object. No markdown, no commentary.\n"
    "SCHEMA: {\"risks\": [{\"category\": one of " + "|".join(RISK_CATEGORIES) + ", "
    "\"severity\": one of " + "|".join(SEVERITIES) + ", \"title\": string, \"summary\": string, "
    "\"evidence_quote\": string, \"source_ids\": [integer], \"source_document\": string, "
    "\"source_section\": string, \"source_pages\": string, \"financial_exposure\": string|null, "
    "\"recommended_action\": string}]}\n"
    "FORMAT EXAMPLE ONLY (fictional; never reuse its content):\n"
    "[1] document=example.pdf | section=Payment | pages=4\n"
    "TEXT: Late payments accrue interest at 1.5% per month.\n"
    "{\"risks\": [{\"category\": \"FINANCIAL_COVENANT\", \"severity\": \"LOW\", \"title\": \"Late payment interest\", "
    "\"summary\": \"Late payments accrue interest at 1.5% per month.\", "
    "\"evidence_quote\": \"Late payments accrue interest at 1.5% per month.\", \"source_ids\": [1], "
    "\"source_document\": \"example.pdf\", \"source_section\": \"Payment\", \"source_pages\": \"4\", "
    "\"financial_exposure\": \"1.5% per month\", \"recommended_action\": \"Confirm payment terms.\"}]}"
)
SYSTEM_PROMPT = EXTRACTION_PROMPT  # backward-compatible alias


# ------------------------------------------------------------------ data model
@dataclass
class RetrievedContext:
    """One retrieved snippet. Interoperable with Stage 1 ``Chunk`` via ``from_chunk``."""

    ref: int
    chunk_id: str
    filename: str
    section: str
    page_start: int
    page_end: int
    text: str
    similarity: Optional[float] = None
    source: str = "vector"
    anonymized: bool = False
    clause_types: List[str] = field(default_factory=list)

    @classmethod
    def from_chunk(
        cls, chunk: Chunk, ref: int = 0, similarity: Optional[float] = None,
        source: str = "vector", use_anonymized: bool = False,
    ) -> "RetrievedContext":
        """Build a context from a Stage 1 ``Chunk``."""
        anon = bool(use_anonymized and chunk.anon_text)
        return cls(
            ref=ref, chunk_id=chunk.chunk_id, filename=chunk.filename, section=chunk.section,
            page_start=chunk.page_start, page_end=chunk.page_end,
            text=chunk.anon_text if anon else chunk.text, similarity=similarity, source=source,
            anonymized=anon, clause_types=[c["type"] for c in chunk.clauses],
        )


@dataclass
class RiskFinding:
    """One extracted risk clause (metadata fields are set from the retrieved excerpt, not trusted from the model)."""

    category: str
    severity: str
    title: str
    summary: str
    evidence_quote: str
    source_ids: List[int]
    financial_exposure: Optional[str] = None
    recommended_action: str = ""
    source_document: str = ""
    source_section: str = ""
    source_pages: str = ""


@dataclass
class ExtractionResult:
    """Findings for one query. Only ever returned after the triad gate passes."""

    query: str
    findings: List[RiskFinding]
    contexts: List[RetrievedContext]
    model: str
    latency_s: float
    triad: Optional[Any] = None

    def to_dict(self, include_context_text: bool = False) -> Dict[str, Any]:
        """JSON-serializable view; context text is omitted unless requested."""
        contexts = []
        for c in self.contexts:
            row = asdict(c)
            if not include_context_text:
                row.pop("text")
            contexts.append(row)
        return {
            "query": self.query, "model": self.model, "latency_s": round(self.latency_s, 2),
            "findings": [asdict(f) for f in self.findings], "contexts": contexts,
            "triad": self.triad.to_dict() if hasattr(self.triad, "to_dict") else None,
        }


class AnswerGate(Protocol):
    """Evaluator contract; ``observability.triad_eval.TriadEvaluator`` implements it."""

    def gate(self, query: str, findings: Sequence[RiskFinding], contexts: Sequence[RetrievedContext]) -> Any:
        """Return a report or raise to block the answer."""


class ExtractionParseError(RuntimeError):
    """The model did not return parseable JSON."""


class LLMUnavailableError(RuntimeError):
    """The local model server is unreachable or returned an error."""


# ------------------------------------------------------------------ LLM client
LOW_VRAM_MODELS = ("qwen2.5:3b-instruct", "mistral:7b-instruct-q4_K_M")  # tested-target Ollama tags


@dataclass(frozen=True)
class LLMConfig:
    """Local model-server settings (Ollama or vLLM). Defaults suit low-VRAM models and small contexts.

    Set ``APEX_LLM_MODEL`` to one of ``LOW_VRAM_MODELS`` (or any local tag). 4096 tokens covers the
    system prompt, <=5 excerpts of ~250 tokens and the JSON reply.
    """

    backend: str = os.getenv("APEX_LLM_BACKEND", "ollama")
    base_url: str = os.getenv("APEX_LLM_URL", "")
    model: str = os.getenv("APEX_LLM_MODEL", "")
    temperature: float = 0.0
    num_ctx: int = 2048
    max_tokens: int = 400
    timeout_s: float = 90.0
    seed: int = 7

    @property
    def url(self) -> str:
        """Resolved base URL (loopback defaults)."""
        if self.base_url:
            return self.base_url.rstrip("/")
        return "http://127.0.0.1:8000" if self.backend == "vllm" else "http://127.0.0.1:11434"

    @property
    def model_name(self) -> str:
        """Resolved model identifier (Ollama default: qwen2.5:3b-instruct)."""
        if self.model:
            return self.model
        return "mistralai/Mistral-7B-Instruct-v0.3" if self.backend == "vllm" else LOW_VRAM_MODELS[0]


class LocalLLMClient:
    """Minimal JSON-mode chat client for a local Ollama/vLLM server (stdlib only)."""

    def __init__(self, cfg: LLMConfig, allowed_hosts: Sequence[str] = ()) -> None:
        """Bind config; system proxies are disabled and the guard restricts hosts."""
        if cfg.backend not in ("ollama", "vllm"):
            raise ValueError("backend must be 'ollama' or 'vllm'")
        self.cfg = cfg
        self.allowed_hosts = tuple(allowed_hosts)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.cfg.url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with local_processing_guarantee(self.allowed_hosts, allow_loopback=True):
            try:
                with self._opener.open(req, timeout=self.cfg.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "ignore")[:300]
                raise LLMUnavailableError(f"{self.cfg.backend} HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                raise LLMUnavailableError(f"cannot reach {self.cfg.url}: {exc}") from exc

    def ping(self) -> bool:
        """True if the server answers and lists models."""
        try:
            self._request("GET", "/api/tags" if self.cfg.backend == "ollama" else "/v1/models")
            return True
        except LLMUnavailableError:
            return False

    def chat_json(
        self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> str:
        """Return the model's raw JSON-mode reply text."""
        temp = self.cfg.temperature if temperature is None else temperature
        limit = max_tokens or self.cfg.max_tokens
        if self.cfg.backend == "ollama":
            data = self._request("POST", "/api/chat", {
                "model": self.cfg.model_name, "stream": False, "format": "json",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temp, "num_ctx": self.cfg.num_ctx, "num_predict": limit, "seed": self.cfg.seed},
            })
            return str(data["message"]["content"])
        data = self._request("POST", "/v1/chat/completions", {
            "model": self.cfg.model_name, "temperature": temp, "max_tokens": limit, "seed": self.cfg.seed,
            "messages": [{"role": "user", "content": f"{system}\n\n{user}"}],  # v0.3 template: no system role
            "response_format": {"type": "json_object"},
        })
        return str(data["choices"][0]["message"]["content"])


# ------------------------------------------------------------------- retrieval
class HybridRetriever:
    """Vector search on ChromaDB, expanded through the Neo4j knowledge graph."""

    def __init__(
        self, embedder: LocalEmbedder, chroma: ChromaStore, graph: Optional[Neo4jStore] = None,
        use_anonymized: bool = False, allowed_hosts: Sequence[str] = (),
    ) -> None:
        """Bind the Stage 1 stores. ``use_anonymized`` feeds the LLM ``anon_text`` instead of raw text."""
        self.embedder, self.chroma, self.graph = embedder, chroma, graph
        self.use_anonymized = use_anonymized
        self.allowed_hosts = tuple(allowed_hosts)
        self._aliases: Optional[Dict[str, str]] = None

    def _read(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        if self.graph is None:
            return []
        with local_processing_guarantee(self.allowed_hosts):
            with self.graph.driver.session() as session:
                return session.execute_read(lambda tx: [r.data() for r in tx.run(query, **params)])

    def _text(self, raw: Optional[str], anon: Optional[str]) -> str:
        return (anon if self.use_anonymized and anon else raw) or ""

    def _vector(self, query: str, k: int, where: Optional[Dict[str, Any]]) -> List[RetrievedContext]:
        with local_processing_guarantee(self.allowed_hosts):
            total = self.chroma.collection.count()
        if total == 0:
            return []
        with local_processing_guarantee(allow_loopback=False):
            vector = self.embedder.embed([BGE_QUERY_PREFIX + query])
        kwargs: Dict[str, Any] = dict(
            query_embeddings=vector, n_results=min(k, total), include=["documents", "metadatas", "distances"]
        )
        if where:
            kwargs["where"] = where
        with local_processing_guarantee(self.allowed_hosts):
            res = self.chroma.collection.query(**kwargs)
        out: List[RetrievedContext] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(RetrievedContext(
                ref=0, chunk_id=cid, filename=meta.get("filename", ""), section=meta.get("section", ""),
                page_start=int(meta.get("page_start", 0)), page_end=int(meta.get("page_end", 0)),
                text=self._text(doc, meta.get("anon_text")), similarity=1.0 - float(dist), source="vector",
                anonymized=self.use_anonymized,
                clause_types=["CHANGE_OF_CONTROL"] if meta.get("has_change_of_control") else [],
            ))
        return out

    def _graph_rows(self, rows: List[Dict[str, Any]], source: str) -> List[RetrievedContext]:
        return [RetrievedContext(
            ref=0, chunk_id=r["chunk_id"], filename=r.get("filename") or "", section=r.get("section") or "",
            page_start=int(r.get("page_start") or 0), page_end=int(r.get("page_end") or 0),
            text=self._text(r.get("text"), r.get("anon_text")), source=source, anonymized=self.use_anonymized,
            clause_types=list(r.get("clause_types") or []),
        ) for r in rows]

    def _graph_clauses(self, clause_types: Sequence[str], limit: int) -> List[RetrievedContext]:
        rows = self._read(
            """
            MATCH (d:Document)-[:HAS_CHUNK]->(k:Chunk)-[:CONTAINS_CLAUSE]->(c:Clause)
            WHERE c.type IN $types
            RETURN k.chunk_id AS chunk_id, d.filename AS filename, k.section AS section, k.idx AS idx,
                   k.page_start AS page_start, k.page_end AS page_end, k.text AS text, k.anon_text AS anon_text,
                   collect(DISTINCT c.type) AS clause_types
            ORDER BY filename, idx LIMIT $limit
            """,
            types=list(clause_types), limit=limit,
        )
        return self._graph_rows(rows, "graph_clause")

    def _graph_neighbors(self, chunk_ids: Sequence[str]) -> List[RetrievedContext]:
        rows = self._read(
            """
            UNWIND $ids AS cid
            MATCH (k:Chunk {chunk_id: cid})
            OPTIONAL MATCH (p:Chunk)-[:NEXT]->(k)
            OPTIONAL MATCH (k)-[:NEXT]->(n:Chunk)
            WITH [x IN [p, n] WHERE x IS NOT NULL] AS nbrs
            UNWIND nbrs AS x
            MATCH (d:Document)-[:HAS_CHUNK]->(x)
            RETURN DISTINCT x.chunk_id AS chunk_id, d.filename AS filename, x.section AS section,
                   x.page_start AS page_start, x.page_end AS page_end, x.text AS text, x.anon_text AS anon_text
            """,
            ids=list(chunk_ids),
        )
        return self._graph_rows(rows, "graph_neighbor")

    def retrieve(
        self, query: str, k: int = ApexConfig.retrieval_top_k, where: Optional[Dict[str, Any]] = None,
        clause_types: Sequence[str] = (), expand_neighbors: int = 0,
        max_contexts: int = ApexConfig.max_context_chunks, clause_limit: int = 2,
    ) -> List[RetrievedContext]:
        """Retrieve top-k vector hits plus a few graph-tagged clause chunks, deduplicated and numbered.

        Defaults keep the prompt small for low-VRAM models: top_k=3, <=2 graph clause chunks,
        no neighbour expansion, <=5 contexts total. Raise ``expand_neighbors`` to re-enable it.
        """
        merged: Dict[str, RetrievedContext] = {}
        vector_hits = self._vector(query, k, where)
        for ctx in vector_hits:
            merged.setdefault(ctx.chunk_id, ctx)
        if self.graph is not None:
            if clause_types:
                for ctx in self._graph_clauses(clause_types, limit=clause_limit):
                    merged.setdefault(ctx.chunk_id, ctx)
            seeds = [c.chunk_id for c in vector_hits[:expand_neighbors]]
            if seeds:
                for ctx in self._graph_neighbors(seeds):
                    merged.setdefault(ctx.chunk_id, ctx)
        contexts = list(merged.values())[:max_contexts]
        for number, ctx in enumerate(contexts, start=1):
            ctx.ref = number
        return contexts

    def alias_map(self) -> Dict[str, str]:
        """Placeholder -> original name map stored on Stage 1 ``Entity`` nodes."""
        if self._aliases is None:
            rows = self._read("MATCH (e:Entity) RETURN e.alias AS alias, e.name AS name")
            self._aliases = {r["alias"]: r["name"] for r in rows if r.get("alias") and r.get("name")}
        return self._aliases

    def _chroma_clauses(self, clause_types: Sequence[str], limit: int) -> List[RetrievedContext]:
        """Clause-tagged chunks via ChromaDB metadata (zero-setup replacement for the Neo4j clause lookup)."""
        if "CHANGE_OF_CONTROL" not in clause_types:
            return []
        with local_processing_guarantee(self.allowed_hosts):
            res = self.chroma.collection.get(where={"has_change_of_control": True}, limit=limit,
                                             include=["documents", "metadatas"])
        return [RetrievedContext(
            ref=0, chunk_id=cid, filename=meta.get("filename", ""), section=meta.get("section", ""),
            page_start=int(meta.get("page_start", 0)), page_end=int(meta.get("page_end", 0)),
            text=self._text(doc, meta.get("anon_text")), source="chroma_clause", anonymized=self.use_anonymized,
            clause_types=["CHANGE_OF_CONTROL"],
        ) for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

    def retrieve(
        self, query: str, k: int = ApexConfig.retrieval_top_k, where: Optional[Dict[str, Any]] = None,
        clause_types: Sequence[str] = (), expand_neighbors: int = 0,
        max_contexts: int = ApexConfig.max_context_chunks, clause_limit: int = 2,
    ) -> List[RetrievedContext]:
        """Top-k vector hits plus clause-tagged chunks (Neo4j if attached, else ChromaDB metadata), deduplicated and numbered."""
        merged: Dict[str, RetrievedContext] = {}
        vector_hits = self._vector(query, k, where)
        for ctx in vector_hits:
            merged.setdefault(ctx.chunk_id, ctx)
        if clause_types:
            tagged = (self._graph_clauses(clause_types, limit=clause_limit) if self.graph is not None
                      else self._chroma_clauses(clause_types, clause_limit))
            for ctx in tagged:
                merged.setdefault(ctx.chunk_id, ctx)
        if self.graph is not None and expand_neighbors:
            for ctx in self._graph_neighbors([c.chunk_id for c in vector_hits[:expand_neighbors]]):
                merged.setdefault(ctx.chunk_id, ctx)
        contexts = list(merged.values())[:max_contexts]
        for number, ctx in enumerate(contexts, start=1):
            ctx.ref = number
        return contexts


# ------------------------------------------------------------------ extraction
def build_user_prompt(query: str, contexts: Sequence[RetrievedContext]) -> str:
    """Render the numbered excerpt prompt (header line + TEXT) that the extraction rules refer to."""
    blocks = []
    for c in contexts:
        pages = str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
        blocks.append(f"[{c.ref}] document={c.filename} | section={c.section or 'n/a'} | pages={pages}\nTEXT: {c.text}")
    return f"QUESTION:\n{query}\n\nEXCERPTS:\n" + "\n\n".join(blocks) + "\n\nReturn the JSON object now."


def _loads_lenient(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ExtractionParseError("model output was not valid JSON")


def parse_findings(raw: str) -> List[RiskFinding]:
    """Parse and normalize the model's JSON reply into ``RiskFinding`` objects."""
    data = _loads_lenient(raw)
    if isinstance(data, dict):
        items = next((data[k] for k in ("risks", "findings", "items", "clauses", "results") if isinstance(data.get(k), list)), None)
        if items is None and any(k in data for k in ("title", "summary", "evidence_quote", "category")):
            items = [data]
    else:
        items = data
    if not isinstance(items, list):
        raise ExtractionParseError("JSON did not contain a 'risks' list")
    findings: List[RiskFinding] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_ids = item.get("source_ids", [])
        raw_ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
        ids = [int(n) for value in raw_ids for n in re.findall(r"\d+", str(value))]
        category = str(item.get("category", "OTHER")).strip().upper().replace(" ", "_")
        severity = str(item.get("severity", "MEDIUM")).strip().upper()
        exposure = item.get("financial_exposure")
        finding = RiskFinding(
            category=category if category in RISK_CATEGORIES else "OTHER",
            severity=severity if severity in SEVERITIES else "MEDIUM",
            title=str(item.get("title", "")).strip(), summary=str(item.get("summary", "")).strip(),
            evidence_quote=str(item.get("evidence_quote", "")).strip(), source_ids=ids,
            financial_exposure=str(exposure).strip() if exposure not in (None, "", "null") else None,
            recommended_action=str(item.get("recommended_action", "")).strip(),
            source_document=str(item.get("source_document", "")).strip(),
            source_section=str(item.get("source_section", "")).strip(),
            source_pages=str(item.get("source_pages", "")).strip(),
        )
        key = (finding.category, re.sub(r"\W+", " ", finding.evidence_quote.casefold()).strip())
        if finding.summary and key not in seen:
            seen.add(key)
            findings.append(finding)
    return findings


def attach_metadata(findings: Sequence[RiskFinding], contexts: Sequence[RetrievedContext]) -> List[RiskFinding]:
    """Overwrite model-reported document/section/pages with the retrieved excerpt's real metadata."""
    by_ref = {c.ref: c for c in contexts}
    for finding in findings:
        refs = [r for r in finding.source_ids if r in by_ref]
        if not refs:
            continue
        ctx = by_ref[refs[0]]
        finding.source_document, finding.source_section = ctx.filename, ctx.section
        finding.source_pages = str(ctx.page_start) if ctx.page_start == ctx.page_end else f"{ctx.page_start}-{ctx.page_end}"
    return list(findings)


@dataclass(frozen=True)
class AuditQuery:
    """A standard due-diligence question; ``clause_types`` triggers graph clause lookups."""

    name: str
    question: str
    clause_types: Tuple[str, ...] = ()


AUDIT_QUERIES: Tuple[AuditQuery, ...] = (
    AuditQuery("change_of_control",
               "Which customers or counterparties may terminate, cancel or renegotiate upon a change of control or "
               "acquisition of the company, and what share of revenue do they represent?", ("CHANGE_OF_CONTROL",)),
    AuditQuery("customer_concentration",
               "How concentrated is revenue among the largest customers, and what are the stated percentages and amounts?"),
    AuditQuery("termination_rights",
               "Which agreements grant termination rights without penalty, and under what conditions can they be exercised?"),
    AuditQuery("liability_indemnity",
               "What are the liability caps, indemnification obligations and insurance requirements in the commercial agreements?"),
    AuditQuery("financial_commitments",
               "Identify debt facilities, covenants, minimum purchase commitments and other material financial obligations."),
)


class LocalRiskExtractor:
    """Retrieval -> local LLM -> triad gate -> findings."""

    def __init__(
        self, retriever: HybridRetriever, llm: LocalLLMClient, evaluator: Optional[AnswerGate] = None,
        k: int = ApexConfig.retrieval_top_k, max_contexts: int = ApexConfig.max_context_chunks,
    ) -> None:
        """Compose the pipeline. Without an evaluator, answers are returned ungated (not recommended)."""
        self.retriever, self.llm, self.evaluator = retriever, llm, evaluator
        self.k, self.max_contexts = k, max_contexts
    @classmethod
    def build(
        cls, cfg: Optional[ApexConfig] = None, llm_cfg: Optional[LLMConfig] = None, use_graph: bool = False,
        use_anonymized: bool = False, evaluate: bool = True, groundedness_threshold: float = 0.75,
    ) -> "LocalRiskExtractor":
        """Wire Stage 1 stores, the local LLM and the triad evaluator together."""
        cfg, llm_cfg = cfg or ApexConfig(), llm_cfg or LLMConfig()
        with local_processing_guarantee(cfg.allowed_hosts):
            with local_processing_guarantee(allow_loopback=False):
                embedder = LocalEmbedder(cfg.embedding_model_path, cfg.embed_batch_size)
            chroma = ChromaStore(cfg.chroma_dir, cfg.chroma_collection)
            graph = Neo4jStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password) if use_graph else None
        llm = LocalLLMClient(llm_cfg, cfg.allowed_hosts)
        if not llm.ping():
            raise LLMUnavailableError(f"No local model server at {llm_cfg.url} ({llm_cfg.backend}). Start it first.")
        evaluator = None
        if evaluate:
            from observability.triad_eval import TriadConfig, TriadEvaluator  # lazy: avoids an import cycle

            evaluator = TriadEvaluator(embedder, llm, TriadConfig(
                groundedness_threshold=groundedness_threshold,
                use_llm_judge=False,
                metrics_path=cfg.data_dir / "observability" / "triad_metrics.jsonl",
            ))
        retriever = HybridRetriever(embedder, chroma, graph, use_anonymized, cfg.allowed_hosts)
        return cls(retriever, llm, evaluator)

    def extract(
        self, query: str, clause_types: Sequence[str] = (), where: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Answer one question, keeping only claims that survive the triad gate.

        Raises ``LowConfidenceRetractionException`` only when every claim fails verification.
        """
        started = time.perf_counter()
        contexts = self.retriever.retrieve(
            query, k=self.k, where=where, clause_types=clause_types, max_contexts=self.max_contexts
        )
        model = self.llm.cfg.model_name
        if not contexts:
            return ExtractionResult(query, [], [], model, time.perf_counter() - started)
        prompt = build_user_prompt(query, contexts)
        try:
            findings = parse_findings(self.llm.chat_json(EXTRACTION_PROMPT, prompt))
        except ExtractionParseError:
            LOG.warning("unparseable model output; retrying once")
            try:
                findings = parse_findings(self.llm.chat_json(
                    EXTRACTION_PROMPT, prompt + "\nYour previous reply was not valid JSON. Output ONE valid JSON object only."
                ))
            except ExtractionParseError as exc:
                LOG.warning("skipping unparseable query after retry: %s", exc)
                findings = []
        report = self.evaluator.gate(query, findings, contexts) if self.evaluator else None
        keep = set(getattr(report, "retained", range(len(findings))))
        resolved = {c.index: c.cited_refs for c in getattr(report, "claims", [])}
        kept: List[RiskFinding] = []
        for i, finding in enumerate(findings):
            if i not in keep:
                continue
            if resolved.get(i):
                finding.source_ids = list(resolved[i])  # citation as verified by the evaluator
            kept.append(finding)
        if len(kept) < len(findings):
            LOG.warning("triad gate dropped %d of %d claims for query: %s", len(findings) - len(kept), len(findings), query[:80])
        attach_metadata(kept, contexts)
        return ExtractionResult(query, kept, contexts, model, time.perf_counter() - started, report)

    def audit(self, queries: Sequence[AuditQuery] = AUDIT_QUERIES) -> List[ExtractionResult]:
        """Run the standard question battery. Any retraction blocks the whole audit output."""
        return [self.extract(q.question, q.clause_types) for q in queries]

    def restore(self, text: str) -> str:
        """Re-identify placeholders using the alias map persisted in Neo4j (local only)."""
        aliases = self.retriever.alias_map()
        return _PLACEHOLDER_RE.sub(lambda m: aliases.get(m.group(0), m.group(0)), text)


# ------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m llm.extractor --audit``."""
    parser = argparse.ArgumentParser(description="Apex Stage 2 offline risk extractor.")
    parser.add_argument("--query", help="Ask a single question.")
    parser.add_argument("--audit", action="store_true", help="Run the standard audit battery.")
    parser.add_argument("--backend", choices=["ollama", "vllm"])
    parser.add_argument("--model")
    parser.add_argument("--url")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--anonymized", action="store_true", help="Feed the LLM anonymized text.")
    parser.add_argument("--no-eval", action="store_true", help="Disable the triad gate (not recommended).")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--restore", action="store_true", help="Re-identify placeholders in the printed output.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not args.query and not args.audit:
        parser.error("provide --query or --audit")

    from observability.triad_eval import LowConfidenceRetractionException

    overrides = {k: v for k, v in (("backend", args.backend), ("model", args.model), ("base_url", args.url)) if v}
    extractor = LocalRiskExtractor.build(
        llm_cfg=dataclasses.replace(LLMConfig(), **overrides), use_graph=not args.no_graph,
        use_anonymized=args.anonymized, evaluate=not args.no_eval, groundedness_threshold=args.threshold,
    )
    try:
        results = extractor.audit() if args.audit else [extractor.extract(args.query, ("CHANGE_OF_CONTROL",) if args.query and "change of control" in args.query.lower() else ())]
    except LowConfidenceRetractionException as exc:
        print(json.dumps({"blocked": True, "reason": str(exc), "triad": exc.report.to_dict()}, indent=2))
        return 5
    payload = json.dumps([r.to_dict() for r in results], indent=2)
    print(extractor.restore(payload) if args.restore else payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())