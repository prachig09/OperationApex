"""Central configuration for Operation Apex, Stage 1."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ApexConfig:
    """Immutable runtime configuration.

    Attributes:
        data_dir: Root directory for PDFs and the ChromaDB store.
        embedding_model_path: Local directory holding bge-large-en-v1.5.
        spacy_model: Locally installed spaCy pipeline used for NER.
        chunk_tokens: Maximum tokens per chunk (header included).
        chunk_overlap: Token overlap between consecutive chunks.
        embed_batch_size: Batch size for local embedding.
        chroma_collection: ChromaDB collection name.
        neo4j_uri / neo4j_user / neo4j_password: Self-hosted Neo4j credentials.
        allowed_hosts: Extra non-loopback hosts (e.g. a LAN Neo4j box) the
            guard may reach. Loopback is always permitted for the DB stage.
    """

    data_dir: Path = Path(os.getenv("APEX_DATA_DIR", "./apex_data"))
    embedding_model_path: str = os.getenv("APEX_EMBED_MODEL", "./models/bge-large-en-v1.5")
    spacy_model: str = os.getenv("APEX_SPACY_MODEL", "en_core_web_lg")
    chunk_tokens: int = 250
    chunk_overlap: int = 30
    retrieval_top_k: int = 3
    max_context_chunks: int = 2
    embed_batch_size: int = 16
    chroma_collection: str = "apex_target_docs"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "apex-local-only")
    allowed_hosts: Tuple[str, ...] = ()

    use_graph: bool = os.getenv("APEX_USE_GRAPH", "0") == "1"   # Neo4j is opt-in

    @property
    def pdf_dir(self) -> Path:
        """Directory containing the target-company PDFs."""
        return self.data_dir / "pdfs"

    @property
    def chroma_dir(self) -> Path:
        """Persistent ChromaDB directory."""
        return self.data_dir / "chroma"