"""SQLite persistence for AegisDeal: ``apex_data/aegisdeal.db``.

Five tables: documents, audit_findings, hitl_decisions, dpo_pairs, negotiation_runs.
``audit_findings`` and ``dpo_pairs`` reuse the DDL owned by ``FindingsStore`` / ``DPOStore`` (single source of truth),
so those classes can open the same file and keep working unchanged. Only anonymized content is stored.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.copilot import FindingsStore
from llm.quant_dpo import DPOStore

TABLES = ("documents", "audit_findings", "hitl_decisions", "dpo_pairs", "negotiation_runs")
DDL = (
    "CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, display_name TEXT, doc_type TEXT, kind TEXT, sha256 TEXT, "
    "size_bytes INTEGER, chunks INTEGER, mean_tokens REAL, max_tokens INTEGER, coc_chunks INTEGER, ingested_at TEXT)",
    FindingsStore.SCHEMA,
    "CREATE TABLE IF NOT EXISTS hitl_decisions (decision_id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id TEXT, action TEXT, "
    "reviewer TEXT, note TEXT, before_json TEXT, after_json TEXT, pair_id TEXT, created_at TEXT)",
    DPOStore.SCHEMA,
    "CREATE TABLE IF NOT EXISTS negotiation_runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, base_price_musd REAL, "
    "ebitda_musd REAL, exercise_probability REAL, upfront_musd REAL, earnout_musd REAL, headline_musd REAL, engine TEXT, "
    "statuses_json TEXT, transcript_json TEXT, memo_md TEXT, evidence_json TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_findings_hitl ON audit_findings(hitl_status)",
    "CREATE INDEX IF NOT EXISTS idx_hitl_finding ON hitl_decisions(finding_id)",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Database:
    """Thread-safe SQLite helper (WAL mode, busy timeout) owning the schema and the non-store tables."""

    def __init__(self, path: Path) -> None:
        """Open/create the database and all tables."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=5000")
            for statement in DDL:
                self._db.execute(statement)
            self._db.commit()

    def counts(self) -> Dict[str, int]:
        """Row count per table."""
        with self._lock:
            return {t: int(self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in TABLES}

    def upsert_document(self, *, doc_id: str, display_name: str, doc_type: str, kind: str, sha256: str, size_bytes: int,
                        chunks: int, mean_tokens: float, max_tokens: int, coc_chunks: int) -> None:
        """Insert or refresh a document row (display name is a pseudonym, never the raw filename)."""
        with self._lock:
            self._db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET display_name=excluded.display_name, "
                "doc_type=excluded.doc_type, kind=excluded.kind, size_bytes=excluded.size_bytes, chunks=excluded.chunks, "
                "mean_tokens=excluded.mean_tokens, max_tokens=excluded.max_tokens, coc_chunks=excluded.coc_chunks, ingested_at=excluded.ingested_at",
                (doc_id, display_name, doc_type, kind, sha256, size_bytes, chunks, mean_tokens, max_tokens, coc_chunks, _now()))
            self._db.commit()

    def list_documents(self) -> List[Dict[str, Any]]:
        """All registered documents."""
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM documents ORDER BY kind, display_name").fetchall()]

    def record_hitl(self, *, finding_id: str, action: str, reviewer: str, note: str, before_json: str, after_json: str, pair_id: str) -> int:
        """Append an audit-trail row for a reviewer decision; returns its id."""
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO hitl_decisions (finding_id, action, reviewer, note, before_json, after_json, pair_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (finding_id, action, reviewer, note, before_json, after_json, pair_id, _now()))
            self._db.commit()
            return int(cur.lastrowid)

    def record_negotiation(self, *, base_price_musd: float, ebitda_musd: float, exercise_probability: float, upfront_musd: float,
                           earnout_musd: float, headline_musd: float, engine: str, statuses: Dict[str, Any],
                           transcript: List[Dict[str, Any]], memo_md: str, evidence: Dict[str, Any]) -> int:
        """Persist one negotiation run with its memo and air-gap evidence; returns ``run_id``."""
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO negotiation_runs (created_at, base_price_musd, ebitda_musd, exercise_probability, upfront_musd, earnout_musd, "
                "headline_musd, engine, statuses_json, transcript_json, memo_md, evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), base_price_musd, ebitda_musd, exercise_probability, upfront_musd, earnout_musd, headline_musd, engine,
                 json.dumps(statuses), json.dumps(transcript, default=str), memo_md, json.dumps(evidence, default=str)))
            self._db.commit()
            return int(cur.lastrowid)

    def get_negotiation(self, run_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """One run by id, or the latest when ``run_id`` is None."""
        with self._lock:
            if run_id is None:
                row = self._db.execute("SELECT * FROM negotiation_runs ORDER BY run_id DESC LIMIT 1").fetchone()
            else:
                row = self._db.execute("SELECT * FROM negotiation_runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None