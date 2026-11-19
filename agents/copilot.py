"""Agentic Due Diligence Copilot.

A tool-using chat agent with four deterministic tools (SQLite + ChromaDB + CSV backed). An optional local
LLM only ROUTES a question to tools; answers are composed from tool output, so they cannot be hallucinated.
Every response is passed through ``AirGappedProxy.scrub()`` and leak-checked, and cites its source tool(s).

Deal-math dollar amounts (from ``DealTerms``, not target data) are wrapped in sentinels, shielded during
scrubbing and restored afterwards. Everything else, including target financials, is scrubbed.
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import security.network_guard as ng
from agents.core import AgentError, DealTerms
from agents.nodes import revenue_exposure_pct
from llm.extractor import ExtractionResult, build_user_prompt
from pipeline.ingest import BGE_QUERY_PREFIX, infer_doc_type
from security.network_guard import NetworkViolation, local_processing_guarantee, verify_guard_blocks_egress
from security.proxy import AirGappedProxy, LeakageError

LOG = logging.getLogger("apex.copilot")
_SEV = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_SENTINEL = re.compile(r"«([^«»]{1,24})»")
_STOP = {"the", "and", "for", "are", "what", "which", "who", "how", "does", "did", "with", "that", "this", "about",
         "tell", "show", "from", "have", "has", "any", "all", "can", "you", "our", "their", "there", "into"}
_SYNONYMS = {"control": "change_of_control", "acquisition": "change_of_control", "acquire": "change_of_control",
             "takeover": "change_of_control", "terminate": "termination", "cancel": "termination", "exit": "termination",
             "concentration": "customer_concentration", "liability": "liability_indemnity",
             "indemnity": "liability_indemnity", "indemnification": "liability_indemnity",
             "covenant": "financial_covenant", "debt": "financial_covenant", "facility": "financial_covenant"}

TOOL_DESCRIPTIONS = {
    "query_financial_csv": "Reads demo_financials.csv: revenue, income, balance sheet, customer concentration.",
    "search_audit_findings": "Retrieves RAG-Triad-verified risk clauses with scores and citations (SQLite + ChromaDB).",
    "inspect_airgap_logs": "PII proxy redaction counts, blocked/allowed network operations, egress self-test.",
    "explain_valuation_math": "Breaks down base price, sector markdown, churn penalty, upfront cash and earn-out.",
}
TITLES = {"query_financial_csv": "Financial metrics", "search_audit_findings": "Verified audit findings",
          "inspect_airgap_logs": "Air-gap and PII evidence", "explain_valuation_math": "Valuation bridge"}
INTENTS = {
    "query_financial_csv": r"revenue|ebitda|margin|net income|profit|cash\b|financial|csv|growth|yoy|r&d|balance|income|concentration|customer revenue|top (?:3|three) customers|\barr\b|churn|retention|quarter|nrr|northgate",    "search_audit_findings": r"clause|change of control|\bcoc\b|finding|risk|triad|ground|citation|verified|terminat|indemn|liabil|covenant|contract|evidence|quote|hitl|override",
    "inspect_airgap_logs": r"air.?gap|\bpii\b|redact|scrub|egress|network|blocked|leak|privacy|proxy|socket|anonymi|secur|confidential",
    "explain_valuation_math": r"valuation|price|\bbid\b|earn.?out|negotiat|multiple|penalt|discount|churn|34\.5|offer|math|bridge|haircut",
}


# ================================================================== egress monitor
class EgressMonitor:
    """Counts guarded network operations (allowed loopback vs blocked) by pipeline stage."""

    def __init__(self) -> None:
        """Create an empty monitor."""
        self.events: deque = deque(maxlen=2000)
        self.totals: Counter = Counter()
        self.stage_name = "idle"
        self.selftest: Optional[bool] = None
        self._lock = threading.Lock()

    def install(self) -> None:
        """Wrap the guard's checker once so every guarded operation is recorded."""
        if getattr(ng, "_copilot_patched", False):
            return
        original, monitor = ng._check, self  # noqa: SLF001

        def recording_check(kind: str, host: Any) -> None:
            active = bool(ng._STACK)  # noqa: SLF001
            try:
                original(kind, host)
            except NetworkViolation:
                monitor._add(kind, host, "BLOCKED")
                raise
            if active:
                monitor._add(kind, host, "ALLOWED")

        ng._check = recording_check  # noqa: SLF001
        ng._copilot_patched = True  # noqa: SLF001

    def _add(self, kind: str, host: Any, verdict: str) -> None:
        with self._lock:
            self.totals[(self.stage_name, verdict)] += 1
            self.events.append({"stage": self.stage_name, "op": kind, "target": str(host), "verdict": verdict})

    @contextlib.contextmanager
    def stage(self, name: str):
        """Tag operations with the stage that produced them."""
        previous, self.stage_name = self.stage_name, name
        try:
            yield
        finally:
            self.stage_name = previous

    def counts(self) -> Tuple[int, int]:
        """``(allowed, blocked)`` excluding the intentional self-test probes."""
        with self._lock:
            allowed = sum(v for (s, verdict), v in self.totals.items() if verdict == "ALLOWED" and s != "selftest")
            blocked = sum(v for (s, verdict), v in self.totals.items() if verdict == "BLOCKED" and s != "selftest")
        return allowed, blocked

    def recent_blocked(self, n: int = 5) -> List[Dict[str, str]]:
        """Most recent blocked operations (excluding self-test probes)."""
        with self._lock:
            return [e for e in reversed(self.events) if e["verdict"] == "BLOCKED" and e["stage"] != "selftest"][:n]

    def run_selftest(self) -> bool:
        """Prove the guard blocks external IP and DNS egress (cached)."""
        with self.stage("selftest"):
            self.selftest = bool(verify_guard_blocks_egress())
        return self.selftest


# ================================================================= findings store
class FindingsStore:
    """SQLite table of anonymized, triad-verified findings (the Copilot's and the HITL endpoint's source of truth)."""

    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS audit_findings (finding_id TEXT PRIMARY KEY, created_at TEXT, query_name TEXT, category TEXT, "
        "severity TEXT, title TEXT, summary TEXT, evidence_quote TEXT, financial_exposure TEXT, chunk_id TEXT, doc_type TEXT, "
        "section TEXT, pages TEXT, status TEXT, context_relevance REAL, groundedness REAL, answer_relevance REAL, claim_score REAL, "
        "prompt TEXT, model_output TEXT, hitl_status TEXT DEFAULT 'pending', human_output TEXT, resolved_at TEXT)")

    def __init__(self, db_path: Path) -> None:
        """Open/create the table."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(self.SCHEMA)
            self._db.commit()

    def persist(self, named_results: Sequence[Tuple[str, ExtractionResult]], proxy: AirGappedProxy) -> int:
        """Store retained findings, anonymized (text, prompts and filenames), preserving earlier HITL decisions."""
        stored = 0
        for qname, res in named_results:
            if not res.findings:
                continue
            anon_ctx = {c.ref: dataclasses.replace(
                c, text=proxy.anonymize(c.text), section=proxy.anonymize(c.section or ""),
                filename=f"{infer_doc_type(c.filename)}_{c.chunk_id}.pdf") for c in res.contexts}
            prompt = build_user_prompt(proxy.anonymize(res.query), list(anon_ctx.values()))
            t = res.triad
            retained = list(getattr(t, "retained", []) or [])
            claims = {c.index: c for c in getattr(t, "claims", [])}
            status = "PARTIAL" if getattr(t, "partial", False) else "PASS"
            for k, f in enumerate(res.findings):
                ref = f.source_ids[0] if f.source_ids else None
                ctx = anon_ctx.get(ref)
                pages = "" if ctx is None else (str(ctx.page_start) if ctx.page_start == ctx.page_end else f"{ctx.page_start}-{ctx.page_end}")
                item = {"category": f.category, "severity": f.severity, "title": proxy.anonymize(f.title),
                        "summary": proxy.anonymize(f.summary), "evidence_quote": proxy.anonymize(f.evidence_quote),
                        "source_ids": [ref] if ref else [], "source_document": ctx.filename if ctx else "",
                        "source_section": ctx.section if ctx else "", "source_pages": pages,
                        "financial_exposure": proxy.anonymize(f.financial_exposure) if f.financial_exposure else None,
                        "recommended_action": proxy.anonymize(f.recommended_action)}
                fid = hashlib.sha256(f"{qname}|{f.category}|{item['evidence_quote']}".encode()).hexdigest()[:12]
                claim = claims.get(retained[k] if k < len(retained) else k)
                row = (fid, time.strftime("%Y-%m-%dT%H:%M:%S%z"), qname, f.category, f.severity, item["title"], item["summary"],
                       item["evidence_quote"], item["financial_exposure"], ctx.chunk_id if ctx else "",
                       infer_doc_type(ctx.filename) if ctx else "", item["source_section"], pages, status,
                       getattr(t, "context_relevance", None), getattr(t, "groundedness", None), getattr(t, "answer_relevance", None),
                       getattr(claim, "score", None), prompt, json.dumps({"risks": [item]}, ensure_ascii=False))
                with self._lock:
                    self._db.execute(
                        "INSERT INTO audit_findings (finding_id, created_at, query_name, category, severity, title, summary, evidence_quote, "
                        "financial_exposure, chunk_id, doc_type, section, pages, status, context_relevance, groundedness, answer_relevance, "
                        "claim_score, prompt, model_output) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(finding_id) DO UPDATE SET status=excluded.status, context_relevance=excluded.context_relevance, "
                        "groundedness=excluded.groundedness, answer_relevance=excluded.answer_relevance, claim_score=excluded.claim_score, "
                        "prompt=excluded.prompt, model_output=excluded.model_output", row)
                    self._db.commit()
                stored += 1
        return stored

    def get(self, finding_id: str) -> Optional[Dict[str, Any]]:
        """One finding by id."""
        with self._lock:
            row = self._db.execute("SELECT * FROM audit_findings WHERE finding_id=?", (finding_id,)).fetchone()
        return dict(row) if row else None

    def list(self, hitl_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """All findings (optionally by HITL status), most severe first."""
        with self._lock:
            sql, args = "SELECT * FROM audit_findings", ()
            if hitl_status:
                sql, args = sql + " WHERE hitl_status=?", (hitl_status,)
            rows = [dict(r) for r in self._db.execute(sql, args).fetchall()]
        return sorted(rows, key=lambda r: (_SEV.get(r["severity"], 9), -(r["groundedness"] or 0)))

    def mark_resolved(self, finding_id: str, hitl_status: str, human_output: str = "") -> None:
        """Record the reviewer decision."""
        with self._lock:
            self._db.execute("UPDATE audit_findings SET hitl_status=?, human_output=?, resolved_at=? WHERE finding_id=?",
                             (hitl_status, human_output, time.strftime("%Y-%m-%dT%H:%M:%S%z"), finding_id))
            self._db.commit()

    def count(self) -> int:
        """Number of stored findings."""
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM audit_findings").fetchone()[0])


def ensure_demo_financials(path: Path) -> Path:
    """Return a valid ``demo_financials.csv``; (re)generate it from the synthetic generator if missing or in the old schema."""
    from data.synthetic_gen import validate_financials_csv, write_financials_csv

    path = Path(path)
    if path.exists():
        try:
            validate_financials_csv(path)
            return path
        except ValueError:
            path.rename(path.with_suffix(".csv.old"))
    return write_financials_csv(path)


# ====================================================================== the agent
@dataclass
class ToolResult:
    """Output of one tool call."""

    tool: str
    ok: bool
    text: str
    citations: List[str] = field(default_factory=list)


def _m(value: float, signed: bool = False) -> str:
    """Mark a mandate-derived dollar amount so it survives scrubbing (restored after the proxy runs)."""
    return f"«{value:+,.1f}M»" if signed else f"«{value:,.1f}M»"


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class AgenticCopilot:
    """Conversational due-diligence agent with four deterministic tools and a scrub-and-cite response gate."""

    def __init__(self, proxy: AirGappedProxy, findings: FindingsStore, financial_csv: Path, monitor: EgressMonitor,
                 store_provider: Optional[Callable[[], Optional[Tuple[Any, Any]]]] = None, planner_llm: Any = None,
                 terms: Optional[DealTerms] = None, sector_move_pct: float = -18.0) -> None:
        """Bind the Stage 1-3 components. ``store_provider`` lazily returns ``(ChromaStore, embedder)`` or None."""
        self.proxy, self.findings, self.csv_path, self.monitor = proxy, findings, Path(financial_csv), monitor
        self.store_provider, self.planner_llm = store_provider, planner_llm
        self.terms, self.sector_move_pct = terms or DealTerms(), sector_move_pct
        self._last: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._dispatch: Dict[str, Callable[[str], ToolResult]] = {
            "query_financial_csv": self.query_financial_csv, "search_audit_findings": self.search_audit_findings,
            "inspect_airgap_logs": lambda _m: self.inspect_airgap_logs(), "explain_valuation_math": lambda _m: self.explain_valuation_math()}

    # ------------------------------------------------------------------ tools
    def query_financial_csv(self, focus: str = "") -> ToolResult:
        """Summarize the 12-quarter financials CSV. Amounts are emitted as ``USD x.xM`` so the proxy scrubs them; ratios stay in clear."""
        from data.synthetic_gen import validate_financials_csv

        name = "query_financial_csv"
        if not self.csv_path.exists():
            return ToolResult(name, False, f"`{self.csv_path.name}` was not found. Call `POST /api/generate-demo`.")
        try:
            rows = validate_financials_csv(self.csv_path)
        except ValueError as exc:
            return ToolResult(name, False, f"`{self.csv_path.name}` does not match the expected schema ({exc}).")
        f, last4, latest = focus.casefold(), rows[-4:], rows[-1]
        num = lambda r, c: float(r[c])  # noqa: E731
        usd = lambda r, c: f"USD {num(r, c):,.1f}M"  # noqa: E731
        if re.search(r"customer|concentration|northgate|client|top", f):
            head = "| Quarter | Northgate Logistics | Solara Retail Partners | Halcyon Pharma Services | Top three |"
            body = [f"| {r['quarter']} | {num(r, 'northgate_share_pct'):.1f}% | {num(r, 'solara_share_pct'):.1f}% | "
                    f"{num(r, 'halcyon_share_pct'):.1f}% | **{num(r, 'top3_share_pct'):.1f}%** |" for r in last4]
        elif re.search(r"churn|retention|nrr|logo", f):
            head = "| Quarter | Gross churn | Net revenue retention | Logos churned | Customers |"
            body = [f"| {r['quarter']} | {num(r, 'gross_churn_pct'):.2f}% | {num(r, 'net_revenue_retention_pct'):.1f}% | "
                    f"{int(num(r, 'logo_churn_count'))} | {int(num(r, 'customers_total'))} |" for r in last4]
        else:
            head = "| Quarter | Revenue | ARR | Net new ARR | EBITDA | EBITDA margin |"
            body = [f"| {r['quarter']} | {usd(r, 'total_revenue_musd')} | {usd(r, 'arr_musd')} | {usd(r, 'net_new_arr_musd')} | "
                    f"{usd(r, 'ebitda_musd')} | {num(r, 'ebitda_margin_pct'):.1f}% |" for r in last4]
        table = "\n".join([head, "|" + "---|" * (head.count("|") - 1)] + body)
        derived = [f"latest quarter **{latest['quarter']}**, {len(rows)} quarters on file",
                   f"top-three customer share averaged **{sum(num(r, 'top3_share_pct') for r in last4) / 4:.1f}%** over the last four quarters"]
        if len(rows) >= 5 and num(rows[-5], "arr_musd"):
            derived.append(f"ARR growth **{(num(latest, 'arr_musd') / num(rows[-5], 'arr_musd') - 1) * 100:+.1f}%** year over year")
            derived.append(f"EBITDA margin **{num(latest, 'ebitda_margin_pct') - num(rows[-5], 'ebitda_margin_pct'):+.1f} pp** year over year")
        return ToolResult(name, True, table + "\n\nDerived: " + "; ".join(derived) + ".", [self.csv_path.name])

    def search_audit_findings(self, query: str = "") -> ToolResult:
        """Rank stored verified findings against the question and add nearest ChromaDB chunks as extra citations."""
        rows = self.findings.list()
        if not rows:
            return ToolResult("search_audit_findings", False,
                              "No verified findings are stored yet. Run the ingest and audit steps (`POST /api/ingest`, `POST /api/audit/run`) first.")
        tokens = [w for w in re.findall(r"[a-z0-9%]+", query.casefold()) if w not in _STOP and len(w) > 2]
        tokens += [_SYNONYMS[w] for w in list(tokens) if w in _SYNONYMS]
        generic = not tokens or bool(re.search(r"\b(all|every|list|findings|risks)\b", query.casefold()))

        def score(r: Dict[str, Any]) -> int:
            hay = f"{r['category'].casefold()} {r['category'].replace('_', ' ').casefold()} {r['title']} {r['summary']} {r['evidence_quote']}".casefold()
            return sum(1 for t in tokens if t in hay)

        ranked = sorted(rows, key=lambda r: (-score(r), _SEV.get(r["severity"], 9)))
        top = ranked[:4] if generic or score(ranked[0]) == 0 else [r for r in ranked if score(r) > 0][:4]
        out, cites = [], []
        for r in top:
            quote = r["evidence_quote"][:240] + ("…" if len(r["evidence_quote"]) > 240 else "")
            cite = f"{r['doc_type']} chunk {r['chunk_id']} p.{r['pages']}"
            cites.append(cite)
            tri = " · ".join(f"{label} {r[col]:.2f}" for label, col in (("context", "context_relevance"), ("groundedness", "groundedness"),
                                                                       ("answer", "answer_relevance")) if r[col] is not None)
            out.append(f"- **[{r['severity']}] {r['title']}** · `{r['category']}` · {r['status']} · HITL: {r['hitl_status']} · id `{r['finding_id']}`\n"
                       f"  > {quote}\n  Triad: {tri or 'n/a'} · citation: {cite}")
        extra = self._chroma_support(query)
        return ToolResult("search_audit_findings", True, "\n".join(out) + extra[0], cites + extra[1])

    def _chroma_support(self, query: str) -> Tuple[str, List[str]]:
        pair = self.store_provider() if self.store_provider else None
        if not pair or not query.strip():
            return "", []
        try:
            chroma, embedder = pair
            with local_processing_guarantee(allow_loopback=False):
                vec = embedder.embed([BGE_QUERY_PREFIX + query])
            with local_processing_guarantee():
                res = chroma.collection.query(query_embeddings=vec, n_results=2, include=["metadatas", "distances"])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("chroma support lookup failed: %s", exc)
            return "", []
        lines, cites = ["\n\nNearest stored chunks (ChromaDB, anonymized):"], []
        for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            pages = meta.get("page_start") if meta.get("page_start") == meta.get("page_end") else f"{meta.get('page_start')}-{meta.get('page_end')}"
            cite = f"{meta.get('doc_type')} chunk {cid} p.{pages}"
            cites.append(cite)
            lines.append(f"- similarity {1 - float(dist):.2f} · {cite}: {str(meta.get('anon_text', ''))[:160]}…")
        return "\n".join(lines), cites

    def inspect_airgap_logs(self) -> ToolResult:
        """Report PII redactions, network allow/block counts and the egress self-test."""
        stats = self.proxy.stats()
        allowed, blocked = self.monitor.counts()
        selftest = self.monitor.selftest if self.monitor.selftest is not None else self.monitor.run_selftest()
        verdict = ("**AIR-GAP INTACT**: no external connection attempt was observed." if blocked == 0
                   else f"**{blocked} external attempt(s) were intercepted and blocked**; nothing left the process.")
        lines = [verdict, "",
                 f"- PII proxy redactions by type: {', '.join(f'{k}={v}' for k, v in sorted(stats.items())) or 'none yet'} (total {sum(stats.values())})",
                 f"- Guarded network operations: loopback/allow-listed **{allowed}**, blocked **{blocked}**",
                 f"- Egress self-test (probes to an external IP and an external DNS name): **{'PASS' if selftest else 'FAIL'}**",
                 f"- Findings stored anonymized-only: **{self.findings.count()}**",
                 "- Parsing, chunking, embedding and scrubbing run under a zero-socket guard; only local DB and model servers use loopback."]
        for e in self.monitor.recent_blocked():
            lines.append(f"- blocked: stage `{e['stage']}` · {e['op']} → `{e['target']}`")
        return ToolResult("inspect_airgap_logs", True, "\n".join(lines), ["security.network_guard audit", "AirGappedProxy.stats()"])

    def explain_valuation_math(self) -> ToolResult:
        """Break down 50 → 34.5 + 10 earn-out, including the reconciliation caveat."""
        t = self.terms
        pct, source = 40.0, "mandate default (no verified Change-of-Control finding stored yet)"
        coc = [r for r in self.findings.list() if r["category"] == "CHANGE_OF_CONTROL"]
        if coc:
            try:
                pct, _ = revenue_exposure_pct([f"{r['summary']} {r['evidence_quote']} {r['financial_exposure'] or ''}" for r in coc])
                source = f"verified finding `{coc[0]['finding_id']}`"
            except AgentError:
                pass
        base, move = t.base_price_musd, self.sector_move_pct
        gross, sector = base * pct / 100.0, base * abs(move) / 100.0
        churn, earn = gross * t.exercise_probability, gross * t.earnout_coverage
        upfront = base - sector - churn
        worst = base - sector - gross
        naive = (base - gross) * (1 + move / 100.0)
        mult = base / t.ebitda_musd
        text = "\n".join([
            f"Revenue exposure of **{pct:.0f}%** comes from {source}.", "",
            "| Step | Amount | Basis |", "|---|---|---|",
            f"| Opening base price | {_m(base)} | EBITDA {_m(t.ebitda_musd)} × {mult:.2f}x implied multiple |",
            f"| Sector multiple compression | {_m(-sector, True)} | {move:+.1f}% public sector de-rating applied to the base (mock market feed) |",
            f"| Churn penalty (risk-weighted) | {_m(-churn, True)} | gross {_m(gross)} ({pct:.0f}% revenue cliff × base) × {t.exercise_probability:.1%} exercise probability |",
            f"| **Upfront cash** | **{_m(upfront)}** | {upfront / t.ebitda_musd:.2f}x EBITDA |",
            f"| Contingent earn-out | {_m(earn, True)} | {t.earnout_coverage:.0%} of gross exposure; payable only if all three customers waive or do not exercise their exit rights |",
            f"| **Maximum consideration** | **{_m(upfront + earn)}** | upfront + earn-out |", "",
            f"- Worst case (all three customers terminate): value {_m(worst)}; the upfront price exceeds it by {_m(upfront - worst)}, which the earn-out does **not** cover.",
            f"- Reconciliation: applying the deductions sequentially, ({_m(base)} − {_m(gross)}) × (1 − {abs(move):.0f}%) = {_m(naive)}, not {_m(upfront)}. "
            "The bridge is additive, and the exercise probability is a **calibration input chosen to reproduce the mandate target**, not a diligence-derived estimate.",
            "- EBITDA and the opening base are mandate inputs, unrelated to the synthetic 10-K revenue."])
        return ToolResult("explain_valuation_math", True, text, ["agents.core.DealTerms", "agents.negotiation bridge arithmetic"])

    # ---------------------------------------------------------------- routing
    def _plan(self, message: str, session_id: str) -> Tuple[List[str], str]:
        if self.planner_llm is not None:
            try:
                system = ("You route questions to tools. Tools: " + "; ".join(f"{k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())
                          + '. Reply with JSON only: {"tools": [one to three tool names from the list]}.')
                names = [n for n in json.loads(self.planner_llm.chat_json(system, f"Question: {message}", max_tokens=60, temperature=0.0)).get("tools", [])
                         if n in self._dispatch]
                if names:
                    return list(dict.fromkeys(names))[:3], "llm"
            except Exception as exc:  # noqa: BLE001  (routing failure degrades to rules)
                LOG.warning("LLM planner unavailable (%s); using rules", exc)
        low = message.casefold()
        names = [n for n, pat in INTENTS.items() if re.search(pat, low)][:3]
        if not names and re.search(r"\b(why|how come|explain|and|what about)\b", low):
            names = list(self._last.get(session_id, []))
        return names, "rules"

    # ---------------------------------------------------------------- chat gate
    def _release(self, text: str) -> Tuple[str, Dict[str, int]]:
        mapping: Dict[str, str] = {}

        def protect(m: re.Match) -> str:
            token = f"zzmndt{len(mapping)}zz"
            mapping[token] = m.group(1)
            return token

        protected = _SENTINEL.sub(protect, text)
        with local_processing_guarantee(allow_loopback=False):
            scrubbed, spans = self.proxy.scrub(protected)
        self.proxy.assert_clean(scrubbed)  # raises LeakageError before anything is returned
        for token, value in mapping.items():
            scrubbed = scrubbed.replace(token, value)
        return scrubbed, dict(Counter(s.label for s in spans))

    def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Answer a question with tools; the returned text is scrubbed, leak-checked and cites its source tools."""
        started = time.perf_counter()
        msg = (message or "").strip()[:1000]
        if not msg:
            return {"answer": "Please ask a question.", "tools_called": [], "sources": [], "planner": "none", "redactions": {}, "scrubbed": True}
        with self._lock:
            names, planner = self._plan(msg, session_id)
            results: List[ToolResult] = []
            for name in names:
                try:
                    results.append(self._dispatch[name](msg))
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("tool %s failed", name)
                    results.append(ToolResult(name, False, f"The tool failed with `{type(exc).__name__}`."))
            if names:
                self._last[session_id] = names
            blocks = []
            for r in results:
                body = r.text if r.tool == "explain_valuation_math" else r.text.replace("«", "").replace("»", "")
                cite = f" · {', '.join(r.citations)}" if r.citations else ""
                blocks.append(f"#### {TITLES[r.tool]}\n{body}\n\n*Source tool: `{r.tool}`{cite}*")
            if not blocks:
                blocks = ["I can answer questions with four tools: " + "; ".join(f"`{k}` ({v})" for k, v in TOOL_DESCRIPTIONS.items())
                          + "\n\nTry: *Why is the bid 34.5 plus a 10 earn-out?*, *Which clauses were verified?*, *Was anything sent outside?*"
                          "\n\n*Source tool: `copilot_help`*"]
            try:
                answer, redactions = self._release("\n\n".join(blocks))
            except (LeakageError, NetworkViolation) as exc:
                LOG.error("copilot response withheld: %s", exc)
                return {"answer": "Response withheld: the egress gate detected sensitive material.", "tools_called": names, "sources": [],
                        "planner": planner, "redactions": {}, "scrubbed": True, "withheld": True}
        return {"answer": answer, "tools_called": names, "planner": planner, "redactions": redactions, "scrubbed": True,
                "sources": [{"tool": r.tool, "ok": r.ok, "citations": r.citations} for r in results],
                "session_id": session_id, "elapsed_ms": round((time.perf_counter() - started) * 1000)}