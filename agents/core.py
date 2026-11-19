"""Shared contracts for the Stage 3 agent system.

Holds deal terms, agent assessments, the anonymized brief, the egress ledger
(every hand-off passes the Stage 1 proxy gate and is recorded) and the external
market-search tool boundary.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from llm.extractor import ExtractionResult
from security.proxy import AirGappedProxy, LeakageError


class AgentError(RuntimeError):
    """An agent could not complete its task from verified inputs."""


@dataclass(frozen=True)
class DealTerms:
    """Mandate-level deal inputs shared by all agents (USD millions unless noted).

    Attributes:
        base_price_musd: Opening base price.
        ebitda_musd: Baseline EBITDA used to derive the implied earnings multiple.
        exercise_probability: Probability-weight applied to the gross churn exposure.
        earnout_coverage: Share of the gross exposure converted into a contingent earn-out.
        earnout_cap_pct: Maximum earn-out the Financial agent accepts, as a share of base price.
        sector_opening_passthrough: Share of the sector markdown the Financial agent opens with.
        tolerance_musd: Gap at which a lever is considered agreed.
        max_rounds: Debate rounds before the mediator settles any open lever.
    """

    base_price_musd: float = 50.0
    ebitda_musd: float = 6.25
    exercise_probability: float = 0.325
    earnout_coverage: float = 0.5
    earnout_cap_pct: float = 0.25
    sector_opening_passthrough: float = 0.5
    tolerance_musd: float = 0.1
    max_rounds: int = 12


@dataclass
class AnonymizedBrief:
    """Everything the agents may see: proxy-scrubbed findings and vector signals only."""

    findings: List[Dict[str, Any]]
    signals: Dict[str, Any]
    sector_terms: List[str]


@dataclass
class Position:
    """One agent's stance on a price lever. ``value`` is a deduction in USD M (positive lowers price)."""

    lever: str
    agent: str
    side: str          # "buyer" (wants a larger deduction) or "seller" (wants a smaller one)
    value: float
    limit: float       # buyer: lowest deduction accepted; seller: highest deduction accepted
    rate: float        # fraction of the gap conceded per round
    rationale: str


@dataclass
class LegalAssessment:
    """Output of the Audit & Legal agent."""

    exposure_pct: float
    reconciled: bool
    gross_penalty_musd: float
    earnout_ask_musd: float
    risk_rating: str
    clauses: List[Dict[str, str]]


@dataclass
class MarketAssessment:
    """Output of the Market Intelligence agent."""

    query: str
    sector_move_pct: float
    corroborated: bool
    breadth: str
    sources: List[Dict[str, str]]
    tool: str


@dataclass
class FinancialAssessment:
    """Output of the Financial Audit agent."""

    ebitda_musd: float
    base_multiple: float
    adjusted_multiple: float
    full_markdown_musd: float
    sector_opening_musd: float
    risk_weighted_churn_musd: float
    earnout_cap_musd: float
    exercise_probability: float


# ------------------------------------------------------------------ egress ledger
@dataclass
class EgressEvent:
    """One gated release across a boundary."""

    seq: int
    boundary: str      # "internal" (agent hand-off) or "external" (public query)
    channel: str
    sha256: str
    chars: int
    preview: str


class EgressLedger:
    """Gates every payload through the proxy and records what was released."""

    def __init__(self, proxy: AirGappedProxy) -> None:
        """Bind the Stage 1 proxy."""
        self.proxy = proxy
        self.events: List[EgressEvent] = []
        self._texts: List[str] = []

    def release(self, boundary: str, channel: str, payload: Any) -> Any:
        """Sanitize + verify ``payload`` (raises ``LeakageError`` on failure) and record the release."""
        clean = self.proxy.egress_gate(payload)
        text = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        self.events.append(EgressEvent(
            len(self.events) + 1, boundary, channel,
            hashlib.sha256(text.encode()).hexdigest(), len(text), text[:160],
        ))
        self._texts.append(text)
        return clean

    def audit(self, canaries: Sequence[str]) -> Dict[str, Any]:
        """Re-scan every released payload against the (now complete) vault, rules and canary list."""
        leaks: List[str] = []
        hits: List[str] = []
        for event, text in zip(self.events, self._texts):
            try:
                self.proxy.assert_clean(text)
            except LeakageError as exc:
                leaks.append(f"#{event.seq} {event.channel}: {exc}")
            for canary in canaries:
                if re.search(re.escape(canary).replace(r"\ ", r"\s+"), text, re.I):
                    hits.append(f"#{event.seq} {event.channel}: {canary}")
        external = [e for e in self.events if e.boundary == "external"]
        return {
            "events": len(self.events), "external_events": len(external),
            "internal_events": len(self.events) - len(external), "leaks": leaks, "canary_hits": hits,
            "canaries_checked": len(canaries), "clean": not leaks and not hits,
            "event_table": [
                {"seq": e.seq, "boundary": e.boundary, "channel": e.channel, "sha256": e.sha256[:12],
                 "chars": e.chars, "preview": e.preview if e.boundary == "external" else ""}
                for e in self.events
            ],
        }


# ------------------------------------------------------- Stage 1 identifiers
def stage1_known_entities() -> Dict[str, List[str]]:
    """Seed names for the proxy deny-list (same entities as the synthetic data room)."""
    from data.synthetic_gen import CUSTOMERS, TARGET

    return {
        "COMPANY": [TARGET] + [c.name for c in CUSTOMERS],
        "EXEC": ["Dana R. Whitfield", "Marcus Okonkwo-Reyes", "Priya Venkataraman"]
        + [c.signatory.split(",")[0].strip() for c in CUSTOMERS],
    }


def stage1_canaries() -> List[str]:
    """Independent canary strings that must never appear in any released payload."""
    from data.synthetic_gen import CUSTOMERS

    known = stage1_known_entities()
    canaries = list(known["COMPANY"]) + list(known["EXEC"])
    canaries += ["Meridian Helix", "Northgate", "Solara", "Halcyon", "Whitfield", "Okonkwo-Reyes",
                 "Venkataraman", "Albrecht", "Castellanos", "Menon"]
    canaries += [c.contract_id for c in CUSTOMERS]
    canaries += ["412.6", "371.9", "70,100,000", "65,000,000", "126.5", "84-3917265", "001-39482", "555-0147"]
    return list(dict.fromkeys(canaries))


# ------------------------------------------------------------------ brief builder
SECTOR_VOCAB = ("supply chain", "analytics", "orchestration", "cloud", "subscription", "life-sciences", "enterprise software")


def build_brief(results: Sequence[ExtractionResult], proxy: AirGappedProxy, ledger: EgressLedger) -> AnonymizedBrief:
    """Turn Stage 2 results into the anonymized brief; every field passes the proxy gate."""
    findings: List[Dict[str, Any]] = []
    texts: List[str] = []
    sims: List[float] = []
    clause_chunks = 0
    for res in results:
        for f in res.findings:
            findings.append(ledger.release("internal", "legal_agent_input", {
                "query_text": res.query, "category": f.category, "severity": f.severity, "title": f.title,
                "summary": f.summary, "evidence_quote": f.evidence_quote,
                "financial_exposure": f.financial_exposure, "recommended_action": f.recommended_action,
            }))
        for ctx in res.contexts:
            texts.append(proxy.anonymize(ctx.text))
            if ctx.similarity is not None:
                sims.append(ctx.similarity)
            clause_chunks += 1 if ctx.clause_types else 0
    blob = " ".join(texts).casefold()
    terms = [t for t in SECTOR_VOCAB if t in blob] or ["enterprise software"]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    signals = ledger.release("internal", "vector_signals", {
        "contexts": str(len(texts)), "mean_similarity": f"{mean_sim:.3f}",
        "clause_chunks": str(clause_chunks), "sector_terms": terms,
    })
    return AnonymizedBrief(findings=findings, signals=signals, sector_terms=terms)


# ---------------------------------------------------------------- market tool
class MarketSearchTool(Protocol):
    """Contract for the external public-search tool."""

    name: str

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Return result dicts with ``title``, ``url``, ``snippet`` and optional ``metric``."""


class MockWebSearchTool:
    """Deterministic stand-in for a public web-search / market-data API. Opens no sockets."""

    name = "mock_web_search"

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Return canned 90-day sector-multiple results (mock data, labelled as such)."""
        sector = "supply chain analytics software" if "supply chain" in query.casefold() else "enterprise software"
        return [
            {"title": f"[MOCK] Public {sector} multiples, 90-day review", "url": "https://example.invalid/sector-index",
             "snippet": "Sector multiple index fell from 100.0 to 82.0 over the trailing 90 days.",
             "metric": {"name": "sector_multiple_index", "previous": 100.0, "current": 82.0, "window_days": 90}},
            {"title": f"[MOCK] Peer EV/Revenue medians, {sector}", "url": "https://example.invalid/peer-ev-revenue",
             "snippet": "Median EV/Revenue compressed from 6.1x to 5.0x across listed peers.",
             "metric": {"name": "peer_ev_revenue_median", "previous": 6.1, "current": 5.0, "window_days": 90}},
            {"title": "[MOCK] Breadth of the sector sell-off", "url": "https://example.invalid/breadth",
             "snippet": "8 of 9 listed peers de-rated; no peer re-rated upward.", "metric": None},
        ]


def current_year() -> str:
    """Current year as a string (used in the public query)."""
    return time.strftime("%Y")