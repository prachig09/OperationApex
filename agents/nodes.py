"""The three specialised agents: Audit & Legal, Market Intelligence, Financial Audit."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from agents.core import (
    AgentError, AnonymizedBrief, DealTerms, EgressLedger, FinancialAssessment, LegalAssessment,
    MarketAssessment, MarketSearchTool, current_year,
)

_PCT_BEFORE = re.compile(r"(\d+(?:\.\d+)?)\s?%(?=[^.]{0,60}revenue)", re.I)
_PCT_AFTER = re.compile(r"revenue[^.%]{0,60}?(\d+(?:\.\d+)?)\s?%", re.I)
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def revenue_exposure_pct(texts: List[str]) -> Tuple[float, bool]:
    """Extract the aggregate revenue share (%) tied to Change of Control language.

    Only percentages stated next to the word "revenue" are considered, so ownership
    thresholds such as "(50%)" in the clause definition are ignored. Returns the largest
    value and whether the remaining figures sum to it (per-customer shares reconcile).
    """
    values: Set[float] = set()
    for text in texts:
        for pattern in (_PCT_BEFORE, _PCT_AFTER):
            values.update(float(m) for m in pattern.findall(text))
    values = {v for v in values if 0 < v <= 100}
    if not values:
        raise AgentError("no revenue-exposure percentage found in verified Change of Control findings")
    top = max(values)
    rest = values - {top}
    return top, bool(rest) and abs(sum(rest) - top) <= 0.5


class AuditLegalAgent:
    """Consumes anonymized Stage 2 findings and quantifies the Change of Control exposure."""

    name = "audit_legal"

    def __init__(self, terms: DealTerms) -> None:
        """Bind deal terms."""
        self.terms = terms

    def run(self, brief: AnonymizedBrief) -> LegalAssessment:
        """Assess legal risk strictly from the anonymized brief."""
        coc = [f for f in brief.findings if f["category"] == "CHANGE_OF_CONTROL"]
        if not coc:
            raise AgentError("no verified CHANGE_OF_CONTROL finding is available; cannot price the deal")
        texts = [" ".join(str(f.get(k) or "") for k in ("summary", "evidence_quote", "financial_exposure")) for f in coc]
        pct, reconciled = revenue_exposure_pct(texts)
        gross = round(self.terms.base_price_musd * pct / 100.0, 6)
        ranked = sorted(brief.findings, key=lambda f: _SEVERITY_ORDER.get(f["severity"], 9))
        seen, clauses = set(), []
        for f in ranked:
            key = (f["category"], f["title"])
            if key in seen:
                continue
            seen.add(key)
            clauses.append({k: str(f.get(k) or "") for k in ("category", "severity", "title", "summary", "evidence_quote")})
        rating = "CRITICAL" if pct >= 30 else "HIGH" if pct >= 15 else "MEDIUM"
        return LegalAssessment(
            exposure_pct=pct, reconciled=reconciled, gross_penalty_musd=gross,
            earnout_ask_musd=round(gross * self.terms.earnout_coverage, 6), risk_rating=rating, clauses=clauses[:10],
        )


class MarketIntelligenceAgent:
    """Turns anonymized vector signals into a public sector query and reads valuation trends."""

    name = "market_intel"

    def __init__(self, tool: MarketSearchTool, ledger: EgressLedger) -> None:
        """Bind the external tool and the egress ledger (the only route out)."""
        self.tool, self.ledger = tool, ledger

    def run(self, brief: AnonymizedBrief) -> MarketAssessment:
        """Run the gated public query and extract the sector multiple move."""
        query = f"{' '.join(brief.sector_terms)} sector valuation multiples EV/revenue trend {current_year()}"
        gated = self.ledger.release("external", "market_search_query", {"query": query})["query"]
        results = self.tool.search(gated)
        metrics: Dict[str, Dict[str, Any]] = {r["metric"]["name"]: r["metric"] for r in results if r.get("metric")}
        index = metrics.get("sector_multiple_index")
        if not index or not index["previous"]:
            raise AgentError("market tool returned no sector multiple index")
        move = round((index["current"] / index["previous"] - 1.0) * 100.0, 1)
        ev = metrics.get("peer_ev_revenue_median")
        corroborated = bool(ev) and abs((ev["current"] / ev["previous"] - 1.0) * 100.0 - move) <= 1.0
        breadth = next((r["snippet"] for r in results if "peers" in r.get("snippet", "") and not r.get("metric")), "n/a")
        return MarketAssessment(
            query=gated, sector_move_pct=move, corroborated=corroborated, breadth=breadth,
            sources=[{"title": r["title"], "url": r["url"]} for r in results], tool=self.tool.name,
        )


class FinancialAuditAgent:
    """Derives the baseline from earnings multiples and prepares risk-weighted counter-positions."""

    name = "financial_audit"

    def __init__(self, terms: DealTerms) -> None:
        """Bind deal terms."""
        self.terms = terms

    def run(self, legal: LegalAssessment, market: MarketAssessment) -> FinancialAssessment:
        """Evaluate the baseline valuation (EBITDA x multiple) and the sector-adjusted value."""
        t = self.terms
        multiple = t.base_price_musd / t.ebitda_musd
        adjusted = multiple * (1.0 + market.sector_move_pct / 100.0)
        markdown = round(t.base_price_musd - t.ebitda_musd * adjusted, 6)
        return FinancialAssessment(
            ebitda_musd=t.ebitda_musd, base_multiple=round(multiple, 4), adjusted_multiple=round(adjusted, 4),
            full_markdown_musd=markdown, sector_opening_musd=round(markdown * t.sector_opening_passthrough, 6),
            risk_weighted_churn_musd=round(legal.gross_penalty_musd * t.exercise_probability, 6),
            earnout_cap_musd=round(t.base_price_musd * t.earnout_cap_pct, 6),
            exercise_probability=t.exercise_probability,
        )