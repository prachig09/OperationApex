"""LangGraph wiring: legal -> market -> financial -> debate loop -> finalize.

Falls back to an equivalent in-process sequential runner if ``langgraph`` is not installed.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, TypedDict

from agents.core import AnonymizedBrief, DealTerms, FinancialAssessment, LegalAssessment, MarketAssessment
from agents.negotiation import FinalOffer, NegotiationState, final_offer, open_negotiation, run_round
from agents.nodes import AuditLegalAgent, FinancialAuditAgent, MarketIntelligenceAgent

try:  # pragma: no cover - import guard
    from langgraph.graph import END, START, StateGraph

    HAVE_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAVE_LANGGRAPH = False


class DealState(TypedDict, total=False):
    """Shared graph state."""

    brief: AnonymizedBrief
    legal: LegalAssessment
    market: MarketAssessment
    financial: FinancialAssessment
    negotiation: NegotiationState
    offer: FinalOffer


def _make_nodes(legal: AuditLegalAgent, market: MarketIntelligenceAgent, fin: FinancialAuditAgent, terms: DealTerms) -> Dict[str, Callable]:
    def audit_legal(state: DealState) -> Dict[str, Any]:
        return {"legal": legal.run(state["brief"])}

    def market_intel(state: DealState) -> Dict[str, Any]:
        return {"market": market.run(state["brief"])}

    def financial_audit(state: DealState) -> Dict[str, Any]:
        assessment = fin.run(state["legal"], state["market"])
        return {"financial": assessment, "negotiation": open_negotiation(terms, state["legal"], state["market"], assessment)}

    def debate_round(state: DealState) -> Dict[str, Any]:
        return {"negotiation": run_round(state["negotiation"], terms)}

    def route(state: DealState) -> str:
        return "done" if state["negotiation"].all_settled else "continue"

    def finalize(state: DealState) -> Dict[str, Any]:
        return {"offer": final_offer(state["negotiation"], terms)}

    return {"audit_legal": audit_legal, "market_intel": market_intel, "financial_audit": financial_audit,
            "debate_round": debate_round, "route": route, "finalize": finalize}


def run_consensus(
    brief: AnonymizedBrief, legal: AuditLegalAgent, market: MarketIntelligenceAgent, fin: FinancialAuditAgent, terms: DealTerms,
) -> Tuple[DealState, str]:
    """Execute the agent graph and return ``(final_state, engine_name)``."""
    nodes = _make_nodes(legal, market, fin, terms)
    if HAVE_LANGGRAPH:
        graph = StateGraph(DealState)
        for name in ("audit_legal", "market_intel", "financial_audit", "debate_round", "finalize"):
            graph.add_node(name, nodes[name])
        graph.add_edge(START, "audit_legal")
        graph.add_edge("audit_legal", "market_intel")
        graph.add_edge("market_intel", "financial_audit")
        graph.add_edge("financial_audit", "debate_round")
        graph.add_conditional_edges("debate_round", nodes["route"], {"continue": "debate_round", "done": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile().invoke({"brief": brief}, config={"recursion_limit": 100}), "langgraph"
    state: Dict[str, Any] = {"brief": brief}
    for name in ("audit_legal", "market_intel", "financial_audit"):
        state.update(nodes[name](state))
    while True:
        state.update(nodes["debate_round"](state))
        if nodes["route"](state) == "done":
            break
    state.update(nodes["finalize"](state))
    return state, "sequential-fallback"  # type: ignore[return-value]