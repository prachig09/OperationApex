"""Multi-agent negotiation loop over three price levers plus the earn-out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.core import DealTerms, FinancialAssessment, LegalAssessment, MarketAssessment, Position

EPS = 1e-9


@dataclass
class LeverState:
    """A negotiable deduction with a buyer-side and a seller-side position."""

    name: str
    buyer: Position
    seller: Position
    requires_earnout: bool = False
    settled: bool = False
    value: Optional[float] = None
    status: str = "open"


@dataclass
class NegotiationState:
    """Mutable debate state carried through the graph."""

    levers: Dict[str, LeverState]
    base_price_musd: float
    earnout_ask: float
    earnout_cap: float
    earnout_agreed: Optional[float] = None
    round: int = 0
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def all_settled(self) -> bool:
        """True when every lever is settled."""
        return all(lv.settled for lv in self.levers.values())


@dataclass
class FinalOffer:
    """The consensus bid."""

    base_price_musd: float
    sector_deduction_musd: float
    churn_deduction_musd: float
    gross_churn_musd: float
    upfront_musd: float
    earnout_musd: float
    headline_musd: float
    full_loss_value_musd: float
    residual_exposure_musd: float
    statuses: Dict[str, str]


def open_negotiation(terms: DealTerms, legal: LegalAssessment, market: MarketAssessment, fin: FinancialAssessment) -> NegotiationState:
    """Build opening positions from the three agents' assessments."""
    full = round(terms.base_price_musd * abs(market.sector_move_pct) / 100.0, 6)
    churn = LeverState(
        "churn_penalty",
        buyer=Position("churn_penalty", "audit_legal", "buyer", legal.gross_penalty_musd, fin.risk_weighted_churn_musd, 0.5,
                       f"{legal.exposure_pct:.0f}% of revenue sits behind no-penalty Change of Control exits; demand the full gross penalty."),
        seller=Position("churn_penalty", "financial_audit", "seller", fin.risk_weighted_churn_musd, legal.gross_penalty_musd, 0.0,
                        f"Expected loss is gross exposure x {fin.exercise_probability:.1%} exercise probability; analytic, not a lowball."),
        requires_earnout=True,
    )
    sector = LeverState(
        "sector_multiple",
        buyer=Position("sector_multiple", "market_intel", "buyer", full, full, 0.0,
                       f"Public sector multiples moved {market.sector_move_pct:+.1f}% (corroborated={market.corroborated}); market data is not negotiable."),
        seller=Position("sector_multiple", "financial_audit", "seller", fin.sector_opening_musd, full, terms.sector_opening_passthrough,
                        "Open with partial pass-through: part of the de-rating may be transitory."),
    )
    return NegotiationState(
        {churn.name: churn, sector.name: sector}, terms.base_price_musd,
        earnout_ask=legal.earnout_ask_musd, earnout_cap=fin.earnout_cap_musd,
    )


def _say(state: NegotiationState, agent: str, lever: str, action: str, value: Optional[float], message: str) -> None:
    state.transcript.append({"round": state.round, "agent": agent, "lever": lever, "action": action,
                             "value": value, "message": message})


def _settle(state: NegotiationState, lv: LeverState, status: str, why: str) -> None:
    lv.value = round((lv.buyer.value + lv.seller.value) / 2.0, 1)
    lv.settled, lv.status = True, status
    _say(state, "mediator", lv.name, "settle" if status == "agreed" else "mediate", lv.value, why)


def run_round(state: NegotiationState, terms: DealTerms) -> NegotiationState:
    """Advance the debate by one round (concessions, settlement, final mediation)."""
    state.round += 1
    if state.round == 1:
        _say(state, "financial_audit", "base_price", "propose", state.base_price_musd,
             "Baseline equals EBITDA x implied earnings multiple.")
        _say(state, "audit_legal", "base_price", "accept", state.base_price_musd, "No legal objection to the opening base.")
        _say(state, "market_intel", "base_price", "accept", state.base_price_musd, "Base accepted pending sector adjustment.")
        agreed = min(state.earnout_ask, state.earnout_cap)
        _say(state, "audit_legal", "earn_out", "propose", state.earnout_ask,
             "Convert half of the gross exposure into consideration payable only if the three customers waive exit rights.")
        secured = agreed >= state.earnout_ask - EPS
        _say(state, "financial_audit", "earn_out", "accept" if secured else "counter", agreed,
             "Within the earn-out cap; accepted." if secured else "Clipped to the earn-out cap.")
        state.earnout_agreed = agreed
    secured = state.earnout_agreed is not None and state.earnout_agreed >= state.earnout_ask - EPS

    for lv in state.levers.values():
        if lv.settled:
            continue
        b, s = lv.buyer, lv.seller
        if state.round == 1:
            _say(state, b.agent, lv.name, "offer", b.value, b.rationale)
            _say(state, s.agent, lv.name, "offer", s.value, s.rationale)
        gap = b.value - s.value
        if gap <= terms.tolerance_musd:
            _settle(state, lv, "agreed", f"Gap {gap:.3f} within tolerance; settled at the midpoint.")
            continue
        b_rate = b.rate if (secured or not lv.requires_earnout) else 0.0
        new_b = max(b.limit, b.value - b_rate * gap)
        new_s = min(s.limit, s.value + s.rate * gap)
        stalled = abs(new_b - b.value) < EPS and abs(new_s - s.value) < EPS
        if abs(new_b - b.value) > EPS:
            _say(state, b.agent, lv.name, "concede", new_b, "Concession conditional on the secured earn-out." if lv.requires_earnout else "Concession toward counter-offer.")
        if abs(new_s - s.value) > EPS:
            _say(state, s.agent, lv.name, "concede", new_s, "Raised offer toward the evidence-backed figure.")
        b.value, s.value = new_b, new_s
        if stalled:
            _settle(state, lv, "mediated", "Both sides at their limits; mediator split the remaining gap.")

    if state.round >= terms.max_rounds:
        for lv in state.levers.values():
            if not lv.settled:
                _settle(state, lv, "mediated", "Round limit reached; mediator split the remaining gap.")
    state.history.append({
        "round": float(state.round),
        "churn_buyer": state.levers["churn_penalty"].buyer.value, "churn_seller": state.levers["churn_penalty"].seller.value,
        "sector_buyer": state.levers["sector_multiple"].buyer.value, "sector_seller": state.levers["sector_multiple"].seller.value,
    })
    return state


def final_offer(state: NegotiationState, terms: DealTerms) -> FinalOffer:
    """Compute the consensus bid from settled levers."""
    churn, sector = state.levers["churn_penalty"], state.levers["sector_multiple"]
    gross = churn.buyer.limit if churn.buyer.limit > churn.seller.value else churn.seller.limit
    gross = max(churn.seller.limit, churn.buyer.limit)
    upfront = round(state.base_price_musd - sector.value - churn.value, 2)
    earnout = round(state.earnout_agreed or 0.0, 2)
    full_loss = round(state.base_price_musd - sector.value - gross, 2)
    return FinalOffer(
        base_price_musd=state.base_price_musd, sector_deduction_musd=sector.value, churn_deduction_musd=churn.value,
        gross_churn_musd=round(gross, 2), upfront_musd=upfront, earnout_musd=earnout,
        headline_musd=round(upfront + earnout, 2), full_loss_value_musd=full_loss,
        residual_exposure_musd=round(upfront - full_loss, 2),
        statuses={name: lv.status for name, lv in state.levers.items()},
    )