"""Executive memo synthesizer: Investment Recommendation Memo with air-gap integrity evidence."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from agents.core import AnonymizedBrief, FinancialAssessment, LegalAssessment, MarketAssessment
from agents.negotiation import FinalOffer, NegotiationState


@dataclass
class MemoInputs:
    """Everything the memo needs; all text fields are already anonymized."""

    offer: FinalOffer
    negotiation: NegotiationState
    legal: LegalAssessment
    market: MarketAssessment
    financial: FinancialAssessment
    brief: AnonymizedBrief
    triad_rows: List[Dict[str, Any]]
    drift: Dict[str, Any]
    evidence: Dict[str, Any]
    engine: str


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _m(value: float, signed: bool = False) -> str:
    return f"{value:+,.1f}" if signed else f"{value:,.1f}"


def build_memo(m: MemoInputs) -> str:
    """Render the Investment Recommendation Memo (markdown, anonymized placeholders only)."""
    o, n, ev = m.offer, m.negotiation, m.evidence
    led = ev["ledger"]
    passed = led["clean"] and ev["network_blocked"] == 0 and ev["guard_selftest"]
    parts: List[str] = [
        "# Investment Recommendation Memo: Project Apex",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M %Z')} | classification: CONFIDENTIAL | entities shown as proxy placeholders | orchestration: {m.engine}*",
        "## 1. Recommendation",
        f"**Submit a revised bid of ${_m(o.upfront_musd)}M upfront cash plus a ${_m(o.earnout_musd)}M contingent earn-out "
        f"(maximum headline consideration ${_m(o.headline_musd)}M), replacing the ${_m(o.base_price_musd)}M opening base price.**",
        f"The earn-out is payable only if all three top customers (≈{m.legal.exposure_pct:.0f}% of revenue) waive or fail to exercise "
        "their Change of Control termination rights inside the contractual window. Obtaining those waivers before signing is the preferred path.",
        "## 2. Valuation bridge (USD millions)",
        _table(["Step", "USD M", "Basis"], [
            ["Opening base price", _m(o.base_price_musd), f"EBITDA {m.financial.ebitda_musd:.2f} x {m.financial.base_multiple:.2f}x implied multiple"],
            ["Sector multiple compression", _m(-o.sector_deduction_musd, True),
             f"{m.market.sector_move_pct:+.1f}% public sector de-rating ({m.financial.base_multiple:.2f}x → {m.financial.adjusted_multiple:.2f}x)"],
            ["Churn risk penalty (risk-weighted)", _m(-o.churn_deduction_musd, True),
             f"gross {_m(o.gross_churn_musd)} ({m.legal.exposure_pct:.0f}% revenue cliff) x {m.financial.exercise_probability:.1%} exercise probability"],
            ["**Upfront cash at close**", f"**{_m(o.upfront_musd)}**", "consensus of the three agents"],
            ["Contingent earn-out", _m(o.earnout_musd, True), f"{m.legal.earnout_ask_musd / o.gross_churn_musd:.0%} of gross exposure; retention-conditional"],
            ["**Maximum total consideration**", f"**{_m(o.headline_musd)}**", ""],
        ]),
        "## 3. Key legal and financial risks (Audit & Legal agent)",
    ]
    for c in m.legal.clauses[:8]:
        quote = c["evidence_quote"][:220] + ("…" if len(c["evidence_quote"]) > 220 else "")
        parts.append(f"- **[{c['severity']}] {c['title']}** ({c['category']}): {c['summary']}  \n  Evidence: *\"{quote}\"*")
    parts += [
        f"\nRevenue-exposure figure reconciles across per-customer shares: **{m.legal.reconciled}**. Risk rating: **{m.legal.risk_rating}**.",
        "## 4. Negotiation record",
        _table(["Round", "Churn: Legal ask", "Churn: Financial offer", "Sector: Market ask", "Sector: Financial offer"],
               [[int(h["round"]), _m(h["churn_buyer"]), _m(h["churn_seller"]), _m(h["sector_buyer"]), _m(h["sector_seller"])] for h in n.history]),
        "\n" + _table(["Lever", "Settled deduction (USD M)", "Status"],
                      [[name, _m(lv.value or 0.0), lv.status] for name, lv in n.levers.items()]
                      + [["earn_out", _m(n.earnout_agreed or 0.0), "agreed" if n.earnout_agreed else "none"]]),
        "\n**Key exchanges**\n" + "\n".join(
            f"- R{t['round']} `{t['agent']}` → {t['lever']} ({t['action']}"
            + (f", {t['value']:,.2f}" if t["value"] is not None else "") + f"): {t['message']}"
            for t in n.transcript if t["action"] in ("propose", "counter", "settle", "mediate", "accept")),
        "## 5. Market intelligence",
        f"Tool: `{m.market.tool}` (query released through the proxy gate): *{m.market.query}*  \n"
        f"Sector move: **{m.market.sector_move_pct:+.1f}%**; corroborated by peer EV/Revenue: **{m.market.corroborated}**; breadth: {m.market.breadth}",
        "\n".join(f"- {s['title']} ({s['url']})" for s in m.market.sources),
        "## 6. Observability (RAG Triad and drift)",
        _table(["Query", "Status", "Context rel.", "Groundedness", "Answer rel."],
               [[r["name"], r["status"], f"{r.get('context_relevance', 0):.2f}", f"{r.get('groundedness', 0):.2f}", f"{r.get('answer_relevance', 0):.2f}"]
                for r in m.triad_rows]),
    ]
    retracted = [r for r in m.triad_rows if r["status"] == "RETRACTED"]
    if retracted:
        parts.append("\n**Retracted (excluded from agent input):** " + "; ".join(f"{r['name']} ({'; '.join(r.get('reasons', []))})" for r in retracted))
    d = m.drift
    if d.get("status") in ("STABLE", "DRIFT_WARNING"):
        parts.append(
            f"\nDrift check on held-out document `{d['holdout']}`: **{d['status']}** (cosine {d['centroid_cosine_distance']:.4f} vs "
            f"threshold {d['cosine_threshold']:.4f}; KL {d['kl_divergence']:.4f} vs {d['kl_threshold']:.4f}; baseline n={d['n_baseline']}, new n={d['n_new']}). "
            "A warning is advisory here because the held-out file is a different document class.")
    else:
        parts.append(f"\nDrift check: **{d.get('status', 'SKIPPED')}**: {d.get('reason', '')}")
    parts += [
        "## 7. Residual risk and conditions",
        f"- If all three customers terminate, value falls to **${_m(o.full_loss_value_musd)}M**; the upfront price exceeds that by **${_m(o.residual_exposure_musd)}M**, which the earn-out does **not** cover.",
        "- Recommended conditions: pre-signing waivers or amendments from the three customers; escrow or holdback sized to the residual exposure; earn-out tested at 12 and 24 months.",
        "- The exercise probability, EBITDA and opening base are mandate inputs, not diligence outputs. Calibrate them before relying on the figures.",
        "## 8. Air-gap integrity evidence",
        f"**Overall result: {'PASS' if passed else 'FAIL'}**",
        _table(["Check", "Result"], [
            ["Guard self-test (external IP + DNS probes blocked)", "PASS" if ev["guard_selftest"] else "FAIL"],
            ["Pipeline network audit (stages 1-3, loopback only)", f"{ev['network_summary']}"],
            ["Ingestion-stage audit (parse/embed: zero sockets)", ev["ingest_audit"]],
            ["Boundary releases (internal hand-offs / external queries)", f"{led['internal_events']} / {led['external_events']}, all passed the proxy gate"],
            ["Residual-PII scan of every release (vault + regex rules)", f"{len(led['leaks'])} leaks"],
            [f"Canary scan ({led['canaries_checked']} known identifiers)", f"{len(led['canary_hits'])} hits"],
            ["Proxy replacements by label", ", ".join(f"{k}={v}" for k, v in sorted(ev['proxy_stats'].items())) or "none"],
            ["Raw documents, embeddings or contexts sent externally", "0 (only the generic sector query left the process boundary)"],
        ]),
        "\n" + _table(["#", "Boundary", "Channel", "SHA-256 (12)", "Chars", "External payload"],
                      [[e["seq"], e["boundary"], e["channel"], e["sha256"], e["chars"], e["preview"]] for e in led["event_table"]]),
        "\n*Scope: this is evidence for the checks above (rule/NER detection, vault re-scan, canary list, process-level socket guard), "
        "not a formal proof. The guard does not cover subprocesses or native extensions; enforce host egress rules for a hard guarantee.*",
    ]
    return "\n\n".join(parts) + "\n"