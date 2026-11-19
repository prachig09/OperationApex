"""Synthetic confidential data-room generator.

Produces three fictional target-company PDFs for Meridian Helix Technologies,
Inc. (10-K risk filing, Master Services Agreement, and a customer schedule with
an omnibus amendment). The three largest customers, representing 40% of
revenue, each hold a buried "Change of Control" right to terminate without
penalty. A ground_truth.json records where the clause lives for later
evaluation of the audit agents. All entities are fictional.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import ApexConfig

import csv
import time
from typing import Any, Dict, List, Optional

TARGET = "Meridian Helix Technologies, Inc."
TOTAL_REVENUE_MUSD = 412.6
PRIOR_REVENUE_MUSD = 371.9


@dataclass(frozen=True)
class Customer:
    """A top-three customer of the fictional target."""

    name: str
    short: str
    share_pct: float
    revenue_musd: float
    contract_id: str
    effective: str
    signatory: str


CUSTOMERS = (
    Customer("Northgate Logistics Group, Inc.", "Northgate", 17.0, 70.1, "MSA-2023-004817", "March 1, 2023", "Thomas J. Albrecht, Chief Procurement Officer"),
    Customer("Solara Retail Partners LLC", "Solara", 13.0, 53.6, "MSA-2022-003291", "June 15, 2022", "Elena M. Castellanos, Chief Operating Officer"),
    Customer("Halcyon Pharma Services plc", "Halcyon", 10.0, 41.3, "MSA-2021-002058", "October 1, 2021", "Rajiv S. Menon, Chief Financial Officer"),
)

COC_DEFINITION = (
    "&quot;Change of Control&quot; means (a) the acquisition by any person or group of beneficial ownership of "
    "more than fifty percent (50%) of the voting securities of Supplier; (b) a merger or consolidation of "
    "Supplier in which its pre-transaction stockholders hold less than fifty percent (50%) of the voting power "
    "of the surviving entity; or (c) the sale of all or substantially all of the assets of Supplier."
)

GENERIC_RISK = [
    "Our results of operations may fluctuate significantly from quarter to quarter, which makes our future performance difficult to predict.",
    "We operate in highly competitive markets, and competitors with greater resources may reduce our market share or pressure our pricing.",
    "Our success depends on the continued service of our senior management team and our ability to hire and retain skilled engineers.",
    "Security incidents or unauthorized access to customer data could damage our reputation and expose us to material liability.",
    "Changes in laws governing data privacy and cross-border data transfers may increase our compliance costs.",
    "Our insurance coverage may not be adequate to cover all claims that may arise in the ordinary course of business.",
    "Economic downturns may cause customers to delay or reduce technology spending, adversely affecting our revenue growth.",
    "We rely on third-party cloud infrastructure providers, and any disruption of those services could impair our platform.",
    "Failure to protect our intellectual property rights could weaken our competitive position.",
    "We may be unable to integrate acquired businesses successfully or realize anticipated benefits from them.",
]

FINANCIAL_COLUMNS = (
    "quarter", "period_end", "total_revenue_musd", "arr_musd", "net_new_arr_musd", "ebitda_musd",
    "ebitda_margin_pct", "northgate_share_pct", "solara_share_pct", "halcyon_share_pct", "top3_share_pct",
    "gross_churn_pct", "net_revenue_retention_pct", "logo_churn_count", "customers_total",
)
_FY_REVENUE = {2023: 334.0, 2024: PRIOR_REVENUE_MUSD, 2025: TOTAL_REVENUE_MUSD}   # FY2023 is synthetic
_FY_WEIGHTS = {2023: (0.240, 0.248, 0.253, 0.259), 2024: (0.238, 0.247, 0.254, 0.261), 2025: (0.236, 0.245, 0.255, 0.264)}
_SUB_SHARE = {2023: 0.768, 2024: 286.4 / 371.9, 2025: 318.9 / 412.6}      # subscription share of revenue
_SHARES = {  # customer share of revenue per quarter; FY2025 means are exactly 17.0 / 13.0 / 10.0 (top three = 40.0)
    "northgate": {2023: (11.5, 12.1, 12.8, 13.5), 2024: (14.2, 14.7, 15.2, 15.8), 2025: (16.4, 16.8, 17.2, 17.6)},
    "solara": {2023: (8.6, 9.1, 9.7, 10.3), 2024: (10.8, 11.2, 11.7, 12.2), 2025: (12.5, 12.8, 13.2, 13.5)},
    "halcyon": {2023: (6.3, 6.8, 7.3, 7.7), 2024: (8.0, 8.3, 8.7, 9.1), 2025: (9.6, 9.8, 10.1, 10.5)},
}


def build_financial_rows(seed: int = 20260101) -> List[Dict[str, Any]]:
    """Build 12 quarters (Q1 FY2023..Q4 FY2025) of ARR, EBITDA, customer concentration and churn.

    FY2024/FY2025 quarterly revenue sums exactly to the 10-K totals; ARR = quarterly subscription revenue x 4.
    """
    rng = random.Random(seed ^ 0xF1)  # separate stream: PDFs (and their doc ids) are unaffected
    rows: List[Dict[str, Any]] = []
    prev_arr: Optional[float] = None
    i = 0
    for year in (2023, 2024, 2025):
        revs = [round(_FY_REVENUE[year] * w, 1) for w in _FY_WEIGHTS[year][:3]]
        revs.append(round(_FY_REVENUE[year] - sum(revs), 1))
        for q in range(4):
            rev = revs[q]
            arr = round(rev * _SUB_SHARE[year] * 4, 1)
            prev_arr = round(arr * 0.975, 1) if prev_arr is None else prev_arr
            ebitda = round(rev * (9.5 + 8.0 * i / 11 + rng.uniform(-0.3, 0.3)) / 100, 1)
            gross = round(2.3 - 0.8 * i / 11 + rng.uniform(-0.15, 0.15), 2)
            customers = 410 + int(9.5 * i) + rng.randint(0, 3)
            n, s, h = (_SHARES[k][year][q] for k in ("northgate", "solara", "halcyon"))
            rows.append({
                "quarter": f"Q{q + 1} FY{year}", "period_end": f"{year}-{('03-31', '06-30', '09-30', '12-31')[q]}",
                "total_revenue_musd": rev, "arr_musd": arr, "net_new_arr_musd": round(arr - prev_arr, 1),
                "ebitda_musd": ebitda, "ebitda_margin_pct": round(ebitda / rev * 100, 1),
                "northgate_share_pct": n, "solara_share_pct": s, "halcyon_share_pct": h, "top3_share_pct": round(n + s + h, 1),
                "gross_churn_pct": gross, "net_revenue_retention_pct": round(104.0 + 7.5 * i / 11 + rng.uniform(-0.6, 0.6), 1),
                "logo_churn_count": max(1, round(customers * gross / 100 * 0.8)), "customers_total": customers,
            })
            prev_arr = arr
            i += 1
    return rows


def write_financials_csv(path: Path, seed: int = 20260101) -> Path:
    """Write ``demo_financials.csv`` (12 quarterly rows) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FINANCIAL_COLUMNS))
        writer.writeheader()
        writer.writerows(build_financial_rows(seed))
    return path


def validate_financials_csv(path: Path) -> List[Dict[str, str]]:
    """Return the rows of a financials CSV, or raise ``ValueError`` if columns/values do not match the schema."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in FINANCIAL_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("no data rows")
    for line, row in enumerate(rows, start=2):
        for col in FINANCIAL_COLUMNS[2:]:
            try:
                float(row[col])
            except (TypeError, ValueError):
                raise ValueError(f"row {line}: column '{col}' is not numeric") from None
    return rows


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Times-Bold", fontSize=16, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=13, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Times-BoldItalic", fontSize=11, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
        "center": ParagraphStyle("c", parent=base["BodyText"], fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=4),
    }


S = _styles()


def _footer(doc_id: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(0.9 * inch, 0.5 * inch, f"APEX DATA ROOM | STRICTLY CONFIDENTIAL | {doc_id} | Page {doc.page}")
        canvas.restoreState()

    return draw


def _build(path: Path, title: str, doc_id: str, story: List[Any]) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=LETTER, title=title, author=TARGET, invariant=1,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch, topMargin=0.9 * inch, bottomMargin=0.9 * inch,
    )
    footer = _footer(doc_id)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def _table(rows: List[List[str]], widths: List[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def build_10k(path: Path, rng: random.Random) -> None:
    """Create the 10-K excerpt (Items 1, 1A, 7) with the buried risk-factor disclosure."""
    sample = lambda k: rng.sample(GENERIC_RISK, k)  # noqa: E731
    s: List[Any] = [
        _p("UNITED STATES SECURITIES AND EXCHANGE COMMISSION", "center"),
        _p("Washington, D.C. 20549", "center"),
        _p("FORM 10-K", "title"),
        _p("ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934", "center"),
        _p("For the fiscal year ended December 31, 2025 | Commission File Number 001-39482", "center"),
        Spacer(1, 10),
        _p(f"<b>{TARGET.upper()}</b>", "center"),
        _p("(IRS Employer Identification No. 84-3917265)", "center"),
        _p("1200 Harborview Parkway, Suite 900, Austin, Texas 78701 | Telephone (512) 555-0147", "center"),
        _p("Data room copy. Excerpt of Items 1, 1A and 7. Not for onward distribution.", "center"),
        PageBreak(),
        _p("ITEM 1. BUSINESS", "h1"),
        _p(f"{TARGET} (the &quot;Company&quot;) provides the Helix Platform, a cloud-based supply chain analytics and "
           "orchestration suite used by global shippers, retailers and life-sciences distributors. Our platform "
           "processes more than 1.8 billion shipment events per month and is sold under multi-year subscription and "
           "services agreements. We are led by Dana R. Whitfield, our Chief Executive Officer, and Marcus Okonkwo-Reyes, "
           "our Chief Financial Officer."),
        _p("ITEM 1A. RISK FACTORS", "h1"),
        _p("Investing in our common stock involves a high degree of risk. You should carefully consider the risks "
           "below together with the other information in this report."),
        _p("Risks Related to Our Business", "h1"),
        _p("We have a history of net losses and may not sustain profitability", "h2"),
        _p(" ".join(["We reported net income of $38.4 million for fiscal 2025 but incurred losses in prior years."] + sample(3))),
        _p("Our platform is complex and may contain undetected defects", "h2"),
        _p(" ".join(["Errors in the Helix Platform could result in service interruptions and customer claims."] + sample(3))),
        _p("Our ability to compete depends on continued investment in research and development", "h2"),
        _p(" ".join(["We invested $61.2 million in research and development during fiscal 2025."] + sample(3))),
        _p("Risks Related to Customer Concentration", "h1"),
        _p("We depend on a limited number of customers for a substantial portion of our revenue", "h2"),
        _p("Our three largest customers (&quot;Customer A&quot;, &quot;Customer B&quot; and &quot;Customer C&quot;) accounted for "
           "17%, 13% and 10% of total revenue, respectively, for the year ended December 31, 2025. "
           + " ".join(sample(2))),
        _p("Our revenue growth depends on renewals of long-term agreements", "h2"),
        _p(" ".join(["Most of our subscription agreements have initial terms of three to five years."] + sample(3))),
        _p("Risks Related to Strategic Transactions and Our Securities", "h1"),
        _p("Strategic transactions may be complicated by the rights of our commercial counterparties", "h2"),
        _p(" ".join(sample(2)) +
           " Any acquisition of, or investment in, the Company would be subject to customary regulatory review, including "
           "under the Hart-Scott-Rodino Act, and may divert management attention. "
           "In addition, certain of our long-term commercial agreements, including our agreements with Customer A, Customer B "
           "and Customer C, which together represented approximately 40% of our total revenue in fiscal 2025, contain provisions "
           "that are triggered by a change of control of the Company, under which each such customer may elect to terminate its "
           "agreement without penalty following the consummation of such a transaction. "
           + " ".join(sample(3))),
        _p("The market price of our common stock may be volatile", "h2"),
        _p(" ".join(["Our stock price has fluctuated significantly since our initial public offering."] + sample(3))),
        _p("We do not intend to pay dividends in the foreseeable future", "h2"),
        _p(" ".join(sample(2))),
        PageBreak(),
        _p("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS (EXCERPT)", "h1"),
        _p(f"Total revenue was ${TOTAL_REVENUE_MUSD} million for fiscal 2025, compared with ${PRIOR_REVENUE_MUSD} million for fiscal 2024, "
           "an increase of 10.9%. The increase was driven by expansion within existing enterprise accounts."),
        _table([
            ["(in millions of USD)", "FY2025", "FY2024", "Change"],
            ["Subscription revenue", "$318.9", "$286.4", "11.3%"],
            ["Professional services", "$93.7", "$85.5", "9.6%"],
            ["Total revenue", f"${TOTAL_REVENUE_MUSD}", f"${PRIOR_REVENUE_MUSD}", "10.9%"],
        ], [2.6 * inch, 1.2 * inch, 1.2 * inch, 1.0 * inch]),
        Spacer(1, 10),
        _p("Liquidity. As of December 31, 2025 we held cash and cash equivalents of $126.5 million and had no borrowings "
           "outstanding under our $75.0 million revolving credit facility."),
        _p("SIGNATURES", "h1"),
        _p("/s/ Dana R. Whitfield, Chief Executive Officer (Principal Executive Officer)"),
        _p("/s/ Marcus Okonkwo-Reyes, Chief Financial Officer (Principal Financial Officer)"),
    ]
    _build(path, "Form 10-K FY2025 (Excerpt)", "DOC-10K-FY2025", s)


def build_msa(path: Path) -> None:
    """Create the Northgate Master Services Agreement; the clause is Section 16.2."""
    c = CUSTOMERS[0]
    sections = [
        ("Definitions", [
            "&quot;Affiliate&quot; means any entity controlling, controlled by or under common control with a party.",
            "&quot;Services&quot; means the analytics, orchestration and support services described in each Statement of Work.",
        ]),
        ("Services", ["Supplier shall perform the Services in a professional manner consistent with industry standards.",
                      "Customer may use the Services solely for its internal business operations."]),
        ("Statements of Work", ["Each Statement of Work shall identify deliverables, milestones, fees and acceptance criteria."]),
        ("Fees and Payment", [f"Customer shall pay an annual platform fee of $65,000,000 plus usage fees; total fees for fiscal 2025 were $70,100,000.",
                              "Undisputed invoices are payable within forty-five (45) days of receipt."]),
        ("Service Levels", ["Supplier shall maintain platform availability of 99.9% measured monthly, subject to service credits."]),
        ("Customer Data", ["As between the parties, Customer owns all Customer Data. Supplier shall process Customer Data only to provide the Services."]),
        ("Intellectual Property", ["Supplier retains all rights in the Helix Platform. Customer retains all rights in its pre-existing materials."]),
        ("Confidentiality", ["Each party shall protect the other's Confidential Information using at least reasonable care for five (5) years after disclosure."]),
        ("Representations and Warranties", ["Each party represents that it has full authority to enter into this Agreement."]),
        ("Indemnification", ["Supplier shall defend Customer against third-party claims alleging that the Services infringe intellectual property rights."]),
        ("Limitation of Liability", ["Neither party's aggregate liability shall exceed the fees paid in the twelve (12) months preceding the claim, except for indemnity and confidentiality breaches."]),
        ("Insurance", ["Supplier shall maintain commercial general liability and cyber liability insurance of not less than $10,000,000 per occurrence."]),
        ("Term", ["This Agreement commences on the Effective Date and continues for five (5) years, renewing for successive one-year terms unless either party gives ninety (90) days' notice."]),
        ("Termination for Cause", ["Either party may terminate this Agreement upon thirty (30) days' written notice of an uncured material breach."]),
        ("Effects of Termination", ["Upon termination Supplier shall return or delete Customer Data within sixty (60) days and Customer shall pay all accrued undisputed fees."]),
        ("Assignment; Successors", [
            "Neither party may assign this Agreement without the prior written consent of the other, not to be unreasonably withheld, except to an Affiliate.",
            "Notwithstanding Section 14 or Section 16.1, in the event of a Change of Control of Supplier, Customer may, by written notice delivered at any "
            "time within one hundred eighty (180) days following the consummation of such Change of Control, terminate this Agreement in whole or in part, "
            "effective on the date specified in such notice, without penalty, termination fee, liability, or obligation to repay unamortized implementation "
            "costs or any minimum purchase shortfall. " + COC_DEFINITION,
            "Subject to the foregoing, this Agreement binds and benefits the parties and their permitted successors and assigns.",
        ]),
        ("Force Majeure", ["Neither party is liable for delays caused by events beyond its reasonable control, excluding payment obligations."]),
        ("General Provisions", ["This Agreement is governed by the laws of the State of Delaware. Notices must be in writing and delivered to the addresses set out on the signature page.",
                                "This Agreement constitutes the entire agreement of the parties and may be amended only in a writing signed by both parties."]),
    ]
    assert sections[15][0].startswith("Assignment"), "Assignment section must be Section 16"
    s: List[Any] = [
        _p("MASTER SERVICES AGREEMENT", "title"),
        _p(f"Contract No. {c.contract_id}", "center"),
        _p(f"This Master Services Agreement (the &quot;Agreement&quot;) is entered into as of {c.effective} (the &quot;Effective Date&quot;) between "
           f"<b>{TARGET}</b> (&quot;Supplier&quot;), 1200 Harborview Parkway, Suite 900, Austin, Texas 78701, and <b>{c.name}</b> "
           "(&quot;Customer&quot;), 400 Commerce Drive, Memphis, Tennessee 38103."),
    ]
    for i, (title, clauses) in enumerate(sections, start=1):
        s.append(_p(f"{i}. {title}", "h2"))
        for j, clause in enumerate(clauses, start=1):
            s.append(_p(f"{i}.{j} {clause}"))
    s += [
        _p("EXHIBIT A: COMMERCIAL TERMS", "h1"),
        _table([
            ["Item", "Term"],
            ["Annual platform fee", "$65,000,000"],
            ["Fiscal 2025 total fees", "$70,100,000"],
            ["Share of Supplier FY2025 revenue", "17.0%"],
        ], [3.0 * inch, 3.0 * inch]),
        Spacer(1, 14),
        _p(f"<b>{TARGET}</b><br/>By: /s/ Priya Venkataraman, General Counsel"),
        _p(f"<b>{c.name}</b><br/>By: /s/ {c.signatory}"),
    ]
    _build(path, f"Master Services Agreement {c.contract_id}", f"DOC-{c.contract_id}", s)


def build_schedule(path: Path) -> None:
    """Create the material-customer schedule and Omnibus Amendment No. 2."""
    total_rev = round(sum(c.revenue_musd for c in CUSTOMERS), 1)
    total_pct = sum(c.share_pct for c in CUSTOMERS)
    rows = [["Customer", "Contract No.", "Effective", "FY2025 Revenue (USD M)", "% of Revenue"]]
    rows += [[c.name, c.contract_id, c.effective, f"${c.revenue_musd}", f"{c.share_pct:.1f}%"] for c in CUSTOMERS]
    rows.append(["Top three customers", "", "", f"${total_rev}", f"{total_pct:.1f}%"])
    s: List[Any] = [
        _p("SCHEDULE 3.12: MATERIAL CUSTOMER CONTRACTS", "title"),
        _p(f"Disclosure Schedule of {TARGET}, delivered in connection with the proposed acquisition of the Company."),
        _table(rows, [1.9 * inch, 1.2 * inch, 1.1 * inch, 1.3 * inch, 0.9 * inch]),
        Spacer(1, 10),
        _p(f"Total fiscal 2025 revenue was ${TOTAL_REVENUE_MUSD} million. The three customers listed above represent ${total_rev} million, "
           f"or {total_pct:.1f}% of total revenue."),
        _p("Each agreement listed above is a Master Services Agreement with an initial term of not less than five (5) years and is "
           "in full force and effect. No customer has delivered notice of non-renewal."),
        PageBreak(),
        _p("OMNIBUS AMENDMENT NO. 2 TO MASTER SERVICES AGREEMENTS", "title"),
        _p(f"This Omnibus Amendment No. 2 is made as of January 12, 2026 by {TARGET} and each customer listed in Schedule 3.12."),
        _p("1. Purpose", "h2"),
        _p("The parties wish to conform the assignment provisions of the Master Services Agreements listed in Schedule 3.12 "
           "(each, a &quot;Top Customer Agreement&quot;)."),
        _p("2. Amendment", "h2"),
        _p("2.1 Section 16 of each Top Customer Agreement is amended to include the following as Section 16.2: "
           "In the event of a Change of Control of Supplier, Customer may terminate this Agreement upon written notice delivered within "
           "one hundred eighty (180) days following the consummation thereof, without penalty, termination fee or further liability. "),
        _p("2.2 " + COC_DEFINITION),
        _p("3. Ratification", "h2"),
        _p("Except as expressly amended, each Top Customer Agreement remains in full force and effect."),
        _p("Executed by:", "h2"),
    ]
    s.append(_p(f"{TARGET}: /s/ Dana R. Whitfield, Chief Executive Officer"))
    for c in CUSTOMERS:
        s.append(_p(f"{c.name}: /s/ {c.signatory}"))
    _build(path, "Schedule 3.12 and Omnibus Amendment No. 2", "DOC-SCHED-312", s)


def verify_documents(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Re-open each PDF with PyPDF2 and confirm the hidden clause is extractable."""
    from PyPDF2 import PdfReader

    results: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        reader = PdfReader(str(path))
        hits = []
        for number, page in enumerate(reader.pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "")
            if re.search(r"change of control", text, re.I) and re.search(r"without penalty", text, re.I):
                hits.append(number)
        if not hits:
            raise RuntimeError(f"Hidden Change of Control clause missing from {path.name}")
        results[path.name] = {"pages": len(reader.pages), "clause_pages": hits}
    return results


def generate_all(out_dir: Path, seed: int = 20260101, csv_path: Optional[Path] = None) -> List[Path]:
    """Generate the three PDFs, ``demo_financials.csv`` and ``ground_truth.json``; return the PDF paths."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = [
        out_dir / "Meridian_Helix_10K_FY2025_Excerpt.pdf",
        out_dir / "MSA_Northgate_Logistics_MSA-2023-004817.pdf",
        out_dir / "Schedule_3-12_Customer_Contracts_and_Amendment.pdf",
    ]
    build_10k(paths[0], rng)
    build_msa(paths[1])
    build_schedule(paths[2])
    verification = verify_documents(paths)

    csv_file = write_financials_csv(csv_path or out_dir.parent / "demo_financials.csv", seed)
    fin = validate_financials_csv(csv_file)
    fy25 = fin[-4:]
    top3 = sum(float(r["top3_share_pct"]) for r in fy25) / 4
    rev25 = sum(float(r["total_revenue_musd"]) for r in fy25)
    if len(fin) != 12 or abs(top3 - 40.0) > 0.05 or abs(rev25 - TOTAL_REVENUE_MUSD) > 0.05:
        raise RuntimeError("financials CSV failed its consistency check against the PDFs")

    truth = {
        "target": TARGET,
        "clause": "CHANGE_OF_CONTROL",
        "effect": "Top three customers may terminate without penalty upon acquisition",
        "customers": [{"name": c.name, "share_pct": c.share_pct, "revenue_musd": c.revenue_musd, "contract_id": c.contract_id} for c in CUSTOMERS],
        "combined_share_pct": sum(c.share_pct for c in CUSTOMERS),
        "locations": {
            paths[0].name: "Item 1A, 'Strategic transactions may be complicated...' (buried mid-paragraph)",
            paths[1].name: "Section 16.2 Assignment; Successors",
            paths[2].name: "Omnibus Amendment No. 2, Section 2.1",
        },
        "financials_csv": {
            "file": csv_file.name, "rows": len(fin), "fy2025_top3_share_pct": round(top3, 1),
            "fy2025_northgate_share_pct": round(sum(float(r["northgate_share_pct"]) for r in fy25) / 4, 1),
            "fy2025_revenue_musd": round(rev25, 1),
        },
        "pdf_verification": verification,
        "generation_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic Apex data-room PDFs and demo_financials.csv.")
    parser.add_argument("--out", type=Path, default=ApexConfig().pdf_dir)
    args = parser.parse_args()
    for path in generate_all(args.out):
        print(f"wrote {path}")
    print(f"wrote {args.out.parent / 'demo_financials.csv'}")
    print(f"ground truth: {args.out / 'ground_truth.json'}")

if __name__ == "__main__":
    main()