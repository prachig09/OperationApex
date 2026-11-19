import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_demo_pdf(output_filename="MSA_Northgate_Logistics_MSA-2023-004817.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A2B4C"),
        alignment=0,
        spaceAfter=10
    )
    
    header_style = ParagraphStyle(
        'HeaderMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#D9534F")
    )
    
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1A2B4C"),
        spaceBefore=14,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2C3E50"),
        spaceAfter=8
    )
    
    highlight_style = ParagraphStyle(
        'HighlightBody',
        parent=body_style,
        fontName='Helvetica',
        textColor=colors.HexColor("#000000")
    )

    story = []

    # Confidential Header Banner
    story.append(Paragraph("STRICTLY CONFIDENTIAL // SUBJECT TO NON-DISCLOSURE AGREEMENT", header_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceAfter=15))

    # Title
    story.append(Paragraph("MASTER SERVICES AGREEMENT", title_style))
    story.append(Paragraph("<b>Agreement ID:</b> MSA-2023-004817<br/><b>Effective Date:</b> January 15, 2023", body_style))
    story.append(Spacer(1, 10))

    # Parties Metadata Table
    data = [
        [Paragraph("<b>Provider / Target Entity:</b>", body_style), Paragraph("Stealth Tech Solutions Inc.", body_style)],
        [Paragraph("<b>Enterprise Client Name:</b>", body_style), Paragraph("Northgate Logistics Global LLC", body_style)],
        [Paragraph("<b>Contracted Annual Value (ARR):</b>", body_style), Paragraph("$20,000,000 USD (40% Total ARR)", body_style)],
        [Paragraph("<b>Governing Jurisdiction:</b>", body_style), Paragraph("State of Delaware, United States", body_style)]
    ]
    t = Table(data, colWidths=[200, 304])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Section 1: Services Scope
    story.append(Paragraph("1. Scope of Cloud Infrastructure Services", h2_style))
    story.append(Paragraph(
        "Provider agrees to supply Client with dedicated edge container management platforms, automated orchestration services, "
        "and 24/7 technical monitoring as outlined in Schedule A (Service Level Agreement). Client agrees to maintain primary database "
        "routing through Provider's cloud endpoints for the duration of the Initial Term.",
        body_style
    ))

    # Section 2: Financial Terms & Commitments
    story.append(Paragraph("2. Financial Commitments and Subscription Fees", h2_style))
    story.append(Paragraph(
        "Client shall pay Provider an annual subscription fee of $20,000,000 USD, billed quarterly in advance. "
        "All invoices are subject to Net-30 payment terms from invoice receipt. Late payments beyond sixty (60) days shall accrue "
        "interest at the rate of 1.5% per month or the maximum statutory rate permitted under Delaware law.",
        body_style
    ))

    # Section 12: Termination & Change of Control (CRITICAL FINDING)
    story.append(Paragraph("12. Term, Termination, and Assignment", h2_style))
    story.append(Paragraph(
        "12.1 <b>Initial Term:</b> This Agreement commences on the Effective Date and continues for a period of five (5) consecutive years, "
        "unless terminated earlier pursuant to Section 12.2 or Section 12.3.",
        body_style
    ))
    story.append(Paragraph(
        "12.2 <b>Termination for Cause:</b> Either party may terminate this Agreement immediately upon written notice if the other party "
        "commits a material breach of contract and fails to cure such breach within thirty (30) business days.",
        body_style
    ))
    
    # Hidden Clause targeted by RAG Extractor
    story.append(Paragraph(
        "12.3 <b>Special Condition — Change of Control:</b> In the event of a Direct Merger, Acquisition, Change of Control, or sale of "
        "substantially all equity assets of Provider (Stealth Tech Solutions Inc.) to a third-party enterprise entity (including but not "
        "limited to Acme Corp), Client (Northgate Logistics Global LLC) retains the absolute unilateral right to terminate this Agreement "
        "without penalty, liquidating all remaining ARR commitments upon providing thirty (30) days prior written notice.",
        highlight_style
    ))

    # Signatures
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=15))
    story.append(Paragraph("<b>EXECUTED BY AUTHORIZED REPRESENTATIVES:</b>", body_style))
    story.append(Spacer(1, 10))
    
    sig_data = [
        [Paragraph("<b>For Stealth Tech Solutions Inc.</b>", body_style), Paragraph("<b>For Northgate Logistics Global LLC</b>", body_style)],
        [Paragraph("<br/><br/>___________________________________<br/>Chief Executive Officer", body_style),
         Paragraph("<br/><br/>___________________________________<br/>VP of Enterprise Procurement", body_style)]
    ]
    sig_table = Table(sig_data, colWidths=[252, 252])
    sig_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(sig_table)

    doc.build(story)
    print(f"Generated PDF demo document: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    generate_demo_pdf()