"""Consumer Grievance / Formal Representation Generator for TANGEDCO electricity bills."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from src.tariff.models import TariffCalculationResult
from .engine import ExplanationReport


class GrievanceGenerator:
    """
    Generates a formal, legally respectful complaint/representation petition for submission
    to TANGEDCO Section Offices and Consumer Grievance Redressal Forums (CGRF).
    """

    def generate_complaint_text(
        self,
        consumer_number: str,
        consumer_name: str,
        section_office: str,
        tariff_result: TariffCalculationResult,
        explanation: ExplanationReport,
        contact_phone: str = "[Consumer Phone]",
    ) -> str:
        """Generate structured formal petition letter as text."""
        today = datetime.now().strftime("%d-%B-%Y")
        discrepancy = tariff_result.discrepancy_amount or 0.0

        text = f"""FORMAL REPRESENTATION REGARDING ELECTRICITY BILL DISCREPANCY

Date: {today}
To:
The Assistant Executive Engineer (AEE) / Section Officer,
Operation & Maintenance (O&M),
Tamil Nadu Generation and Distribution Corporation Ltd. (TANGEDCO),
Section Office: {section_office}, Tamil Nadu.

From:
Name: {consumer_name}
LT Domestic Service Connection No: {consumer_number}
Tariff Category: Low Tension Tariff I-A (Domestic)
Contact: {contact_phone}

SUBJECT: Representation regarding Tariff Verification and Audit Discrepancy for Service Connection No: {consumer_number}

Respected Sir / Madam,

I am a domestic consumer holding Low Tension (LT) Service Connection Number {consumer_number} under your jurisdiction. I am writing to respectfully bring to your notice a calculation/recording discrepancy identified upon mathematical verification of my latest electricity bill card.

1. EXECUTIVE SUMMARY OF DISPUTE:
Outcome: {explanation.outcome}
Finding: {explanation.finding_title}
Verification Certainty: {explanation.certainty_level}
Summary: {explanation.english_summary}

2. AUDIT VERIFICATION DATA TABLE:
- Previous Meter Reading: {explanation.previous_meter_reading:.1f}
- Current Meter Reading: {explanation.current_meter_reading:.1f}
- Billing Days: {tariff_result.billing_days} days
- Calculated Units: {tariff_result.calculated_units:.1f} kWh
- Billed Units: {tariff_result.billed_units if tariff_result.billed_units is not None else 'N/A'} kWh
- Statutory Net Energy Charge: ₹{tariff_result.energy_charge_net:.2f}
- Government of Tamil Nadu Subsidy Applied: ₹{tariff_result.subsidy_amount:.2f} (100 units free)
- Fuel Surcharge (FPPCA): ₹{tariff_result.fppca_amount:.2f}
- Past Arrears Assessed: ₹{tariff_result.arrears_amount:.2f}
----------------------------------------------------------------------
- Total Statutory Payable Amount (TNERC Order): ₹{tariff_result.calculated_grand_total:.2f}
- Billed Total Amount on Receipt: ₹{tariff_result.billed_amount if tariff_result.billed_amount is not None else 0.0:.2f}
- Net Audit Discrepancy: ₹{discrepancy:+.2f}
----------------------------------------------------------------------

3. EVIDENCE AND REASONS:
"""
        for pt in explanation.evidence_points_en:
            text += f"- {pt}\n"

        text += f"""
4. LEGAL & REGULATORY REFERENCES:
- TNERC Tariff Order ({tariff_result.tariff_name})
- Clause 11 of Tamil Nadu Electricity Distribution Code (Rectification of Accounts)
- Central Electricity Authority (Installation and Operation of Meters) Regulations (Rule 57)

5. PRAYER / RELIEF REQUESTED:
In light of the above verifiable calculation facts, I respectfully request your office to:
1. Re-audit and revise the latest assessment in accordance with statutory TNERC slab rules.
2. In the event of an overcharge of ₹{abs(discrepancy):.2f}, kindly credit the excess sum toward my forthcoming billing cycle.
3. If an uncharacteristic surge is recorded on the meter, kindly order a field inspection and test meter verification pursuant to CEA regulations without initiating disconnection.

Pending rectification, I remain committed to paying the undisputed statutory charge under respectful protest.

Thanking you.

Yours faithfully,

({consumer_name})
Consumer No: {consumer_number}
"""
        return text

    def generate_complaint_pdf(
        self,
        output_filepath: str,
        consumer_number: str,
        consumer_name: str,
        section_office: str,
        tariff_result: TariffCalculationResult,
        explanation: ExplanationReport,
        contact_phone: str = "[Consumer Phone]",
    ) -> str:
        """Render the formal representation into a publication-quality PDF."""
        p = Path(output_filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(p), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=13, alignment=1, textColor=colors.HexColor('#0A2240'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#555555'), spaceAfter=12)
        body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9.5, leading=13, spaceAfter=8)
        bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontSize=9.5, leading=13, fontName='Helvetica-Bold')

        elements = []

        # Title
        elements.append(Paragraph("TANGEDCO FORMAL CONSUMER REPRESENTATION / GRIEVANCE PETITION", title_style))
        elements.append(Paragraph("Generated via BillGuard Explainable MLOps Verification Audit", sub_style))
        elements.append(Spacer(1, 8))

        # Metadata Table
        today = datetime.now().strftime("%d-%B-%Y")
        meta_data = [
            [Paragraph("<b>Date:</b>", body_style), Paragraph(today, body_style), Paragraph("<b>To:</b>", body_style), Paragraph(f"Assistant Executive Engineer, TANGEDCO<br/>Section: {section_office}", body_style)],
            [Paragraph("<b>Consumer Name:</b>", body_style), Paragraph(consumer_name, body_style), Paragraph("<b>Service No:</b>", body_style), Paragraph(consumer_number, body_style)],
            [Paragraph("<b>Tariff Category:</b>", body_style), Paragraph("Domestic (LT Tariff I-A)", body_style), Paragraph("<b>Outcome:</b>", body_style), Paragraph(f"<b>{explanation.finding_title}</b>", body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[90, 160, 80, 210])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F6F9')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # Subject & Summary
        elements.append(Paragraph(f"<b>SUBJECT:</b> Dispute and Request for Audit Revision of Electricity Bill (Service No: {consumer_number})", bold_style))
        elements.append(Paragraph(f"<b>Summary:</b> {explanation.english_summary}", body_style))
        elements.append(Spacer(1, 8))

        # Audit Breakdown Table
        elements.append(Paragraph("<b>Itemized Statutory Tariff Audit Table:</b>", bold_style))
        audit_data = [
            ["Assessment Item", "Statutory Rule / Rate", "Calculated Value", "Bill Card Record"],
            ["Consumed Units", "Meter reading difference", f"{tariff_result.calculated_units:.1f} kWh", f"{tariff_result.billed_units:.1f} kWh" if tariff_result.billed_units else "N/A"],
            ["GoTN Subsidy (0-100 U)", "100% Free allowance", f"₹{tariff_result.subsidy_amount:.2f}", "Subsidized"],
            ["Net Energy Charges", f"{tariff_result.tariff_code}", f"₹{tariff_result.energy_charge_net:.2f}", "-"],
            ["FPPCA / Surcharges", "Fuel adjustment rate", f"₹{tariff_result.fppca_amount:.2f}", "-"],
            ["Assessed Arrears", "Past dues", f"₹{tariff_result.arrears_amount:.2f}", f"₹{tariff_result.arrears_amount:.2f}"],
            ["Grand Payable Total", "Total statutory obligation", f"₹{tariff_result.calculated_grand_total:.2f}", f"₹{tariff_result.billed_amount:.2f}" if tariff_result.billed_amount else "N/A"],
            ["Audit Discrepancy", "Net difference", f"₹{tariff_result.discrepancy_amount or 0.0:+.2f}", explanation.certainty_level],
        ]
        audit_table = Table(audit_data, colWidths=[150, 130, 130, 130])
        audit_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0A2240')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (2,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('BACKGROUND', (0,1), (-1,1), colors.white),
            ('BACKGROUND', (0,-2), (-1,-1), colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        elements.append(audit_table)
        elements.append(Spacer(1, 10))

        # Evidence
        elements.append(Paragraph("<b>Audit Evidence Points:</b>", bold_style))
        for ev in explanation.evidence_points_en:
            elements.append(Paragraph(f"• {ev}", body_style))

        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>Recommended Action / Relief Requested:</b><br/>{explanation.recommended_action_en}", body_style))
        elements.append(Spacer(1, 14))

        # Signoff
        elements.append(Paragraph(f"Respectfully submitted,<br/><br/><b>{consumer_name}</b><br/>Consumer Service Connection No: {consumer_number}", body_style))

        doc.build(elements)
        return str(p)
