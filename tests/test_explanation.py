"""Unit tests for bilingual explanation engine and grievance petition generation."""

import os
from pathlib import Path
import pytest

from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from src.explanation.engine import ExplanationEngine
from src.explanation.grievance_generator import GrievanceGenerator


@pytest.fixture
def explanation_setup():
    tariff_engine = TariffEngine()
    exp_engine = ExplanationEngine()
    grievance_gen = GrievanceGenerator()
    return tariff_engine, exp_engine, grievance_gen


def test_arithmetic_error_explanation(explanation_setup):
    tariff_engine, exp_engine, _ = explanation_setup

    b = BillInput(
        consumer_number="09-245-001",
        previous_meter_reading=1000.0,
        current_meter_reading=1250.0,
        billed_amount=950.0,  # Overbilled
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = tariff_engine.calculate_bill(b)
    report = exp_engine.generate_explanation(res, ml_score=0.2)

    assert report.is_anomaly is True
    assert report.category == "CONFIRMED_ARITHMETIC_ERROR"
    assert report.certainty_level == "HIGH_CERTAINTY"
    assert "CONFIRMED ARITHMETIC DISCREPANCY" in report.english_summary
    assert "கட்டணக் கணக்கீட்டுப் பிழை" in report.tamil_summary
    assert len(report.evidence_points_en) >= 2


def test_extended_billing_cycle_explanation(explanation_setup):
    tariff_engine, exp_engine, _ = explanation_setup

    b = BillInput(
        consumer_number="09-245-002",
        previous_reading_date="2024-05-01",
        current_reading_date="2024-07-20",  # 80 days
        previous_meter_reading=1000.0,
        current_meter_reading=1400.0,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = tariff_engine.calculate_bill(b)
    b.billed_amount = res.calculated_grand_total
    res = tariff_engine.calculate_bill(b)
    report = exp_engine.generate_explanation(res)

    assert report.is_anomaly is True
    assert report.category == "SUSPECTED_EXTENDED_CYCLE"
    assert "BILLING CYCLE DURATION IRREGULARITY" in report.english_summary
    assert "கணக்கீட்டுக் கால நீட்டிப்பு" in report.tamil_summary


def test_normal_bill_explanation(explanation_setup):
    tariff_engine, exp_engine, _ = explanation_setup

    b = BillInput(
        consumer_number="09-245-003",
        previous_reading_date="2024-06-01",
        current_reading_date="2024-08-01",  # 61 days
        previous_meter_reading=2000.0,
        current_meter_reading=2200.0,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = tariff_engine.calculate_bill(b)
    # Match billed amount exactly to grand total
    b.billed_amount = res.calculated_grand_total
    res = tariff_engine.calculate_bill(b)

    report = exp_engine.generate_explanation(res, historical_mean_units=200.0)
    assert report.is_anomaly is False
    assert report.category == "NORMAL_VERIFIED"
    assert "BILL VERIFIED NORMAL" in report.english_summary
    assert "ரசீது சரிபார்க்கப்பட்டது" in report.tamil_summary


def test_grievance_text_and_pdf_generation(explanation_setup, tmp_path):
    tariff_engine, exp_engine, grievance_gen = explanation_setup

    b = BillInput(
        consumer_number="09-245-999",
        consumer_name="M. Selvakumar",
        previous_meter_reading=3000.0,
        current_meter_reading=3300.0,
        billed_amount=1500.0,  # Overbilled
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = tariff_engine.calculate_bill(b)
    report = exp_engine.generate_explanation(res)

    text = grievance_gen.generate_complaint_text(
        consumer_number="09-245-999",
        consumer_name="M. Selvakumar",
        section_office="Anna Nagar West",
        tariff_result=res,
        explanation=report,
    )
    assert "FORMAL REPRESENTATION REGARDING ELECTRICITY BILL DISCREPANCY" in text
    assert "M. Selvakumar" in text
    assert "09-245-999" in text

    pdf_out = tmp_path / "test_complaint.pdf"
    pdf_path = grievance_gen.generate_complaint_pdf(
        output_filepath=str(pdf_out),
        consumer_number="09-245-999",
        consumer_name="M. Selvakumar",
        section_office="Anna Nagar West",
        tariff_result=res,
        explanation=report,
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000
