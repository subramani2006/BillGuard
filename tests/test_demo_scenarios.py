"""Focused test suite for the three predefined BillGuard demonstration scenarios."""

import pytest
import numpy as np
import joblib

from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from src.explanation.engine import ExplanationEngine
from src.ml.isolation_forest import BillIsolationForest


@pytest.fixture
def engine():
    return TariffEngine()


@pytest.fixture
def explanation_engine():
    return ExplanationEngine()


@pytest.fixture
def feature_extractor():
    return joblib.load("./models/feature_extractor.joblib")


@pytest.fixture
def isolation_forest():
    iso = BillIsolationForest()
    iso.load("./models/isolation_forest.joblib")
    return iso


def test_scenario_a_normal_bill(engine, explanation_engine, feature_extractor, isolation_forest):
    """
    Scenario A — Normal Bill
    Input: Consistent 200 kWh consumption matching past 200 kWh baseline.
    Expected Result: Normal — No anomaly detected.
    """
    bill_input = BillInput(
        consumer_number="09-245-001-042",
        previous_reading_date="2024-06-15",
        current_reading_date="2024-08-15",
        previous_meter_reading=1000.0,
        current_meter_reading=1200.0,
        billed_units=200.0,
        billed_amount=279.0,
        arrears_amount=0.0,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    t_result = engine.calculate_bill(bill_input)

    # 1. Deterministic validation
    assert t_result.is_valid is True
    assert t_result.calculated_units == 200.0
    assert abs(t_result.discrepancy_amount) <= 0.05
    assert t_result.is_arithmetic_exact_match is True

    # 2. Machine Learning Anomaly Screening
    hist_units = [190.0, 205.0, 195.0, 210.0, 200.0]
    hist_mean = float(np.mean(hist_units))
    hist_records = [{"recorded_units": u, "billing_days": 60} for u in hist_units]
    curr_record = {
        "recorded_units": t_result.calculated_units,
        "billed_units": 200.0,
        "billing_days": t_result.billing_days,
        "cycle_in_year": 3,
        "calculated_grand_total": t_result.calculated_grand_total,
        "discrepancy_amount": t_result.discrepancy_amount or 0.0,
        "arrears_amount": 0.0,
    }
    feat_vec = feature_extractor.extract_features_from_history(curr_record, hist_records)
    feat_vec_norm = feature_extractor.transform(feat_vec.reshape(1, -1))
    
    pred = isolation_forest.predict(feat_vec_norm)[0]
    score = isolation_forest.score_samples(feat_vec_norm)[0]

    assert pred == 0  # 0 indicates normal, 1 indicates anomaly
    assert score < 0.50  # Below anomaly threshold

    # 3. Investigation Report
    report = explanation_engine.generate_explanation(
        tariff_result=t_result,
        ml_score=float(score),
        ml_predicted=bool(pred == 1),
        historical_mean_units=hist_mean,
        previous_bill_amount=279.0,
        previous_meter_reading=1000.0,
        current_meter_reading=1200.0,
    )

    assert report.is_anomaly is False
    assert "No discrepancy detected" in report.outcome


def test_scenario_b_unusual_bill(engine, explanation_engine, feature_extractor, isolation_forest):
    """
    Scenario B — Unusual Bill
    Input: Consumption surges from 120 kWh baseline to 420 kWh (+250%).
    Expected Result: Anomaly detected — Unusual consumption increase.
    """
    bill_input = BillInput(
        consumer_number="09-245-001-042",
        previous_reading_date="2024-06-15",
        current_reading_date="2024-08-15",
        previous_meter_reading=2000.0,
        current_meter_reading=2420.0,
        billed_units=420.0,
        billed_amount=1467.40,
        arrears_amount=0.0,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    t_result = engine.calculate_bill(bill_input)

    # 1. Deterministic validation - Input itself is valid and arithmetic matches
    assert t_result.is_valid is True
    assert t_result.calculated_units == 420.0
    assert abs(t_result.discrepancy_amount) <= 0.05

    # 2. Machine Learning Anomaly Screening detects unusual surge
    hist_units = [120.0, 115.0, 125.0, 118.0, 122.0]
    hist_mean = float(np.mean(hist_units))
    hist_records = [{"recorded_units": u, "billing_days": 60} for u in hist_units]
    curr_record = {
        "recorded_units": t_result.calculated_units,
        "billed_units": 420.0,
        "billing_days": t_result.billing_days,
        "cycle_in_year": 3,
        "calculated_grand_total": t_result.calculated_grand_total,
        "discrepancy_amount": t_result.discrepancy_amount or 0.0,
        "arrears_amount": 0.0,
    }
    feat_vec = feature_extractor.extract_features_from_history(curr_record, hist_records)
    feat_vec_norm = feature_extractor.transform(feat_vec.reshape(1, -1))

    pred = isolation_forest.predict(feat_vec_norm)[0]
    score = isolation_forest.score_samples(feat_vec_norm)[0]

    assert pred == 1  # 1 indicates detected anomaly
    assert score >= 0.50  # Meets or exceeds anomaly threshold

    # 3. Investigation Report
    report = explanation_engine.generate_explanation(
        tariff_result=t_result,
        ml_score=float(score),
        ml_predicted=bool(pred == 1),
        historical_mean_units=hist_mean,
        previous_bill_amount=100.0,
        previous_meter_reading=2000.0,
        current_meter_reading=2420.0,
    )

    assert report.is_anomaly is True
    assert "Unusual bill requiring investigation" in report.outcome


def test_scenario_c_invalid_bill_data(engine):
    """
    Scenario C — Invalid Bill Data
    Input: Current meter reading is less than previous meter reading without replacement flag.
    Expected Result: Invalid data — Please correct the highlighted field.
    The detector must NOT run on invalid input.
    """
    bill_input = BillInput(
        consumer_number="09-245-001-042",
        previous_reading_date="2024-06-15",
        current_reading_date="2024-08-15",
        previous_meter_reading=4850.0,
        current_meter_reading=4200.0,  # INVERTED: 4200 < 4850
        billed_units=0.0,
        billed_amount=0.0,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    t_result = engine.calculate_bill(bill_input)

    # Deterministic validation stops corrupted data
    assert t_result.is_valid is False
    assert len(t_result.validation_errors) > 0
    assert any("Inverted meter reading" in err or "Current reading" in err for err in t_result.validation_errors)
