"""Unit tests for synthetic data generation and consumer archetypes."""

import pytest
import pandas as pd
from src.data_gen.bill_generator import SyntheticBillGenerator
from src.data_gen.ocr_image_generator import BillImageGenerator
from src.data_gen.scenario_definitions import CONSUMER_ARCHETYPES, BIMONTHLY_SEASONAL_FACTORS


@pytest.fixture
def bill_gen():
    return SyntheticBillGenerator(seed=123)


def test_consumer_history_length(bill_gen):
    records = bill_gen.generate_consumer_history(
        consumer_id="09-100-001",
        archetype_key="MODERATE",
        num_cycles=6,
        anomaly_probability=0.0,
    )
    assert len(records) == 6
    assert records[0]["cycle_idx"] == 0
    assert records[5]["cycle_idx"] == 5


def test_seasonal_summer_peak(bill_gen):
    """High AC consumers must exhibit higher consumption in May-June (cycle 3) than Nov-Dec (cycle 6)."""
    records = bill_gen.generate_consumer_history(
        consumer_id="09-100-002",
        archetype_key="HIGH_AC",
        num_cycles=6,
        anomaly_probability=0.0,
    )
    cycle_3_units = [r["recorded_units"] for r in records if r["cycle_in_year"] == 3][0]
    cycle_6_units = [r["recorded_units"] for r in records if r["cycle_in_year"] == 6][0]
    assert cycle_3_units > cycle_6_units * 1.3


def test_isolated_splits_no_leakage(bill_gen):
    """Ensure train, validation, and test sets share zero consumers."""
    train_df, val_df, test_df = bill_gen.generate_full_dataset(
        num_consumers=30,
        cycles_per_consumer=4,
    )
    train_c = set(train_df["consumer_id"])
    val_c = set(val_df["consumer_id"])
    test_c = set(test_df["consumer_id"])

    assert len(train_c.intersection(val_c)) == 0
    assert len(train_c.intersection(test_c)) == 0
    assert len(val_c.intersection(test_c)) == 0


def test_bill_image_generator_output():
    img_gen = BillImageGenerator(seed=42)
    img, boxes = img_gen.generate_full_bill_image(
        consumer_no="09-245-001",
        consumer_name="T. SELVARAJ",
        bill_date="2024-08-15",
        prev_date="2024-06-15",
        prev_reading=4500.0,
        curr_reading=4850.0,
        units=350.0,
        net_amount=1250.0,
        arrears=0.0,
    )
    assert img.size == (600, 750)
    assert "consumer_number" in boxes
    assert "billed_amount" in boxes
    assert "billed_units" in boxes


def test_historical_records_mathematical_consistency(bill_gen):
    """Verify that every generated record satisfies discrepancy = billed_amount - calculated_grand_total."""
    records = bill_gen.generate_consumer_history(
        consumer_id="09-100-TEST",
        archetype_key="MODERATE",
        num_cycles=8,
        anomaly_probability=0.35,
    )
    for r in records:
        expected_disc = round(r["billed_amount"] - r["calculated_grand_total"], 2)
        assert abs(r["discrepancy_amount"] - expected_disc) < 1e-5
        assert r["current_reading_date"] > r["previous_reading_date"]
        assert r["recorded_units"] >= 0.0

