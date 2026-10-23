"""Unit tests for the Tamil Nadu Tariff Engine & Verification Rules."""

import pytest
from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from src.tariff.rules import (
    calculate_billing_days,
    check_meter_reading_validity,
    evaluate_billing_cycle_duration,
)


@pytest.fixture
def engine():
    return TariffEngine()


def test_lifeline_100_units_free(engine):
    """Under Tamil Nadu domestic tariff, first 100 units should have 0 net energy charge."""
    b = BillInput(
        consumer_number="01-100-001",
        previous_meter_reading=500.0,
        current_meter_reading=600.0,  # exactly 100 units
        billed_amount=0.0,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = engine.calculate_bill(b)
    assert res.calculated_units == 100.0
    assert res.energy_charge_net == 0.0
    assert res.fppca_amount == 0.0
    assert res.calculated_grand_total == 0.0
    assert res.is_arithmetic_exact_match is True
    assert res.discrepancy_category == "EXACT_MATCH"


def test_tier_2_calculation(engine):
    """Test 200 units consumption (100 free + 100 @ 2.47 under 2024 tariff)."""
    b = BillInput(
        consumer_number="01-100-002",
        previous_meter_reading=1000.0,
        current_meter_reading=1200.0,  # 200 units
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = engine.calculate_bill(b)
    assert res.calculated_units == 200.0
    # 0-100 free, 101-200 @ 2.47 = 247.0
    assert res.energy_charge_net == 247.0
    # FPPCA on 100 billable units: 100 * 0.32 = 32.0
    assert res.fppca_amount == 32.0
    assert res.calculated_grand_total == round(247.0 + 32.0, 2)


def test_overbilled_detection(engine):
    """Detect when billed amount exceeds calculated grand total."""
    b = BillInput(
        consumer_number="01-100-003",
        previous_meter_reading=2000.0,
        current_meter_reading=2250.0,  # 250 units
        billed_amount=950.0,  # Unfairly overbilled
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = engine.calculate_bill(b)
    assert res.discrepancy_category == "OVERBILLED"
    assert res.discrepancy_amount > 0
    assert res.is_arithmetic_exact_match is False


def test_arrears_addition(engine):
    """Ensure arrears are correctly segregated and added to grand total."""
    b = BillInput(
        consumer_number="01-100-004",
        previous_meter_reading=1000.0,
        current_meter_reading=1100.0,  # 100 units = 0 current charge
        arrears_amount=350.50,
        billed_amount=350.50,
        tariff_code="TN_DOMESTIC_LT1A_2024",
    )
    res = engine.calculate_bill(b)
    assert res.calculated_current_cycle_total == 0.0
    assert res.arrears_amount == 350.50
    assert res.calculated_grand_total == 350.50
    assert res.is_arithmetic_exact_match is True


def test_meter_reading_rollover():
    """Verify detection of physical meter register rollover."""
    is_valid, units, msg = check_meter_reading_validity(
        prev_reading=99920.0,
        curr_reading=80.0,  # 5-digit rollover from 99999 to 00000 + 80
        meter_constant=1.0
    )
    assert is_valid is True
    assert units == 160.0
    assert "rollover" in msg.lower()


def test_meter_inverted_reading():
    """Detect inverted or tampered meter reading."""
    is_valid, units, msg = check_meter_reading_validity(
        prev_reading=1500.0,
        curr_reading=1200.0,
        meter_constant=1.0
    )
    assert is_valid is False
    assert "inverted reading" in msg.lower()


def test_billing_cycle_proration_warning():
    """Warn when billing cycle deviates from standard 60 days."""
    is_abnormal, notes = evaluate_billing_cycle_duration(75, standard_days=60)
    assert is_abnormal is True
    assert "extended billing cycle" in notes.lower()

    is_normal, notes2 = evaluate_billing_cycle_duration(61, standard_days=60)
    assert is_normal is False
