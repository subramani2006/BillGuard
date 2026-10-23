"""Tariff rules, proration logic, and validation checks."""

from datetime import datetime
from typing import Tuple, Optional


def calculate_billing_days(start_date_str: str, end_date_str: str) -> int:
    """Calculate the number of days between two billing dates (YYYY-MM-DD)."""
    try:
        d1 = datetime.strptime(start_date_str, "%Y-%m-%d")
        d2 = datetime.strptime(end_date_str, "%Y-%m-%d")
        delta = (d2 - d1).days
        return max(1, delta)
    except Exception:
        # Fallback to standard 60 days if date parsing fails
        return 60


def check_meter_reading_validity(
    prev_reading: float,
    curr_reading: float,
    meter_constant: float = 1.0,
    max_reasonable_meter_value: float = 999999.0
) -> Tuple[bool, float, str]:
    """
    Validate meter readings deterministically:
    1. Readings must be non-negative numbers.
    2. Current reading should generally be >= Previous reading (unless rollover occurred).
    3. Calculate difference multiplied by meter constant.
    """
    if prev_reading < 0 or curr_reading < 0:
        return False, 0.0, "Negative meter reading is physically invalid."

    if curr_reading > max_reasonable_meter_value or prev_reading > max_reasonable_meter_value:
        return False, 0.0, f"Meter reading exceeds normal maximum register ({max_reasonable_meter_value})."

    if curr_reading < prev_reading:
        # Check potential 5-digit or 6-digit meter rollover (e.g., from 99990 to 00050)
        potential_rollover_5 = (100000 - prev_reading) + curr_reading
        potential_rollover_6 = (1000000 - prev_reading) + curr_reading

        if 0 < potential_rollover_5 <= 2000:
            units = potential_rollover_5 * meter_constant
            return True, units, f"Detected probable 5-digit meter rollover. Calculated units: {units:.1f}"
        elif 0 < potential_rollover_6 <= 2000:
            units = potential_rollover_6 * meter_constant
            return True, units, f"Detected probable 6-digit meter rollover. Calculated units: {units:.1f}"
        else:
            return False, 0.0, (
                f"Current meter reading ({curr_reading}) is less than previous reading ({prev_reading}), "
                "indicating inverted reading or meter rollback."
            )

    units = (curr_reading - prev_reading) * meter_constant
    return True, round(units, 2), "Meter reading sequence is valid."


def evaluate_billing_cycle_duration(
    actual_days: int,
    standard_days: int = 60,
    tolerance_days: int = 5
) -> Tuple[bool, str]:
    """
    Evaluate if billing cycle deviates from standard bi-monthly (60 days).
    Delayed billing cycles can push consumers into higher tariff slabs unfairly.
    """
    diff = actual_days - standard_days
    if abs(diff) <= tolerance_days:
        return False, f"Standard billing cycle duration ({actual_days} days, normal range: {standard_days - tolerance_days} - {standard_days + tolerance_days} days)."

    if diff > tolerance_days:
        return True, (
            f"Extended billing cycle detected: {actual_days} days instead of {standard_days} days (+{diff} days). "
            "Without slab proration, an extended cycle can artificially push consumption into more expensive tariff slabs."
        )
    else:
        return True, (
            f"Shortened billing cycle detected: {actual_days} days instead of {standard_days} days ({diff} days). "
            "Consumption was recorded over a shorter time window than usual."
        )
