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


def validate_bill_parameters(
    prev_reading: Optional[float],
    curr_reading: Optional[float],
    prev_date_str: Optional[str],
    curr_date_str: Optional[str],
    billed_units: Optional[float] = None,
    billed_amount: Optional[float] = None,
    meter_constant: float = 1.0,
    meter_replaced: bool = False,
    old_meter_final: Optional[float] = None,
    new_meter_initial: Optional[float] = 0.0,
    is_estimated: bool = False,
    max_reading: float = 999999.0,
) -> Tuple[bool, float, str, list, list]:
    """
    Deterministic input validation engine for domestic electricity billing.
    Returns:
        (is_valid, calculated_units, meter_status, validation_errors, validation_warnings)
    """
    errors = []
    warnings = []
    meter_status = "NORMAL"
    calc_units = 0.0

    # 1. Check missing values
    if prev_reading is None or curr_reading is None:
        errors.append("Meter reading values are missing. Both previous and current readings are required.")
        return False, 0.0, "INVALID", errors, warnings

    if not prev_date_str or not curr_date_str:
        errors.append("Meter reading dates are missing. Both previous and current dates are required.")
        return False, 0.0, "INVALID", errors, warnings

    # 2. Date parsing and ordering
    try:
        d1 = datetime.strptime(prev_date_str, "%Y-%m-%d")
        d2 = datetime.strptime(curr_date_str, "%Y-%m-%d")
        delta_days = (d2 - d1).days

        if delta_days < 0:
            errors.append(
                f"Reversed reading dates: Current reading date ({curr_date_str}) is earlier than "
                f"previous reading date ({prev_date_str})."
            )
        elif delta_days == 0:
            errors.append("Same-day reading: Billing period must span at least 1 day.")
    except ValueError:
        errors.append(f"Invalid date format (must be YYYY-MM-DD). Received: '{prev_date_str}' and '{curr_date_str}'.")

    # 3. Numeric bounds on readings
    if prev_reading < 0 or curr_reading < 0:
        errors.append("Negative meter reading is physically impossible.")
    if prev_reading > max_reading or curr_reading > max_reading:
        errors.append(f"Meter reading exceeds maximum physical register limit ({max_reading:.0f}).")

    if billed_units is not None and billed_units < 0:
        errors.append("Billed units cannot be negative.")
    if billed_amount is not None and billed_amount < 0:
        errors.append("Billed amount cannot be negative.")

    # 4. Meter readings difference / Rollover / Replacement
    if not errors:
        if meter_replaced:
            meter_status = "REPLACED"
            if old_meter_final is not None:
                units_old = max(0.0, old_meter_final - prev_reading)
                units_new = max(0.0, curr_reading - (new_meter_initial or 0.0))
                calc_units = round((units_old + units_new) * meter_constant, 2)
                warnings.append(
                    f"Meter replacement recorded: Old meter units ({units_old:.1f}) + "
                    f"New meter units ({units_new:.1f}) = {calc_units:.1f} kWh."
                )
            else:
                errors.append("Meter was marked as replaced, but old meter final reading was not provided.")
        elif curr_reading < prev_reading:
            # Check rollover
            rollover_5 = (100000 - prev_reading) + curr_reading
            rollover_6 = (1000000 - prev_reading) + curr_reading
            if 0 < rollover_5 <= 2500:
                calc_units = round(rollover_5 * meter_constant, 2)
                meter_status = "ROLLOVER"
                warnings.append(f"5-digit meter register rollover detected. Calculated units: {calc_units:.1f} kWh.")
            elif 0 < rollover_6 <= 2500:
                calc_units = round(rollover_6 * meter_constant, 2)
                meter_status = "ROLLOVER"
                warnings.append(f"6-digit meter register rollover detected. Calculated units: {calc_units:.1f} kWh.")
            else:
                meter_status = "INVERTED"
                errors.append(
                    f"Inverted meter reading: Current reading ({curr_reading}) is less than previous reading "
                    f"({prev_reading}), indicating meter rollback or clerical transcription error."
                )
        else:
            calc_units = round((curr_reading - prev_reading) * meter_constant, 2)
            meter_status = "NORMAL"

    # 5. Daily consumption sanity check
    if not errors:
        days = calculate_billing_days(prev_date_str, curr_date_str)
        daily_kwh = calc_units / max(1, days)
        if daily_kwh > 200.0:
            warnings.append(
                f"Extremely high daily consumption ({daily_kwh:.1f} kWh/day) for domestic low-tension connection. "
                "Physical load verification recommended."
            )

    # 6. Estimated reading warning
    if is_estimated:
        meter_status = "ESTIMATED"
        warnings.append(
            "Bill is marked as an Estimated/Assessed Reading. Actual consumption will be reconciled during next physical cycle."
        )

    is_valid = len(errors) == 0
    return is_valid, calc_units, meter_status, errors, warnings


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

