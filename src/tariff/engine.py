"""Deterministic Tamil Nadu Tariff Calculation & Verification Engine."""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
try:
    import yaml
except ImportError:
    yaml = None

from .models import (
    TariffConfig,
    TariffSlab,
    FixedChargesConfig,
    ElectricityTaxConfig,
    FppcaConfig,
    SlabBreakdown,
    BillInput,
    TariffCalculationResult,
)
from .rules import (
    calculate_billing_days,
    check_meter_reading_validity,
    evaluate_billing_cycle_duration,
)


class TariffEngine:
    """
    Deterministic tariff verification engine for Indian domestic electricity bills.
    Accurately reproduces Tamil Nadu TANGEDCO LT-1A slabs, subsidies, FPPCA, and fixed charges.
    """

    def __init__(self, tariff_dir: Optional[str] = None):
        self.tariff_dir = Path(tariff_dir) if tariff_dir else Path(__file__).resolve().parent.parent.parent / "config" / "tariffs"
        self._tariffs: Dict[str, TariffConfig] = {}
        self.load_all_tariffs()

    def _parse_yaml_file(self, filepath: Path) -> dict:
        """Parse YAML file with PyYAML if available, or safe fallback parser."""
        content = filepath.read_text(encoding="utf-8")
        if yaml is not None:
            return yaml.safe_load(content)

        # Basic fallback parser if pyyaml is not yet loaded
        # Since our tariff yaml files are well-structured, we can parse standard key-values
        import re
        result = {}
        current_section = None
        current_list = None
        current_item = None

        for line in content.splitlines():
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#"):
                continue

            # Check list item
            if line_strip.startswith("- "):
                if current_list is not None:
                    current_item = {}
                    current_list.append(current_item)
                    pair = line_strip[2:].strip()
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        current_item[k.strip()] = self._cast_value(v.strip())
                continue

            if current_item is not None and line.startswith("    "):
                if ":" in line_strip:
                    k, v = line_strip.split(":", 1)
                    current_item[k.strip()] = self._cast_value(v.strip())
                continue

            if ":" in line_strip:
                k, v = line_strip.split(":", 1)
                k = k.strip()
                v = v.strip()
                if not v:
                    if k == "slabs":
                        current_list = []
                        result[k] = current_list
                        current_section = k
                    else:
                        result[k] = {}
                        current_section = k
                        current_list = None
                        current_item = None
                else:
                    val = self._cast_value(v)
                    if current_section and current_section != "slabs" and line.startswith("  "):
                        result[current_section][k] = val
                    else:
                        result[k] = val
                        current_section = None
                        current_list = None

        return result

    @staticmethod
    def _cast_value(v: str):
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        if v.lower() in ("null", "none", "~"):
            return None
        v_clean = v.strip("\"'")
        try:
            if "." in v_clean:
                return float(v_clean)
            return int(v_clean)
        except ValueError:
            return v_clean

    def load_all_tariffs(self) -> None:
        """Load all versioned tariff configuration files."""
        if not self.tariff_dir.exists():
            return

        for p in self.tariff_dir.glob("*.yaml"):
            try:
                data = self._parse_yaml_file(p)
                slabs = [TariffSlab(**s) for s in data.get("slabs", [])]
                fixed = FixedChargesConfig(**data.get("fixed_charges", {}))
                tax = ElectricityTaxConfig(**data.get("electricity_tax", {}))
                fppca = FppcaConfig(**data.get("fppca", {}))

                cfg = TariffConfig(
                    tariff_code=data["tariff_code"],
                    tariff_name=data["tariff_name"],
                    state=data.get("state", "Tamil Nadu"),
                    authority=data.get("authority", "TNERC"),
                    order_reference=data.get("order_reference", ""),
                    verification_status=data.get("verification_status", "VERIFIED_OFFICIAL"),
                    billing_cycle_days=data.get("billing_cycle_days", 60),
                    billing_cycle_type=data.get("billing_cycle_type", "bi-monthly"),
                    free_allowance_units=data.get("free_allowance_units", 100),
                    free_allowance_description=data.get("free_allowance_description", ""),
                    slabs=slabs,
                    fixed_charges=fixed,
                    electricity_tax=tax,
                    fppca=fppca,
                )
                self._tariffs[cfg.tariff_code] = cfg
            except Exception as e:
                print(f"[Warning] Failed to load tariff file {p.name}: {e}")

    def list_available_tariffs(self) -> List[Dict[str, str]]:
        """List loaded tariffs with verification status."""
        return [
            {
                "tariff_code": t.tariff_code,
                "tariff_name": t.tariff_name,
                "verification_status": t.verification_status,
                "authority": t.authority,
            }
            for t in self._tariffs.values()
        ]

    def get_tariff(self, tariff_code: Optional[str] = None) -> TariffConfig:
        """Retrieve tariff configuration by code, falling back to latest official."""
        if tariff_code and tariff_code in self._tariffs:
            return self._tariffs[tariff_code]

        # Preferred default order
        defaults = ["TN_DOMESTIC_LT1A_2024", "TN_DOMESTIC_LT1A_2023", "TN_DOMESTIC_LT1A_2022"]
        for d in defaults:
            if d in self._tariffs:
                return self._tariffs[d]

        if self._tariffs:
            return next(iter(self._tariffs.values()))

        raise ValueError("No tariff configuration loaded. Please check tariff directory.")

    def calculate_bill(
        self,
        bill_input: BillInput,
        tariff_code: Optional[str] = None,
        simulate_proration: bool = False
    ) -> TariffCalculationResult:
        """
        Perform complete deterministic verification of a domestic electricity bill.
        """
        tariff = self.get_tariff(tariff_code or bill_input.tariff_code)

        # 1. Billing period
        billing_days = calculate_billing_days(
            bill_input.previous_reading_date,
            bill_input.current_reading_date
        )
        is_abnormal_cycle, cycle_notes = evaluate_billing_cycle_duration(
            actual_days=billing_days,
            standard_days=tariff.billing_cycle_days
        )

        # 2. Meter reading check
        is_valid_meter, calc_units, meter_note = check_meter_reading_validity(
            prev_reading=bill_input.previous_meter_reading,
            curr_reading=bill_input.current_meter_reading,
            meter_constant=bill_input.meter_constant
        )

        # Billed units check
        billed_units = bill_input.billed_units
        if billed_units is None:
            billed_units = calc_units
            units_match = True
            units_discrepancy = 0.0
        else:
            units_discrepancy = round(billed_units - calc_units, 2)
            units_match = abs(units_discrepancy) <= 0.5

        # 3. Slab calculation
        units_to_calculate = billed_units if units_match else calc_units
        units_to_calculate = max(0.0, units_to_calculate)

        # Proration factor if simulate_proration is requested
        proration_factor = (billing_days / tariff.billing_cycle_days) if (simulate_proration and billing_days > 0) else 1.0

        slab_breakdowns: List[SlabBreakdown] = []
        gross_total = 0.0
        subsidy_total = 0.0
        net_total = 0.0

        for slab in tariff.slabs:
            lower_bound = (slab.min_units - 1 if slab.min_units > 0 else 0) * proration_factor
            upper_bound = (slab.max_units * proration_factor) if slab.max_units is not None else None

            if units_to_calculate <= lower_bound:
                continue

            if upper_bound is not None:
                slab_units = min(units_to_calculate, upper_bound) - lower_bound
            else:
                slab_units = units_to_calculate - lower_bound

            slab_units = max(0.0, round(slab_units, 2))
            if slab_units <= 0:
                continue

            gross_amount = round(slab_units * slab.rate_per_unit, 2)
            subsidy_amount = round(slab_units * slab.subsidy_per_unit, 2)
            net_amount = round(slab_units * slab.net_rate, 2)

            gross_total += gross_amount
            subsidy_total += subsidy_amount
            net_total += net_amount

            label = f"{int(slab.min_units)} - {int(slab.max_units) if slab.max_units is not None else 'Above'}"
            slab_breakdowns.append(
                SlabBreakdown(
                    slab_label=label,
                    units_in_slab=slab_units,
                    gross_rate=slab.rate_per_unit,
                    gross_amount=gross_amount,
                    subsidy_rate=slab.subsidy_per_unit,
                    subsidy_amount=subsidy_amount,
                    net_rate=slab.net_rate,
                    net_amount=net_amount,
                )
            )

        # 4. Fixed charges
        fixed_charge = tariff.fixed_charges.amount_per_cycle if tariff.fixed_charges.enabled else 0.0
        if bill_input.billed_fixed_charges is not None:
            fixed_charge = bill_input.billed_fixed_charges

        # 5. Electricity Tax
        electricity_tax = 0.0
        if tariff.electricity_tax.enabled and units_to_calculate >= tariff.electricity_tax.threshold_units:
            electricity_tax = round((net_total * tariff.electricity_tax.rate_percentage) / 100.0, 2)

        # 6. FPPCA (Fuel Surcharge)
        fppca_amount = 0.0
        if tariff.fppca.enabled:
            # FPPCA applies to net billable units above free quota (or total units depending on regulatory order)
            billable_units = max(0.0, units_to_calculate - tariff.free_allowance_units)
            fppca_amount = round(billable_units * tariff.fppca.rate_per_unit, 2)
        if bill_input.billed_fppca is not None:
            fppca_amount = bill_input.billed_fppca

        # 7. Grand totals
        current_cycle_total = round(net_total + fixed_charge + electricity_tax + fppca_amount, 2)
        grand_total = round(current_cycle_total + bill_input.arrears_amount, 2)

        # 8. Comparison with billed amount
        discrepancy = None
        exact_match = False
        discrepancy_category = "UNKNOWN"

        if bill_input.billed_amount is not None:
            discrepancy = round(bill_input.billed_amount - grand_total, 2)
            if not units_match:
                discrepancy_category = "METER_READING_MISMATCH"
                exact_match = False
            elif abs(discrepancy) <= 2.0:  # Rounding buffer
                discrepancy_category = "EXACT_MATCH"
                exact_match = True
            elif discrepancy > 2.0:
                discrepancy_category = "OVERBILLED"
                exact_match = False
            else:
                discrepancy_category = "UNDERBILLED"
                exact_match = False
        else:
            discrepancy_category = "EXACT_MATCH" if units_match else "METER_READING_MISMATCH"
            exact_match = units_match

        metadata = {
            "meter_validation_note": meter_note,
            "proration_applied": simulate_proration,
            "proration_factor": proration_factor,
            "order_reference": tariff.order_reference,
            "free_allowance_units": tariff.free_allowance_units,
        }

        return TariffCalculationResult(
            tariff_code=tariff.tariff_code,
            tariff_name=tariff.tariff_name,
            verification_status=tariff.verification_status,
            calculated_units=calc_units,
            billed_units=billed_units,
            units_match=units_match,
            units_discrepancy=units_discrepancy,
            billing_days=billing_days,
            standard_cycle_days=tariff.billing_cycle_days,
            is_cycle_abnormal=is_abnormal_cycle,
            cycle_notes=cycle_notes,
            slab_breakdown=slab_breakdowns,
            energy_charge_gross=round(gross_total, 2),
            subsidy_amount=round(subsidy_total, 2),
            energy_charge_net=round(net_total, 2),
            fixed_charges=round(fixed_charge, 2),
            electricity_tax=round(electricity_tax, 2),
            fppca_amount=round(fppca_amount, 2),
            calculated_current_cycle_total=current_cycle_total,
            arrears_amount=round(bill_input.arrears_amount, 2),
            calculated_grand_total=grand_total,
            billed_amount=bill_input.billed_amount,
            discrepancy_amount=discrepancy,
            is_arithmetic_exact_match=exact_match,
            discrepancy_category=discrepancy_category,
            metadata=metadata,
        )
