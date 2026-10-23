"""Synthetic domestic bill generator with realistic consumption profiles and anomaly scenarios."""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from .scenario_definitions import (
    CONSUMER_ARCHETYPES,
    BIMONTHLY_SEASONAL_FACTORS,
    ANOMALY_SCENARIOS,
)


class SyntheticBillGenerator:
    """
    Generates synthetic longitudinal electricity billing histories for Indian domestic consumers.
    Reflects South Indian seasonal variations, Tamil Nadu TANGEDCO LT-1A tariffs,
    and distinct injected anomaly scenarios for training and benchmark evaluation.
    """

    def __init__(self, seed: int = 42, tariff_code: str = "TN_DOMESTIC_LT1A_2024"):
        self.seed = seed
        self.tariff_code = tariff_code
        self.tariff_engine = TariffEngine()
        random.seed(seed)
        np.random.seed(seed)

    def generate_consumer_history(
        self,
        consumer_id: str,
        archetype_key: str,
        start_year: int = 2023,
        num_cycles: int = 12,
        anomaly_probability: float = 0.12,
    ) -> List[Dict[str, Any]]:
        """
        Generate longitudinal sequence of bi-monthly bills for a single consumer.
        """
        archetype = CONSUMER_ARCHETYPES[archetype_key]
        records: List[Dict[str, Any]] = []

        # Initial starting meter reading
        current_meter = float(random.randint(1000, 35000))
        start_date = datetime(start_year, 1, 15)

        for cycle_idx in range(num_cycles):
            # Bi-monthly cycle index within year (1 to 6)
            cycle_in_year = (cycle_idx % 6) + 1
            seasonal_weight = BIMONTHLY_SEASONAL_FACTORS[cycle_in_year]

            # Adjust for AC vs non-AC summer amplification
            if archetype.has_ac and cycle_in_year in (2, 3):
                seasonal_weight *= archetype.summer_multiplier
            elif not archetype.has_ac and cycle_in_year in (2, 3):
                seasonal_weight *= 1.15
            elif cycle_in_year in (1, 6):
                seasonal_weight *= archetype.monsoon_multiplier

            # Base usage with Gaussian noise
            base_units = max(
                20.0,
                float(np.random.normal(archetype.base_bimonthly_units_mean, archetype.base_bimonthly_units_std))
            )
            true_units = round(base_units * seasonal_weight, 1)

            # Dates
            prev_date = start_date + timedelta(days=cycle_idx * 60)
            curr_date = prev_date + timedelta(days=60)
            billing_days = 60

            # Default values
            is_anomaly = 0
            anomaly_type = "NORMAL"
            ground_truth_note = "Normal consumption matching archetype and seasonal expectations."
            meter_constant = 1.0
            arrears = 0.0

            # Chance of anomaly (avoid anomaly on cycle 0 to allow clean baseline)
            if cycle_idx > 0 and random.random() < anomaly_probability:
                is_anomaly = 1
                scenario = random.choice([
                    "ARITHMETIC_OVERCHARGE",
                    "METER_FAULT_SURGE",
                    "DELAYED_READING_SLAB_JUMP",
                    "DEAD_METER_ZERO_USAGE",
                    "PHANTOM_ARREARS",
                ])
                anomaly_type = scenario

                if scenario == "ARITHMETIC_OVERCHARGE":
                    ground_truth_note = "Clerical or billing software error: Billed total inflated above slab calculation."
                    # We will compute the true bill, then tamper with billed_amount

                elif scenario == "METER_FAULT_SURGE":
                    surge_factor = random.uniform(3.5, 6.5)
                    true_units = round(true_units * surge_factor, 1)
                    ground_truth_note = f"Defective meter recording or sudden uncharacteristic jump ({surge_factor:.1f}x normal)."

                elif scenario == "DELAYED_READING_SLAB_JUMP":
                    extra_days = random.randint(15, 28)
                    billing_days = 60 + extra_days
                    curr_date = prev_date + timedelta(days=billing_days)
                    # Units accumulate for the extended period
                    true_units = round(true_units * (billing_days / 60.0), 1)
                    ground_truth_note = (
                        f"Meter reading delayed to {billing_days} days (+{extra_days} days). "
                        "Extended cycle pushed units into a higher tariff slab without proration."
                    )

                elif scenario == "DEAD_METER_ZERO_USAGE":
                    true_units = 0.0
                    ground_truth_note = "Mechanical meter stuck or dead register recording 0 units for an active residence."

                elif scenario == "PHANTOM_ARREARS":
                    arrears = float(random.choice([450.0, 850.0, 1500.0, 2400.0]))
                    ground_truth_note = f"Unexplained arrears of ₹{arrears:.2f} appended without prior default or notification."

            # Update meter readings
            prev_meter = current_meter
            current_meter = round(prev_meter + true_units, 1)

            # Calculate true deterministic tariff
            bill_in = BillInput(
                consumer_number=consumer_id,
                consumer_name=f"Consumer {consumer_id}",
                billing_date=curr_date.strftime("%Y-%m-%d"),
                previous_reading_date=prev_date.strftime("%Y-%m-%d"),
                current_reading_date=curr_date.strftime("%Y-%m-%d"),
                previous_meter_reading=prev_meter,
                current_meter_reading=current_meter,
                meter_constant=meter_constant,
                billed_units=true_units,
                arrears_amount=arrears,
                tariff_code=self.tariff_code,
            )
            calc_res = self.tariff_engine.calculate_bill(bill_in)
            true_total = calc_res.calculated_grand_total

            billed_amount = true_total
            if anomaly_type == "ARITHMETIC_OVERCHARGE":
                overcharge = random.choice([250.0, 480.0, 750.0, 1200.0])
                billed_amount = round(true_total + overcharge, 2)

            records.append({
                "consumer_id": consumer_id,
                "archetype": archetype_key,
                "cycle_idx": cycle_idx,
                "cycle_in_year": cycle_in_year,
                "billing_days": billing_days,
                "previous_reading_date": prev_date.strftime("%Y-%m-%d"),
                "current_reading_date": curr_date.strftime("%Y-%m-%d"),
                "previous_meter_reading": prev_meter,
                "current_meter_reading": current_meter,
                "meter_constant": meter_constant,
                "recorded_units": true_units,
                "billed_units": true_units,
                "calculated_units": calc_res.calculated_units,
                "calculated_energy_gross": calc_res.energy_charge_gross,
                "calculated_subsidy": calc_res.subsidy_amount,
                "calculated_energy_net": calc_res.energy_charge_net,
                "calculated_fppca": calc_res.fppca_amount,
                "arrears_amount": arrears,
                "calculated_grand_total": true_total,
                "billed_amount": billed_amount,
                "discrepancy_amount": round(billed_amount - true_total, 2),
                "is_arithmetic_exact_match": abs(billed_amount - true_total) <= 2.0,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type,
                "ground_truth_explanation": ground_truth_note,
                "tariff_code": self.tariff_code,
            })

        return records

    def generate_full_dataset(
        self,
        num_consumers: int = 250,
        cycles_per_consumer: int = 12,
        output_dir: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate full dataset with consumer-level train/validation/test split
        to completely prevent cross-contamination / data leakage.
        """
        all_records: List[Dict[str, Any]] = []
        archetype_keys = list(CONSUMER_ARCHETYPES.keys())

        # Distribution weights: 30% Lifeline, 40% Moderate, 25% High AC, 5% Villa
        weights = [0.30, 0.40, 0.25, 0.05]

        for i in range(num_consumers):
            consumer_id = f"09-245-{str(i // 100).zfill(3)}-{str(i % 100).zfill(3)}"
            arch = random.choices(archetype_keys, weights=weights, k=1)[0]
            records = self.generate_consumer_history(
                consumer_id=consumer_id,
                archetype_key=arch,
                num_cycles=cycles_per_consumer,
                anomaly_probability=0.12,
            )
            all_records.extend(records)

        df = pd.DataFrame(all_records)

        # Consumer-isolated split
        consumer_list = df["consumer_id"].unique()
        random.shuffle(consumer_list)

        n_total = len(consumer_list)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        train_consumers = set(consumer_list[:n_train])
        val_consumers = set(consumer_list[n_train:n_train + n_val])
        test_consumers = set(consumer_list[n_train + n_val:])

        train_df = df[df["consumer_id"].isin(train_consumers)].copy().reset_index(drop=True)
        val_df = df[df["consumer_id"].isin(val_consumers)].copy().reset_index(drop=True)
        test_df = df[df["consumer_id"].isin(test_consumers)].copy().reset_index(drop=True)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(out_path / "train_bills.csv", index=False)
            val_df.to_csv(out_path / "val_bills.csv", index=False)
            test_df.to_csv(out_path / "test_bills.csv", index=False)
            df.to_csv(out_path / "all_bills.csv", index=False)
            print(f"[Dataset] Generated {len(df)} total bills across {n_total} consumers.")
            print(f"[Dataset] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df


if __name__ == "__main__":
    gen = SyntheticBillGenerator(seed=42)
    train_df, val_df, test_df = gen.generate_full_dataset(
        num_consumers=100,
        cycles_per_consumer=12,
        output_dir="./data/synthetic"
    )
