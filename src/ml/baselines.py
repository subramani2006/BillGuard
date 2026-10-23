"""Simple statistical and rule-based anomaly detection baseline."""

from typing import Dict, Any, List, Tuple
import numpy as np


class RuleBasedBaselineDetector:
    """
    Baseline anomaly detector using classical heuristic rules and statistical 3-sigma tests.
    Serves as the benchmark against which machine learning models are rigorously compared.
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        pct_increase_threshold: float = 0.60,
    ):
        self.z_threshold = z_threshold
        self.pct_increase_threshold = pct_increase_threshold

    def evaluate_bill(
        self,
        current_bill: Dict[str, Any],
        historical_bills: List[Dict[str, Any]],
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluate a single bill using simple statistical heuristics.
        Returns: (is_anomaly, anomaly_score_0_to_1, reasons_list).
        """
        units = float(current_bill.get("recorded_units", current_bill.get("billed_units", 0.0)))
        days = float(current_bill.get("billing_days", 60))
        reasons: List[str] = []
        score = 0.0

        if not historical_bills:
            return False, 0.05, ["Insufficient historical baseline (first billing cycle)."]

        past_units = [float(h.get("recorded_units", h.get("billed_units", 0.0))) for h in historical_bills[-6:]]
        mean_u = float(np.mean(past_units))
        std_u = float(np.std(past_units)) if len(past_units) > 1 else max(15.0, mean_u * 0.20)
        std_u = max(5.0, std_u)

        # 1. Z-Score Test
        z_score = (units - mean_u) / std_u
        if z_score > self.z_threshold:
            reasons.append(f"Consumption exceeds 3-sigma baseline (Z-score: {z_score:.2f} > {self.z_threshold}).")
            score = max(score, min(0.95, 0.5 + (z_score / 10.0)))

        # 2. Percentage Spike Test
        pct_increase = (units - mean_u) / max(10.0, mean_u)
        if pct_increase > self.pct_increase_threshold and units > 150:
            reasons.append(f"Consumption increased by {pct_increase * 100:.1f}% compared to 6-month average.")
            score = max(score, 0.70)

        # 3. Stuck Meter / Zero Consumption Test
        if units == 0.0 and mean_u > 120.0:
            reasons.append("Zero consumption recorded for an active household with established usage baseline.")
            score = max(score, 0.85)

        # 4. Extended Billing Cycle Test
        if days > 70:
            reasons.append(f"Extended billing cycle ({days} days) without demonstrated slab proration.")
            score = max(score, 0.60)

        is_anomaly = len(reasons) > 0
        return is_anomaly, round(score, 3), reasons
