"""Data Drift and Data Quality Monitoring Module."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and current distributions.
    PSI < 0.10: No significant shift
    0.10 <= PSI < 0.25: Moderate shift (Warning)
    PSI >= 0.25: Significant shift (Critical drift alert)
    """
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Determine quantile bins from baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    # Ensure distinct bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Ensure min/max range covers both datasets
    bin_edges[0] = min(bin_edges[0], np.min(current)) - 1e-3
    bin_edges[-1] = max(bin_edges[-1], np.max(current)) + 1e-3

    # Bin counts
    base_counts, _ = np.histogram(baseline, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Fractions with smoothing
    base_pct = (base_counts / max(1, len(baseline))) + epsilon
    curr_pct = (curr_counts / max(1, len(current))) + epsilon

    # Normalize back to 1
    base_pct /= np.sum(base_pct)
    curr_pct /= np.sum(curr_pct)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(round(psi, 4))


class DriftMonitor:
    """
    Continuous data and concept drift monitor for electricity consumption features.
    Provides feature-level PSI, two-sample Kolmogorov-Smirnov tests, and data quality audits.
    """

    def __init__(
        self,
        psi_warning_thresh: float = 0.10,
        psi_critical_thresh: float = 0.25,
        ks_alpha: float = 0.05,
    ):
        self.psi_warning_thresh = psi_warning_thresh
        self.psi_critical_thresh = psi_critical_thresh
        self.ks_alpha = ks_alpha

    def evaluate_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numerical_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute drift statistics across all numerical features.
        """
        features = numerical_features or [
            "recorded_units", "billed_units", "billing_days",
            "calculated_grand_total", "billed_amount", "discrepancy_amount"
        ]

        # Filter to features present in both dataframes
        features = [f for f in features if f in baseline_df.columns and f in current_df.columns]

        report_features = {}
        drift_detected_count = 0
        max_psi = 0.0

        for feat in features:
            base_vals = baseline_df[feat].dropna().to_numpy()
            curr_vals = current_df[feat].dropna().to_numpy()

            if len(base_vals) == 0 or len(curr_vals) == 0:
                continue

            # 1. PSI Calculation
            psi_val = calculate_psi(base_vals, curr_vals)
            max_psi = max(max_psi, psi_val)

            # 2. Two-sample Kolmogorov-Smirnov Test
            ks_res = ks_2samp(base_vals, curr_vals)
            ks_stat = float(round(ks_res.statistic, 4))
            p_val = float(round(ks_res.pvalue, 4))

            # Drift verdict
            is_drift = psi_val >= self.psi_warning_thresh or p_val < self.ks_alpha
            if is_drift:
                drift_detected_count += 1

            severity = "NONE"
            if psi_val >= self.psi_critical_thresh:
                severity = "CRITICAL"
            elif psi_val >= self.psi_warning_thresh:
                severity = "WARNING"

            report_features[feat] = {
                "psi": psi_val,
                "ks_statistic": ks_stat,
                "ks_pvalue": p_val,
                "drift_detected": is_drift,
                "severity": severity,
                "baseline_mean": float(round(np.mean(base_vals), 2)),
                "current_mean": float(round(np.mean(curr_vals), 2)),
                "baseline_std": float(round(np.std(base_vals), 2)),
                "current_std": float(round(np.std(curr_vals), 2)),
            }

        overall_status = "NORMAL"
        if max_psi >= self.psi_critical_thresh:
            overall_status = "CRITICAL_DRIFT"
        elif drift_detected_count > 0:
            overall_status = "WARNING_DRIFT"

        return {
            "overall_status": overall_status,
            "drift_detected_features_count": drift_detected_count,
            "total_monitored_features": len(features),
            "max_psi": max_psi,
            "features": report_features,
            "retraining_recommended": overall_status in ("CRITICAL_DRIFT", "WARNING_DRIFT"),
        }

    def evaluate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess missing values, anomalies, and physical boundary violations."""
        total_rows = len(df)
        if total_rows == 0:
            return {"total_rows": 0, "issues": []}

        issues = []

        # Check negative units
        if "recorded_units" in df.columns:
            neg_units = int((df["recorded_units"] < 0).sum())
            if neg_units > 0:
                issues.append(f"{neg_units} rows have negative recorded units (physically impossible).")

        # Check extreme high units (> 10,000 for domestic)
        if "recorded_units" in df.columns:
            extreme_units = int((df["recorded_units"] > 10000).sum())
            if extreme_units > 0:
                issues.append(f"{extreme_units} rows have recorded units > 10,000 kWh.")

        # Check missing dates or amounts
        missing_count = int(df.isnull().sum().sum())
        if missing_count > 0:
            issues.append(f"{missing_count} total null or missing entries detected across columns.")

        return {
            "total_rows": total_rows,
            "total_issues_count": len(issues),
            "issues": issues,
            "data_quality_score": round(max(0.0, 1.0 - (len(issues) * 0.15)), 2),
        }
