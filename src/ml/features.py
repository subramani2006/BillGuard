"""Feature engineering for electricity bill anomaly detection."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.data_gen.scenario_definitions import BIMONTHLY_SEASONAL_FACTORS


FEATURE_NAMES = [
    "daily_consumption",
    "billing_days",
    "ratio_to_history_mean",
    "seasonal_deviation_ratio",
    "peak_to_average_ratio",
    "discrepancy_kwh_rate",
    "arrears_ratio",
    "slab_jump_magnitude",
]


class BillFeatureExtractor:
    """
    Extracts physically meaningful, normalized features for domestic bill anomaly detection.
    Prevents temporal data leakage by utilizing solely past historical cycles.
    """

    def __init__(self):
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def extract_features_from_history(
        self,
        current_bill: Dict[str, Any],
        historical_bills: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract feature vector for a single bill given past billing records.
        """
        units = float(current_bill.get("recorded_units", current_bill.get("billed_units", 0.0)))
        days = float(current_bill.get("billing_days", 60))
        daily_units = units / max(1.0, days)

        # Historical baseline
        if historical_bills:
            past_units = [float(h.get("recorded_units", h.get("billed_units", 0.0))) for h in historical_bills[-6:]]
            hist_mean = float(np.mean(past_units))
            hist_max = float(np.max(past_units))
        else:
            hist_mean = max(10.0, units)
            hist_max = max(10.0, units)

        ratio_to_history = units / max(5.0, hist_mean)
        peak_to_avg = units / max(5.0, hist_max)

        # Seasonal adjustment
        cycle_in_year = int(current_bill.get("cycle_in_year", 3))
        seasonal_expected_factor = BIMONTHLY_SEASONAL_FACTORS.get(cycle_in_year, 1.0)
        expected_units = hist_mean * seasonal_expected_factor
        seasonal_deviation = (units - expected_units) / max(10.0, expected_units)

        # Arithmetic discrepancy rate
        discrepancy = float(current_bill.get("discrepancy_amount", 0.0))
        discrepancy_rate = discrepancy / max(10.0, float(current_bill.get("calculated_grand_total", 100.0)))

        # Arrears ratio
        arrears = float(current_bill.get("arrears_amount", 0.0))
        calc_total = float(current_bill.get("calculated_grand_total", 100.0))
        arrears_ratio = arrears / max(10.0, calc_total)

        # Slab level estimation (0 to 7)
        def get_slab_idx(u):
            if u <= 100: return 0
            if u <= 200: return 1
            if u <= 400: return 2
            if u <= 500: return 3
            if u <= 600: return 4
            if u <= 800: return 5
            return 6

        curr_slab = get_slab_idx(units)
        hist_slab = get_slab_idx(hist_mean)
        slab_jump = float(curr_slab - hist_slab)

        feat_vector = np.array([
            daily_units,
            days,
            ratio_to_history,
            seasonal_deviation,
            peak_to_avg,
            discrepancy_rate,
            arrears_ratio,
            slab_jump,
        ], dtype=np.float32)

        return feat_vector

    def fit_normalizer(self, X: np.ndarray) -> None:
        """Fit feature normalization parameters on training set."""
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        # Avoid division by zero
        self.feature_stds[self.feature_stds < 1e-5] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize feature matrix."""
        if self.feature_means is None or self.feature_stds is None:
            return X
        return (X - self.feature_means) / self.feature_stds

    def extract_dataset_features(
        self,
        df: pd.DataFrame,
        is_training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features across an entire dataframe preserving consumer histories.
        Returns (X, y).
        """
        features_list = []
        labels_list = []

        # Group by consumer to build sequential history
        for _, group in df.groupby("consumer_id"):
            records = group.sort_values("cycle_idx").to_dict(orient="records")
            history = []
            for r in records:
                feats = self.extract_features_from_history(r, history)
                features_list.append(feats)
                labels_list.append(int(r.get("is_anomaly", 0)))
                history.append(r)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int32)

        if is_training:
            self.fit_normalizer(X)

        X_norm = self.transform(X)
        return X_norm, y
