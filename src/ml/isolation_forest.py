"""Isolation Forest anomaly detector trained from scratch."""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from .features import FEATURE_NAMES


class BillIsolationForest:
    """
    Isolation Forest model trained strictly from scratch on self-generated local data.
    Evaluates multi-dimensional consumption, billing days, seasonal variance, and discrepancy rates.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        contamination: float = 0.08,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> None:
        """Fit Isolation Forest from scratch on training feature matrix."""
        self.model.fit(X_train)
        self.is_fitted = True

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly score in range [0, 1].
        Higher values indicate higher likelihood of anomaly.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        # decision_function outputs negative scores for anomalies, positive for normal
        raw_scores = -self.model.decision_function(X)
        # Normalize to [0, 1] using sigmoid-like scaling
        norm_scores = 1.0 / (1.0 + np.exp(-5.0 * raw_scores))
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels: 1 for anomaly, 0 for normal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        preds = self.model.predict(X)
        # sklearn outputs -1 for outliers, 1 for inliers -> convert to 1 for anomaly, 0 for normal
        return (preds == -1).astype(np.int32)

    def explain_instance(
        self,
        x_single: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute heuristic feature attribution for an individual sample
        by measuring individual feature distance from normal training distribution.
        """
        names = feature_names or FEATURE_NAMES
        attributions = []
        # Magnitude of deviation in normalized space indicates feature contribution
        total_dev = float(np.sum(np.abs(x_single))) + 1e-6
        for name, val in zip(names, x_single):
            contribution_pct = (abs(val) / total_dev) * 100.0
            attributions.append({
                "feature": name,
                "normalized_value": round(float(val), 3),
                "contribution_pct": round(float(contribution_pct), 1),
            })
        # Sort by contribution descending
        attributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
        return attributions

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)

    def load(self, filepath: str) -> None:
        """Load trained model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
