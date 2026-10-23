"""Unit tests for ML Anomaly Detection models and feature engineering."""

import pytest
import numpy as np

from src.ml.features import BillFeatureExtractor, FEATURE_NAMES
from src.ml.baselines import RuleBasedBaselineDetector
from src.ml.isolation_forest import BillIsolationForest
from src.ml.autoencoder import BillAutoencoderDetector


def test_feature_extractor_shape():
    extractor = BillFeatureExtractor()
    current = {
        "recorded_units": 350.0,
        "billing_days": 60,
        "cycle_in_year": 3,
        "calculated_grand_total": 1200.0,
        "discrepancy_amount": 0.0,
        "arrears_amount": 0.0,
    }
    history = [
        {"recorded_units": 240.0, "billing_days": 60},
        {"recorded_units": 260.0, "billing_days": 60},
    ]
    feat = extractor.extract_features_from_history(current, history)
    assert feat.shape == (len(FEATURE_NAMES),)
    assert not np.isnan(feat).any()


def test_rule_based_baseline_surge():
    baseline = RuleBasedBaselineDetector(z_threshold=2.5)
    normal_history = [{"recorded_units": 150.0, "billing_days": 60} for _ in range(5)]
    surge_bill = {"recorded_units": 750.0, "billing_days": 60}

    is_anom, score, reasons = baseline.evaluate_bill(surge_bill, normal_history)
    assert is_anom is True
    assert score > 0.60
    assert len(reasons) > 0


def test_isolation_forest_scratch_fit_and_explain():
    # Synthetic normal cloud + outliers
    np.random.seed(42)
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(100, 8))
    X_outlier = np.random.normal(loc=5.0, scale=1.0, size=(10, 8))
    X_train = np.vstack([X_normal, X_outlier])

    iso = BillIsolationForest(n_estimators=30, contamination=0.10)
    iso.fit(X_train)

    scores = iso.score_samples(X_outlier)
    assert len(scores) == 10
    assert np.mean(scores) > 0.50

    explanation = iso.explain_instance(X_outlier[0])
    assert len(explanation) == 8
    assert "contribution_pct" in explanation[0]


def test_autoencoder_reconstruction_and_attribution():
    np.random.seed(42)
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(60, 8)).astype(np.float32)
    ae = BillAutoencoderDetector(input_dim=8, latent_dim=4)
    loss_history = ae.fit(X_normal, epochs=10, batch_size=16)

    assert len(loss_history) == 10
    assert ae.is_fitted is True

    test_point = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    attributions = ae.explain_instance(test_point)
    # The first feature (value 5.0) should have the highest attribution
    assert attributions[0]["feature"] == FEATURE_NAMES[0]
