"""Unit tests for MLOps: Drift monitoring, Model Registry promotion & rollback, Audit logging."""

import os
import pytest
import numpy as np
import pandas as pd

from src.mlops.drift_monitor import DriftMonitor, calculate_psi
from src.mlops.registry import ModelRegistry
from src.mlops.audit_logger import AuditLogger, AuditEntry


def test_psi_identical_distributions():
    """Identical distributions should exhibit negligible PSI (< 0.05)."""
    np.random.seed(42)
    dist1 = np.random.normal(250.0, 40.0, 1000)
    dist2 = np.random.normal(250.0, 40.0, 1000)

    psi = calculate_psi(dist1, dist2)
    assert psi < 0.05


def test_psi_drifted_distribution():
    """Significantly shifted distribution should trigger high PSI (> 0.25)."""
    np.random.seed(42)
    baseline = np.random.normal(250.0, 30.0, 1000)
    drifted = np.random.normal(550.0, 60.0, 1000)  # Significant shift

    psi = calculate_psi(baseline, drifted)
    assert psi > 0.25


def test_drift_monitor_dataframe_evaluation():
    monitor = DriftMonitor()
    df_base = pd.DataFrame({
        "recorded_units": np.random.normal(200.0, 30.0, 200),
        "billing_days": [60] * 200,
        "discrepancy_amount": [0.0] * 200,
    })
    df_curr = pd.DataFrame({
        "recorded_units": np.random.normal(500.0, 80.0, 200),  # Sharp surge
        "billing_days": [60] * 200,
        "discrepancy_amount": [0.0] * 200,
    })

    report = monitor.evaluate_feature_drift(df_base, df_curr)
    assert report["overall_status"] in ("CRITICAL_DRIFT", "WARNING_DRIFT")
    assert report["retraining_recommended"] is True
    assert "recorded_units" in report["features"]
    assert report["features"]["recorded_units"]["drift_detected"] is True


def test_model_registry_promotion_and_rollback(tmp_path):
    reg_path = tmp_path / "test_registry.json"
    registry = ModelRegistry(registry_file=str(reg_path))

    # Initial state should have v1 in PRODUCTION
    prod = registry.get_production_model()
    assert prod is not None
    assert prod.version == 1

    # Register candidate v2 with superior F1
    cand = registry.register_candidate(
        model_name="Autoencoder_V2",
        artifact_path="./models/ae_v2.pth",
        metrics={"precision": 0.90, "recall": 0.88, "f1_score": 0.89},
        description="Candidate model with enhanced summer feature support"
    )
    assert cand.version == 2
    assert cand.stage == "CANDIDATE"

    # Evaluate promotion gate
    can_promote, reason = registry.evaluate_promotion_gate(candidate_version=2)
    assert can_promote is True

    # Promote candidate to PRODUCTION
    success, msg = registry.promote_candidate(candidate_version=2)
    assert success is True

    new_prod = registry.get_production_model()
    assert new_prod.version == 2
    assert new_prod.stage == "PRODUCTION"

    # Verify v1 is now ARCHIVED
    v1 = next(v for v in registry.versions if v.version == 1)
    assert v1.stage == "ARCHIVED"

    # Test Rollback
    rollback_success, rb_msg = registry.rollback_production()
    assert rollback_success is True

    rolled_back_prod = registry.get_production_model()
    assert rolled_back_prod.version == 1
    assert rolled_back_prod.stage == "PRODUCTION"


def test_audit_logger(tmp_path):
    db_p = tmp_path / "audit.db"
    jsonl_p = tmp_path / "audit.jsonl"
    logger = AuditLogger(db_path=str(db_p), jsonl_path=str(jsonl_p))

    entry = AuditEntry(
        consumer_number="09-245-001",
        billing_date="2024-08-01",
        previous_reading=1000.0,
        current_reading=1250.0,
        calculated_units=250.0,
        billed_units=250.0,
        calculated_total=534.10,
        billed_amount=786.50,
        discrepancy_amount=252.40,
        is_anomaly=True,
        anomaly_category="CONFIRMED_ARITHMETIC_ERROR",
        certainty_level="HIGH_CERTAINTY",
        ml_score=0.85,
        model_version=1,
    )
    log_id = logger.log_prediction(entry)
    assert log_id > 0

    logs = logger.fetch_recent_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["consumer_number"] == "09-245-001"
    assert logs[0]["discrepancy_amount"] == 252.40

    stats = logger.get_audit_summary_stats()
    assert stats["total_bills_audited"] == 1
    assert stats["anomalies_flagged"] == 1


def test_candidate_promotion_gate_rejection(tmp_path):
    """
    Ensure automated gatekeeper rejects candidates failing acceptance criteria (F1 < 0.75 or F1 < Prod),
    and active production model remains intact and operational.
    """
    reg_path = tmp_path / "test_reg_rejection.json"
    registry = ModelRegistry(registry_file=str(reg_path))

    # v1 starts as production with F1 = 0.85
    prod_orig = registry.get_production_model()
    assert prod_orig.version == 1

    # Register candidate v2 with substandard F1 (0.62 < 0.75)
    cand_low = registry.register_candidate(
        model_name="Substandard_Model",
        artifact_path="./models/substandard.pth",
        metrics={"precision": 0.65, "recall": 0.60, "f1_score": 0.62},
        description="Low quality model that must be rejected"
    )
    can_promote, reason = registry.evaluate_promotion_gate(candidate_version=cand_low.version)
    assert can_promote is False
    assert "Gate Failed" in reason

    # Attempt promotion should fail
    promoted, msg = registry.promote_candidate(candidate_version=cand_low.version)
    assert promoted is False

    # Production model must remain v1 unchanged
    active_prod = registry.get_production_model()
    assert active_prod.version == 1
    assert active_prod.stage == "PRODUCTION"

    # Candidate must still be in CANDIDATE stage
    cand_rec = next(v for v in registry.versions if v.version == cand_low.version)
    assert cand_rec.stage == "CANDIDATE"

