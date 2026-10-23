"""Orchestrator for training and comparing anomaly detection models from scratch."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.data_gen.bill_generator import SyntheticBillGenerator
from .features import BillFeatureExtractor, FEATURE_NAMES
from .baselines import RuleBasedBaselineDetector
from .isolation_forest import BillIsolationForest
from .autoencoder import BillAutoencoderDetector


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute standard classification and anomaly evaluation metrics."""
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.50

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
    }


def train_and_benchmark_all_models(
    num_consumers: int = 150,
    cycles_per_consumer: int = 10,
    output_models_dir: str = "./models",
    use_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Generate synthetic data, extract features, train Isolation Forest and Autoencoder
    strictly from scratch, and compare against the rule-based baseline.
    """
    out_dir = Path(output_models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate isolated dataset
    data_gen = SyntheticBillGenerator(seed=42)
    train_df, val_df, test_df = data_gen.generate_full_dataset(
        num_consumers=num_consumers,
        cycles_per_consumer=cycles_per_consumer,
        output_dir="./data/synthetic",
    )

    # 2. Extract features with zero data leakage
    extractor = BillFeatureExtractor()
    X_train, y_train = extractor.extract_dataset_features(train_df, is_training=True)
    X_val, y_val = extractor.extract_dataset_features(val_df, is_training=False)
    X_test, y_test = extractor.extract_dataset_features(test_df, is_training=False)

    # Save feature normalizer
    joblib.dump(extractor, out_dir / "feature_extractor.joblib")

    # 3. Model 1: Rule-Based Baseline
    print("\n--- EVALUATING RULE-BASED BASELINE ---")
    baseline = RuleBasedBaselineDetector()
    baseline_preds = []
    baseline_scores = []
    for _, group in test_df.groupby("consumer_id"):
        records = group.sort_values("cycle_idx").to_dict(orient="records")
        history = []
        for r in records:
            is_anom, score, _ = baseline.evaluate_bill(r, history)
            baseline_preds.append(int(is_anom))
            baseline_scores.append(score)
            history.append(r)

    metrics_baseline = evaluate_predictions(y_test, np.array(baseline_preds), np.array(baseline_scores))
    print(f"Baseline Metrics: {metrics_baseline}")

    # 4. Model 2: Isolation Forest (Trained strictly from scratch)
    print("\n--- TRAINING ISOLATION FOREST FROM SCRATCH ---")
    iso_forest = BillIsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    iso_forest.fit(X_train)
    iso_test_preds = iso_forest.predict(X_test)
    iso_test_scores = iso_forest.score_samples(X_test)
    metrics_iso = evaluate_predictions(y_test, iso_test_preds, iso_test_scores)
    iso_forest.save(str(out_dir / "isolation_forest.joblib"))
    print(f"Isolation Forest Metrics: {metrics_iso}")

    # 5. Model 3: PyTorch Deep Autoencoder (Trained from scratch)
    print("\n--- TRAINING PYTORCH AUTOENCODER FROM SCRATCH ---")
    # Train only on normal samples from training set
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]

    autoencoder = BillAutoencoderDetector(input_dim=X_train.shape[1], latent_dim=4)
    autoencoder.fit(X_train_normal, X_val_normal, epochs=25, batch_size=32, lr=0.005)

    ae_test_preds = autoencoder.predict(X_test)
    ae_test_scores = autoencoder.score_samples(X_test)
    metrics_ae = evaluate_predictions(y_test, ae_test_preds, ae_test_scores)
    autoencoder.save(str(out_dir / "autoencoder_scratch.pth"))
    print(f"Autoencoder Metrics: {metrics_ae}")

    results = {
        "baseline": metrics_baseline,
        "isolation_forest": metrics_iso,
        "autoencoder": metrics_ae,
        "test_samples_count": int(len(y_test)),
        "anomaly_samples_count": int(np.sum(y_test)),
    }

    # Save benchmark json
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # MLflow tracking if available
    try:
        import mlflow
        mlflow.set_experiment("BillGuard_Anomaly_Detection")
        with mlflow.start_run(run_name="Scratch_Models_Benchmark"):
            mlflow.log_params({"num_consumers": num_consumers, "input_dim": X_train.shape[1]})
            for k, v in metrics_iso.items():
                mlflow.log_metric(f"iso_{k}", v)
            for k, v in metrics_ae.items():
                mlflow.log_metric(f"ae_{k}", v)
            for k, v in metrics_baseline.items():
                mlflow.log_metric(f"baseline_{k}", v)
            print("[MLflow] Logged benchmark run successfully.")
    except Exception as e:
        print(f"[MLflow Note] {e}")

    return results


if __name__ == "__main__":
    train_and_benchmark_all_models(num_consumers=80, cycles_per_consumer=8)
