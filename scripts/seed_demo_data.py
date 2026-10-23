"""Seed script: Generates synthetic data and trains models strictly from scratch."""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_gen.bill_generator import SyntheticBillGenerator
from src.data_gen.ocr_image_generator import BillImageGenerator
from src.ocr.train import train_ocr_model
from src.ml.train import train_and_benchmark_all_models
from src.mlops.registry import ModelRegistry
from src.mlops.audit_logger import AuditLogger, AuditEntry
from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from src.explanation.engine import ExplanationEngine


def seed_all():
    print("=" * 70)
    print("BILLGUARD SEEDING PIPELINE (STRICTLY FROM SCRATCH - ZERO PRETRAINED WEIGHTS)")
    print("=" * 70)

    # 1. Generate Synthetic Longitudinal Bills
    print("\n[Step 1/5] Generating Longitudinal Domestic Bill Histories...")
    bill_gen = SyntheticBillGenerator(seed=42)
    train_df, val_df, test_df = bill_gen.generate_full_dataset(
        num_consumers=100,
        cycles_per_consumer=12,
        output_dir=str(ROOT / "data" / "synthetic")
    )
    print(f"Generated {len(train_df)} train, {len(val_df)} val, {len(test_df)} test bills.")

    # 2. Train From-Scratch CRNN OCR Model
    print("\n[Step 2/5] Training CRNN OCR Model from Random Initialization...")
    ocr_metrics = train_ocr_model(
        num_samples=400,
        epochs=5,
        batch_size=16,
        output_model_path=str(ROOT / "models" / "ocr_crnn_scratch.pth")
    )
    print(f"OCR Training complete. Final Test CER: {ocr_metrics['test_cer']:.4f}")

    # 3. Train Anomaly Detection Models from Scratch (Isolation Forest & Autoencoder)
    print("\n[Step 3/5] Training Anomaly Models (Isolation Forest & PyTorch Autoencoder)...")
    benchmarks = train_and_benchmark_all_models(
        num_consumers=80,
        cycles_per_consumer=10,
        output_models_dir=str(ROOT / "models")
    )
    print(f"Anomaly benchmarks saved. Isolation Forest F1: {benchmarks['isolation_forest']['f1_score']:.3f}, Autoencoder F1: {benchmarks['autoencoder']['f1_score']:.3f}")

    # 4. Initialize Model Registry
    print("\n[Step 4/5] Initializing Model Registry...")
    registry = ModelRegistry(registry_file=str(ROOT / "models" / "registry.json"))
    # Register candidate autoencoder
    cand = registry.register_candidate(
        model_name="Autoencoder_Scratch",
        artifact_path=str(ROOT / "models" / "autoencoder_scratch.pth"),
        metrics=benchmarks["autoencoder"],
        description="Deep Bottleneck Autoencoder trained solely on normal consumption patterns",
    )
    print(f"Registered candidate model v{cand.version}")

    # 5. Populate Sample Audit Logs
    print("\n[Step 5/5] Generating Sample Prediction Audit Logs...")
    audit = AuditLogger(
        db_path=str(ROOT / "data" / "billguard_audit.db"),
        jsonl_path=str(ROOT / "data" / "audit_logs.jsonl")
    )
    t_engine = TariffEngine()
    e_engine = ExplanationEngine()

    sample_cases = [
        ("09-245-001-010", 1000.0, 1250.0, 950.0),    # Overbilled
        ("09-245-001-020", 2500.0, 2700.0, 247.0),    # Normal 200 units
        ("09-245-001-030", 3000.0, 3100.0, 0.0),      # Lifeline 100 units
        ("09-245-001-040", 4000.0, 4800.0, 5200.0),   # High surge
    ]
    for c_no, p_read, c_read, b_amt in sample_cases:
        b_in = BillInput(
            consumer_number=c_no,
            previous_meter_reading=p_read,
            current_meter_reading=c_read,
            billed_amount=b_amt,
            tariff_code="TN_DOMESTIC_LT1A_2024",
        )
        res = t_engine.calculate_bill(b_in)
        exp = e_engine.generate_explanation(res)
        audit.log_prediction(AuditEntry(
            consumer_number=c_no,
            billing_date="2024-08-15",
            previous_reading=p_read,
            current_reading=c_read,
            calculated_units=res.calculated_units,
            billed_units=res.calculated_units,
            calculated_total=res.calculated_grand_total,
            billed_amount=b_amt,
            discrepancy_amount=res.discrepancy_amount,
            is_anomaly=exp.is_anomaly,
            anomaly_category=exp.category,
            certainty_level=exp.certainty_level,
            ml_score=0.85 if exp.is_anomaly else 0.12,
        ))

    print("\n✅ SEEDING COMPLETE! BillGuard is fully initialized and operational.")


if __name__ == "__main__":
    seed_all()
