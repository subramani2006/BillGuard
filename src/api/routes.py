"""FastAPI routes for BillGuard verification, OCR, MLOps, and Grievance services."""

import io
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np
import pandas as pd
import joblib

from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from src.data_gen.bill_generator import SyntheticBillGenerator
from src.ml.features import BillFeatureExtractor
from src.ml.baselines import RuleBasedBaselineDetector
from src.ml.isolation_forest import BillIsolationForest
from src.ml.autoencoder import BillAutoencoderDetector
from src.ml.train import train_and_benchmark_all_models
from src.explanation.engine import ExplanationEngine
from src.explanation.grievance_generator import GrievanceGenerator
from src.ocr.extractor import BillOcrExtractor
from src.ocr.train import train_ocr_model
from src.mlops.drift_monitor import DriftMonitor
from src.mlops.registry import ModelRegistry
from src.mlops.audit_logger import AuditLogger, AuditEntry

from .schemas import (
    BillVerifyRequest,
    BillVerifyResponse,
    OCRUploadResponse,
    PromoteRequest,
    RetrainRequest,
)

router = APIRouter()

# Singletons
tariff_engine = TariffEngine()
exp_engine = ExplanationEngine()
grievance_gen = GrievanceGenerator()
drift_monitor = DriftMonitor()
registry = ModelRegistry()
audit_logger = AuditLogger()
ocr_extractor = BillOcrExtractor()
baseline_detector = RuleBasedBaselineDetector()

# Lazy loaded ML components
if os.path.exists("./models/feature_extractor.joblib"):
    try:
        feature_extractor = joblib.load("./models/feature_extractor.joblib")
    except Exception:
        feature_extractor = BillFeatureExtractor()
else:
    feature_extractor = BillFeatureExtractor()

iso_forest: Optional[BillIsolationForest] = None
autoencoder: Optional[BillAutoencoderDetector] = None


def get_active_ml_model():
    """Retrieve or initialize the active production ML model."""
    global iso_forest, autoencoder
    prod = registry.get_production_model()

    if prod and "Autoencoder" in prod.model_name and os.path.exists("./models/autoencoder_scratch.pth"):
        if autoencoder is None:
            autoencoder = BillAutoencoderDetector()
            autoencoder.load("./models/autoencoder_scratch.pth")
        return autoencoder, "Autoencoder"

    if os.path.exists("./models/isolation_forest.joblib"):
        if iso_forest is None:
            iso_forest = BillIsolationForest()
            iso_forest.load("./models/isolation_forest.joblib")
        return iso_forest, "IsolationForest"

    return None, "BaselineOnly"


@router.get("/health")
def health_check():
    prod = registry.get_production_model()
    return {
        "status": "healthy",
        "service": "BillGuard API",
        "version": "1.0.0",
        "active_production_model": prod.model_name if prod else "Baseline",
        "tariffs_loaded": len(tariff_engine.list_available_tariffs()),
    }


@router.get("/tariffs")
def list_tariffs():
    return tariff_engine.list_available_tariffs()


@router.post("/verify-bill", response_model=BillVerifyResponse)
def verify_bill(req: BillVerifyRequest):
    """
    Complete verification pipeline:
    1. Deterministic tariff & meter reading validation.
    2. ML anomaly score calculation & feature attributions.
    3. Bilingual explanation generation.
    4. Grievance petition generation.
    5. Audit logging.
    """
    bill_input = BillInput(
        consumer_number=req.consumer_number,
        consumer_name=req.consumer_name,
        billing_date=req.billing_date,
        previous_reading_date=req.previous_reading_date,
        current_reading_date=req.current_reading_date,
        previous_meter_reading=req.previous_meter_reading,
        current_meter_reading=req.current_meter_reading,
        meter_constant=req.meter_constant,
        billed_units=req.billed_units,
        billed_amount=req.billed_amount,
        arrears_amount=req.arrears_amount,
        tariff_code=req.tariff_code,
    )
    tariff_res = tariff_engine.calculate_bill(bill_input)

    # Historical consumption baseline
    hist_units = req.historical_units or [tariff_res.calculated_units]
    hist_mean = float(np.mean(hist_units))

    # Evaluate ML / Baseline
    hist_records = [{"recorded_units": u, "billing_days": 60} for u in hist_units]
    current_record = {
        "recorded_units": tariff_res.calculated_units,
        "billed_units": req.billed_units or tariff_res.calculated_units,
        "billing_days": tariff_res.billing_days,
        "cycle_in_year": 3,
        "calculated_grand_total": tariff_res.calculated_grand_total,
        "discrepancy_amount": tariff_res.discrepancy_amount or 0.0,
        "arrears_amount": req.arrears_amount,
    }

    # Extract feature vector and transform using fitted normalizer
    feat_vec = feature_extractor.extract_features_from_history(current_record, hist_records)
    feat_vec_norm = feature_extractor.transform(feat_vec.reshape(1, -1))

    ml_model, model_type = get_active_ml_model()
    if ml_model is not None:
        try:
            scores = ml_model.score_samples(feat_vec_norm)
            preds = ml_model.predict(feat_vec_norm)
            ml_score = float(scores[0])
            is_ml_anom = bool(preds[0] == 1)
            attributions = ml_model.explain_instance(feat_vec_norm.flatten())
        except Exception:
            ml_score = 0.20
            is_ml_anom = False
            attributions = []
    else:
        # Fallback to rule-based baseline
        is_ml_anom, ml_score, _ = baseline_detector.evaluate_bill(current_record, hist_records)
        attributions = []

    # Synthesize bilingual explanation
    explanation = exp_engine.generate_explanation(
        tariff_result=tariff_res,
        ml_score=ml_score,
        ml_predicted=is_ml_anom,
        feature_attributions=attributions,
        historical_mean_units=hist_mean,
        previous_meter_reading=req.previous_meter_reading,
        current_meter_reading=req.current_meter_reading,
    )

    # Generate formal representation letter text
    petition_text = grievance_gen.generate_complaint_text(
        consumer_number=req.consumer_number,
        consumer_name=req.consumer_name,
        section_office=req.section_office,
        tariff_result=tariff_res,
        explanation=explanation,
    )

    # Persist in Audit Log
    prod = registry.get_production_model()
    audit_entry = AuditEntry(
        consumer_number=req.consumer_number,
        billing_date=req.billing_date,
        previous_reading=req.previous_meter_reading,
        current_reading=req.current_meter_reading,
        calculated_units=tariff_res.calculated_units,
        billed_units=req.billed_units,
        calculated_total=tariff_res.calculated_grand_total,
        billed_amount=req.billed_amount,
        discrepancy_amount=tariff_res.discrepancy_amount,
        is_anomaly=explanation.is_anomaly,
        anomaly_category=explanation.category,
        certainty_level=explanation.certainty_level,
        ml_score=ml_score,
        model_version=prod.version if prod else 1,
    )
    log_id = audit_logger.log_prediction(audit_entry)

    return BillVerifyResponse(
        tariff_result=tariff_res,
        ml_score=round(ml_score, 3),
        is_anomaly=explanation.is_anomaly,
        explanation=explanation,
        grievance_letter_text=petition_text,
        audit_log_id=log_id,
    )


@router.post("/ocr-upload", response_model=OCRUploadResponse)
async def upload_bill_image(file: UploadFile = File(...)):
    """
    Upload a bill card image, run from-scratch CRNN OCR extraction,
    and return fields with per-field confidence scores and human fallback indicators.
    """
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    ocr_res = ocr_extractor.extract_from_bill_image(img)

    hint = "OCR extraction completed. Please review fields flagged with low confidence before proceeding." if ocr_res["needs_manual_verification"] else "High confidence OCR extraction. Review and proceed to verification."

    return OCRUploadResponse(
        extracted_fields=ocr_res["fields"],
        field_confidences=ocr_res["confidences"],
        low_confidence_fields=ocr_res["low_confidence_fields"],
        needs_manual_verification=ocr_res["needs_manual_verification"],
        verification_hint=hint,
    )


@router.get("/generate-grievance-pdf")
def export_grievance_pdf(
    consumer_number: str = "09-245-001-042",
    consumer_name: str = "M. Senthilkumar",
    section_office: str = "Anna Nagar West, Chennai",
    prev_reading: float = 4500.0,
    curr_reading: float = 4850.0,
    billed_amount: float = 1350.0,
    tariff_code: str = "TN_DOMESTIC_LT1A_2024",
):
    """Generate and download publication-quality PDF representation petition."""
    b = BillInput(
        consumer_number=consumer_number,
        consumer_name=consumer_name,
        previous_meter_reading=prev_reading,
        current_meter_reading=curr_reading,
        billed_amount=billed_amount,
        tariff_code=tariff_code,
    )
    res = tariff_engine.calculate_bill(b)
    exp = exp_engine.generate_explanation(res)

    pdf_out = Path("./data/temp_grievance.pdf")
    grievance_gen.generate_complaint_pdf(
        output_filepath=str(pdf_out),
        consumer_number=consumer_number,
        consumer_name=consumer_name,
        section_office=section_office,
        tariff_result=res,
        explanation=exp,
    )
    return FileResponse(
        str(pdf_out),
        media_type="application/pdf",
        filename=f"BillGuard_Grievance_{consumer_number}.pdf"
    )


@router.get("/mlops/drift")
def get_drift_report():
    """Evaluate data and feature drift comparing training baseline against recent bills."""
    train_path = Path("./data/synthetic/train_bills.csv")
    test_path = Path("./data/synthetic/test_bills.csv")

    if not train_path.exists() or not test_path.exists():
        # Quick synthetic generation
        data_gen = SyntheticBillGenerator(seed=42)
        df_base, _, df_curr = data_gen.generate_full_dataset(num_consumers=30, output_dir="./data/synthetic")
    else:
        df_base = pd.read_csv(train_path)
        df_curr = pd.read_csv(test_path)

    drift_res = drift_monitor.evaluate_feature_drift(df_base, df_curr)
    quality_res = drift_monitor.evaluate_data_quality(df_curr)

    return {
        "drift": drift_res,
        "data_quality": quality_res,
    }


@router.get("/mlops/models")
def get_model_registry():
    """Retrieve model registry versions and active production model."""
    prod = registry.get_production_model()
    return {
        "production_model": prod,
        "all_versions": registry.versions,
    }


@router.post("/mlops/promote")
def promote_model(req: PromoteRequest):
    """Promote a candidate model version to production through safety gate."""
    success, msg = registry.promote_candidate(req.version)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@router.post("/mlops/rollback")
def rollback_model():
    """Roll back production model to preceding archived version."""
    success, msg = registry.rollback_production()
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@router.post("/mlops/retrain")
def trigger_retraining(req: RetrainRequest):
    """Trigger automated end-to-end retraining from scratch."""
    benchmarks = train_and_benchmark_all_models(
        num_consumers=req.num_consumers,
        cycles_per_consumer=req.cycles_per_consumer,
        output_models_dir="./models",
    )
    # Register candidate in registry
    cand = registry.register_candidate(
        model_name="IsolationForest_AutoRetrained",
        artifact_path="./models/isolation_forest.joblib",
        metrics=benchmarks["isolation_forest"],
        description=f"Auto-retrained model across {req.num_consumers} consumers",
    )
    return {
        "status": "completed",
        "candidate_registered_version": cand.version,
        "benchmarks": benchmarks,
    }


@router.get("/mlops/audit-logs")
def get_audit_logs(limit: int = 50):
    """Retrieve recent prediction audit logs."""
    logs = audit_logger.fetch_recent_logs(limit=limit)
    stats = audit_logger.get_audit_summary_stats()
    return {
        "summary": stats,
        "logs": logs,
    }
