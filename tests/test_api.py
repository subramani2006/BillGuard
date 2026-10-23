"""Integration tests for FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.data_gen.ocr_image_generator import BillImageGenerator

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BillGuard API"


def test_tariffs_endpoint():
    res = client.get("/api/tariffs")
    assert res.status_code == 200
    tariffs = res.json()
    assert len(tariffs) >= 3
    assert any("TN_DOMESTIC_LT1A" in t["tariff_code"] for t in tariffs)


def test_verify_bill_arithmetic_mismatch():
    payload = {
        "consumer_number": "09-245-001-042",
        "consumer_name": "R. Murugesan",
        "section_office": "Guindy, Chennai",
        "billing_date": "2024-08-15",
        "previous_reading_date": "2024-06-15",
        "current_reading_date": "2024-08-15",
        "previous_meter_reading": 1000.0,
        "current_meter_reading": 1250.0,  # 250 units
        "billed_units": 250.0,
        "billed_amount": 950.0,  # Overbilled vs calculated ~534.10
        "tariff_code": "TN_DOMESTIC_LT1A_2024",
    }
    res = client.post("/api/verify-bill", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["is_anomaly"] is True
    assert data["explanation"]["category"] == "CONFIRMED_ARITHMETIC_ERROR"
    assert "grievance_letter_text" in data
    assert "FORMAL REPRESENTATION" in data["grievance_letter_text"]
    assert data["audit_log_id"] > 0


def test_ocr_upload_endpoint():
    # Render synthetic bill
    gen = BillImageGenerator(seed=42)
    img, _ = gen.generate_full_bill_image(
        consumer_no="09-245-001",
        consumer_name="M. Selvakumar",
        bill_date="2024-08-15",
        prev_date="2024-06-15",
        prev_reading=4500.0,
        curr_reading=4850.0,
        units=350.0,
        net_amount=1250.0,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/ocr-upload",
        files={"file": ("bill.png", buf, "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "extracted_fields" in data
    assert "field_confidences" in data
    assert "needs_manual_verification" in data


def test_mlops_drift_endpoint():
    res = client.get("/api/mlops/drift")
    assert res.status_code == 200
    data = res.json()
    assert "drift" in data
    assert "data_quality" in data


def test_mlops_models_endpoint():
    res = client.get("/api/mlops/models")
    assert res.status_code == 200
    data = res.json()
    assert "all_versions" in data
    assert len(data["all_versions"]) >= 1
