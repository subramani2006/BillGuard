"""Pydantic Request and Response Schemas for FastAPI."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.tariff.models import BillInput, TariffCalculationResult
from src.explanation.engine import ExplanationReport
from src.mlops.registry import ModelVersionMetadata


class BillVerifyRequest(BaseModel):
    consumer_number: str = "09-245-001-042"
    consumer_name: str = "M. Senthilkumar"
    section_office: str = "Anna Nagar West, Chennai"
    billing_date: str = "2024-08-15"
    previous_reading_date: str = "2024-06-15"
    current_reading_date: str = "2024-08-15"
    previous_meter_reading: float = 4500.0
    current_meter_reading: float = 4850.0
    meter_constant: float = 1.0
    billed_units: Optional[float] = 350.0
    billed_amount: Optional[float] = 1350.0
    arrears_amount: float = 0.0
    tariff_code: Optional[str] = "TN_DOMESTIC_LT1A_2024"
    historical_units: Optional[List[float]] = Field(default_factory=lambda: [280.0, 310.0, 320.0, 290.0, 330.0])


class BillVerifyResponse(BaseModel):
    tariff_result: TariffCalculationResult
    ml_score: float
    is_anomaly: bool
    explanation: ExplanationReport
    grievance_letter_text: str
    audit_log_id: int


class OCRUploadResponse(BaseModel):
    extracted_fields: Dict[str, Any]
    field_confidences: Dict[str, float]
    low_confidence_fields: List[str]
    needs_manual_verification: bool
    verification_hint: str


class PromoteRequest(BaseModel):
    version: int


class RetrainRequest(BaseModel):
    num_consumers: int = 100
    cycles_per_consumer: int = 10
