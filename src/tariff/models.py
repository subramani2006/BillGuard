"""Data models for Tariff Configuration and Calculation Results."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TariffSlab(BaseModel):
    min_units: int
    max_units: Optional[int] = None
    rate_per_unit: float
    subsidy_per_unit: float = 0.0
    net_rate: float
    description: str = ""


class FixedChargesConfig(BaseModel):
    enabled: bool = False
    amount_per_cycle: float = 0.0
    notes: str = ""


class ElectricityTaxConfig(BaseModel):
    enabled: bool = False
    rate_percentage: float = 0.0
    threshold_units: int = 0
    notes: str = ""


class FppcaConfig(BaseModel):
    enabled: bool = False
    rate_per_unit: float = 0.0
    description: str = ""


class TariffConfig(BaseModel):
    tariff_code: str
    tariff_name: str
    tariff_category: str = "LT-1A Domestic"
    state: str = "Tamil Nadu"
    authority: str = "TNERC"
    order_reference: str
    effective_start_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    verification_status: str = "VERIFIED_OFFICIAL"
    billing_cycle_days: int = 60
    billing_cycle_type: str = "bi-monthly"
    free_allowance_units: int = 100
    free_allowance_description: str = ""
    subsidy_conditions: str = (
        "First 100 units free per bi-monthly cycle subsidized by Government of Tamil Nadu "
        "(G.O. Ms. No. 38, Energy Department). Applicable to single domestic connection per premises."
    )
    calculation_methodology: str = (
        "TNERC progressive telescopic slab assessment. Gross energy charge minus GoTN tariff subsidy, "
        "plus FPPCA fuel adjustment surcharge on billable units."
    )
    disclaimer: str = ""
    slabs: List[TariffSlab]
    fixed_charges: FixedChargesConfig
    electricity_tax: ElectricityTaxConfig
    fppca: FppcaConfig


class SlabBreakdown(BaseModel):
    slab_label: str
    units_in_slab: float
    gross_rate: float
    gross_amount: float
    subsidy_rate: float
    subsidy_amount: float
    net_rate: float
    net_amount: float


class BillInput(BaseModel):
    consumer_number: str
    consumer_name: Optional[str] = "Domestic Consumer"
    service_connection_no: Optional[str] = None
    billing_date: str = "2024-08-01"
    previous_reading_date: str = "2024-06-01"
    current_reading_date: str = "2024-08-01"
    previous_meter_reading: float
    current_meter_reading: float
    meter_constant: float = 1.0
    billed_units: Optional[float] = None
    billed_amount: Optional[float] = None
    billed_fppca: Optional[float] = None
    billed_fixed_charges: Optional[float] = None
    arrears_amount: float = 0.0
    additional_charges: float = 0.0
    adjustments: float = 0.0
    tariff_code: Optional[str] = None
    proration_applied: bool = False
    is_estimated_reading: bool = False
    meter_replaced: bool = False
    old_meter_final_reading: Optional[float] = None
    new_meter_initial_reading: Optional[float] = 0.0
    notes: Optional[str] = None


class TariffCalculationResult(BaseModel):
    tariff_code: str
    tariff_name: str
    tariff_category: str = "LT-1A Domestic"
    verification_status: str
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    meter_status: str = "NORMAL"  # NORMAL, ROLLOVER, REPLACED, ESTIMATED, INVERTED, INVALID
    calculated_units: float
    billed_units: Optional[float] = None
    units_match: bool = True
    units_discrepancy: float = 0.0
    billing_days: int
    standard_cycle_days: int
    is_cycle_abnormal: bool
    cycle_notes: str
    slab_breakdown: List[SlabBreakdown]
    energy_charge_gross: float
    subsidy_amount: float
    energy_charge_net: float
    fixed_charges: float
    electricity_tax: float
    fppca_amount: float
    calculated_current_cycle_total: float
    arrears_amount: float
    additional_charges: float = 0.0
    adjustments: float = 0.0
    calculated_grand_total: float
    billed_amount: Optional[float] = None
    discrepancy_amount: Optional[float] = None
    is_arithmetic_exact_match: bool
    discrepancy_category: str  # EXACT_MATCH, OVERBILLED, UNDERBILLED, METER_READING_MISMATCH, UNKNOWN, INVALID_INPUT
    effective_start_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    disclaimer: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
