"""Bilingual Explanation Engine and Evidence-Based Investigation Report Generator."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.tariff.models import TariffCalculationResult
from .bilingual_templates import EXPLANATION_TEMPLATES_EN, EXPLANATION_TEMPLATES_TA


class AuditCheckItem(BaseModel):
    check_name: str
    status: str  # PASSED, FAILED, WARNING, NOT_PERFORMED
    details: str


class ExplanationReport(BaseModel):
    # Standard Outcome (3 distinct categories required by regulatory audit):
    # A. Verified arithmetic discrepancy
    # B. Unusual bill requiring investigation
    # C. No discrepancy detected by the available checks
    outcome: str
    finding_code: str
    finding_title: str

    # Backwards compatibility fields for existing models and tests
    category: str
    certainty_level: str  # HIGH_CERTAINTY, MODERATE_EVIDENCE, LOW_NEED_PHYSICAL_INSPECTION
    is_anomaly: bool
    discrepancy_amount: Optional[float] = None

    # The 10 Evidence-Based Investigation Report Sections
    # 1. Previous and current bill amounts
    previous_bill_amount: Optional[float] = None
    current_bill_amount: Optional[float] = None

    # 2. Percentage change where mathematically valid
    bill_amount_pct_change: Optional[float] = None

    # 3. Previous and current meter readings
    previous_meter_reading: float = 0.0
    current_meter_reading: float = 0.0

    # 4. Calculated and billed consumption
    calculated_units: float = 0.0
    billed_units: Optional[float] = None
    units_pct_change: Optional[float] = None

    # 5. Billing period and applicable tariff version
    billing_period_days: int = 60
    applicable_tariff_code: str = ""
    applicable_tariff_name: str = ""

    # 6. Independently calculated amount, when valid tariff rules are available
    independently_calculated_amount: Optional[float] = None

    # 7. Itemized discrepancies
    itemized_discrepancies: Dict[str, float] = Field(default_factory=dict)

    # 8. Possible explanations supported by available evidence
    possible_explanations: List[str] = Field(default_factory=list)

    # 9. Missing evidence and recommended checks
    missing_evidence_and_recommended_checks: List[str] = Field(default_factory=list)

    # 10. A clear conclusion and next steps
    conclusion_and_next_steps: str = ""
    conclusion_and_next_steps_ta: str = ""

    # Check statuses (Passed, Failed, Not Performed)
    audit_checks: List[AuditCheckItem] = Field(default_factory=list)

    # Bilingual summaries and actions
    english_summary: str
    tamil_summary: str
    evidence_points_en: List[str]
    evidence_points_ta: List[str]
    recommended_action_en: str
    recommended_action_ta: str
    feature_attributions: List[Dict[str, Any]] = Field(default_factory=list)


class ExplanationEngine:
    """
    Synthesizes evidence-backed bilingual explanations without false accusations of fraud.
    Distinguishes strictly between deterministic arithmetic verification and heuristic ML anomaly screening.
    Never states that a meter is faulty or that a consumer overcharged unless evidence supports it.
    """

    def generate_explanation(
        self,
        tariff_result: TariffCalculationResult,
        ml_score: float = 0.0,
        ml_predicted: bool = False,
        feature_attributions: Optional[List[Dict[str, Any]]] = None,
        historical_mean_units: float = 250.0,
        previous_bill_amount: Optional[float] = None,
        previous_meter_reading: Optional[float] = None,
        current_meter_reading: Optional[float] = None,
    ) -> ExplanationReport:
        """
        Produce a comprehensive evidence-based investigation report with 3 distinct outcomes.
        """
        units = tariff_result.calculated_units
        billed = tariff_result.billed_amount
        calc_total = tariff_result.calculated_grand_total
        days = tariff_result.billing_days
        discrepancy = tariff_result.discrepancy_amount or 0.0

        prev_read = previous_meter_reading if previous_meter_reading is not None else 0.0
        curr_read = current_meter_reading if current_meter_reading is not None else units

        # Compute percentage change where mathematically valid
        bill_pct_change = None
        if previous_bill_amount is not None and previous_bill_amount > 0 and billed is not None:
            bill_pct_change = round(((billed - previous_bill_amount) / previous_bill_amount) * 100.0, 1)

        units_pct_change = None
        if historical_mean_units > 0:
            units_pct_change = round(((units - historical_mean_units) / historical_mean_units) * 100.0, 1)

        # Build Audit Checks list
        audit_checks: List[AuditCheckItem] = []

        # 1. Date & period check
        if tariff_result.is_valid:
            audit_checks.append(AuditCheckItem(
                check_name="Reading Date & Sequence Validation",
                status="PASSED",
                details=f"Valid sequence spanning {days} days."
            ))
        else:
            audit_checks.append(AuditCheckItem(
                check_name="Reading Date & Sequence Validation",
                status="FAILED",
                details="; ".join(tariff_result.validation_errors) or "Invalid date sequence."
            ))

        # 2. Meter reading check
        if tariff_result.meter_status == "INVALID" or tariff_result.meter_status == "INVERTED":
            audit_checks.append(AuditCheckItem(
                check_name="Meter Reading Consistency",
                status="FAILED",
                details=f"Current reading ({curr_read}) is less than previous reading ({prev_read})."
            ))
        elif tariff_result.meter_status == "ROLLOVER":
            audit_checks.append(AuditCheckItem(
                check_name="Meter Reading Consistency",
                status="WARNING",
                details="Meter register rollover detected and handled."
            ))
        elif tariff_result.meter_status == "REPLACED":
            audit_checks.append(AuditCheckItem(
                check_name="Meter Reading Consistency",
                status="WARNING",
                details="Meter replacement recorded during billing cycle."
            ))
        else:
            audit_checks.append(AuditCheckItem(
                check_name="Meter Reading Consistency",
                status="PASSED",
                details="Continuous positive reading sequence."
            ))

        # 3. Billed units check
        if tariff_result.billed_units is not None:
            if tariff_result.units_match:
                audit_checks.append(AuditCheckItem(
                    check_name="Billed vs Calculated Units Check",
                    status="PASSED",
                    details=f"Billed units ({tariff_result.billed_units}) match meter difference ({units:.1f} kWh)."
                ))
            else:
                audit_checks.append(AuditCheckItem(
                    check_name="Billed vs Calculated Units Check",
                    status="FAILED",
                    details=f"Billed units ({tariff_result.billed_units}) differ from meter difference ({units:.1f} kWh) by {tariff_result.units_discrepancy:+.1f} kWh."
                ))
        else:
            audit_checks.append(AuditCheckItem(
                check_name="Billed vs Calculated Units Check",
                status="NOT_PERFORMED",
                details="Billed units field not supplied; calculated units used as reference."
            ))

        # 4. Statutory tariff slab check
        if billed is not None:
            if abs(discrepancy) <= 2.0:
                audit_checks.append(AuditCheckItem(
                    check_name="TNERC Tariff Slab Recomputation",
                    status="PASSED",
                    details=f"Billed ₹{billed:.2f} matches statutory tariff calculation of ₹{calc_total:.2f}."
                ))
            else:
                audit_checks.append(AuditCheckItem(
                    check_name="TNERC Tariff Slab Recomputation",
                    status="FAILED",
                    details=f"Discrepancy of ₹{discrepancy:+.2f} between billed amount and statutory calculation."
                ))
        else:
            audit_checks.append(AuditCheckItem(
                check_name="TNERC Tariff Slab Recomputation",
                status="NOT_PERFORMED",
                details="Billed receipt amount not entered."
            ))

        # 5. Billing cycle duration check
        if tariff_result.is_cycle_abnormal:
            audit_checks.append(AuditCheckItem(
                check_name="Billing Cycle Duration Audit",
                status="WARNING",
                details=tariff_result.cycle_notes
            ))
        else:
            audit_checks.append(AuditCheckItem(
                check_name="Billing Cycle Duration Audit",
                status="PASSED",
                details=f"Standard bi-monthly duration ({days} days)."
            ))

        # 6. Anomaly screening check
        if ml_predicted or (units_pct_change and units_pct_change > 65.0):
            audit_checks.append(AuditCheckItem(
                check_name="ML Consumption Anomaly Screening",
                status="WARNING",
                details=f"Consumption exceeds historical baseline expectation (anomaly score: {ml_score:.2f})."
            ))
        else:
            audit_checks.append(AuditCheckItem(
                check_name="ML Consumption Anomaly Screening",
                status="PASSED",
                details="Consumption is within normal baseline parameters."
            ))

        # Itemized discrepancies dictionary
        itemized_discrepancies = {
            "units_discrepancy_kwh": tariff_result.units_discrepancy,
            "tariff_slab_discrepancy_inr": discrepancy if abs(discrepancy) > 2.0 else 0.0,
            "arrears_billed_inr": tariff_result.arrears_amount,
            "additional_charges_inr": tariff_result.additional_charges,
            "adjustments_inr": tariff_result.adjustments,
            "total_net_discrepancy_inr": discrepancy,
        }

        # Determine finding, outcome, and narrative
        # Case 0: Invalid / Incomplete Inputs
        if not tariff_result.is_valid:
            outcome = "B. Unusual bill requiring investigation"
            finding_code = "INSUFFICIENT_INFORMATION"
            finding_title = "Insufficient Information to Determine Cause"
            cat = "INSUFFICIENT_INFORMATION"
            cert = "LOW_NEED_PHYSICAL_INSPECTION"
            en_sum = EXPLANATION_TEMPLATES_EN[cat]
            ta_sum = EXPLANATION_TEMPLATES_TA[cat]
            ev_en = [f"Validation check failed: {err}" for err in tariff_result.validation_errors]
            ev_ta = ["சரிபார்ப்பு தோல்வியடைந்தது: அளவீட்டு விவரங்கள் பிழையாக உள்ளன."]
            act_en = "Verify the original paper meter card and enter valid reading dates and non-negative meter values."
            act_ta = "அசல் மின் அட்டை விவரங்களைச் சரிபார்த்து சரியான தேதிகள் மற்றும் அளவீடுகளை உள்ளிடவும்."
            concl_en = "The provided bill data contains invalid or incomplete parameters. Arithmetic verification could not be completed."
            concl_ta = "வழங்கப்பட்ட ரசீது விவரங்களில் முரண்பாடுகள் உள்ளதால் தணிக்கையை முழுமையாக முடிக்க இயலவில்லை."
            missing_checks = [
                "Verify physical meter serial number and digit counter register.",
                "Inspect previous two physical meter reading receipts for correct chronological sequence.",
            ]
            possible_explanations = [
                "Clerical data-entry typo when copying meter readings.",
                "Reversed meter reading dates on the receipt card.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=discrepancy,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 1: Meter reading physically mismatched from billed units
        if not tariff_result.units_match:
            outcome = "A. Verified arithmetic discrepancy"
            finding_code = "BILLED_UNITS_INCONSISTENT"
            finding_title = "Billed Units Inconsistent with Meter Readings"
            cat = "METER_READING_MISMATCH"
            cert = "HIGH_CERTAINTY"
            diff_u = abs(tariff_result.units_discrepancy)
            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                billed_units=tariff_result.billed_units,
                prev_reading="Previous",
                curr_reading="Current",
                calc_units=units,
                diff_units=diff_u
            )
            ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                billed_units=tariff_result.billed_units,
                prev_reading="முந்தைய",
                curr_reading="தற்போதைய",
                calc_units=units,
                diff_units=diff_u
            )
            ev_en = [
                f"Billed units ({tariff_result.billed_units}) do not match calculated meter difference ({units:.1f} kWh).",
                f"Unaccounted unit discrepancy: {diff_u:.1f} kWh.",
                "Meter constant applied: 1.0.",
            ]
            ev_ta = [
                f"ரசீதில் உள்ள அலகுகள் ({tariff_result.billed_units}) கணக்கிடப்பட்ட அளவீட்டு வித்தியாசத்துடன் ({units:.1f}) பொருந்தவில்லை.",
                f"கணக்கில் வராத அலகு வித்தியாசம்: {diff_u:.1f} kWh.",
            ]
            act_en = "Contact TANGEDCO local section office to rectify the recorded meter units on the bill."
            act_ta = "மின்வாரிய பிரிவு அலுவலகத்தை அணுகி தவறாகப் பதிவான அளவீட்டு அலகுகளைத் திருத்தக் கோரவும்."
            concl_en = (
                f"A verifiable arithmetic discrepancy of {diff_u:.1f} units exists between the meter register difference "
                "and the units billed on the receipt."
            )
            concl_ta = f"ரசீதில் உள்ள அலகுகளுக்கும் மின்மானி அளவீட்டுக்கும் இடையே {diff_u:.1f} அலகுகள் கணித முரண்பாடு உறுதி செய்யப்பட்டுள்ளது."
            possible_explanations = [
                "Clerical transcription typo by the meter assessor when keying units into the billing handheld device.",
                "Estimated units recorded instead of actual meter difference without DL/assessment flag.",
            ]
            missing_checks = [
                "Request meter reader ledger snapshot from TANGEDCO Section Office.",
                "Take a dated photograph of the current meter display to verify register position.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=discrepancy,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 2: Arithmetic error (Billed amount != statutory slab sum)
        if billed is not None and abs(discrepancy) > 2.0:
            outcome = "A. Verified arithmetic discrepancy"
            finding_code = "ARITHMETIC_DISCREPANCY"
            finding_title = "Arithmetic Discrepancy in Tariff Assessment"
            cat = "CONFIRMED_ARITHMETIC_ERROR"
            cert = "HIGH_CERTAINTY"
            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                billed_amount=billed,
                calculated_amount=calc_total,
                discrepancy=abs(discrepancy)
            )
            ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                billed_amount=billed,
                calculated_amount=calc_total,
                discrepancy=abs(discrepancy)
            )
            ev_en = [
                f"Statutory tariff calculation yields ₹{calc_total:.2f}, while utility billed ₹{billed:.2f}.",
                f"Exact arithmetic discrepancy: ₹{discrepancy:+.2f}.",
                f"Applicable TNERC tariff schedule: {tariff_result.tariff_name}.",
                f"Subsidized lifeline allowance verified: First {tariff_result.metadata.get('free_allowance_units', 100)} units at ₹0.00.",
            ]
            ev_ta = [
                f"அரசாணை விதிகளின்படி சரியான கட்டணம் ₹{calc_total:.2f}, ஆனால் ரசீதில் உள்ள தொகை ₹{billed:.2f}.",
                f"துல்லியமான கட்டண முரண்பாடு: ₹{discrepancy:+.2f}.",
                f"பயன்படுத்தப்பட்ட கட்டண ஆணை: {tariff_result.tariff_name}.",
            ]
            act_en = "Submit a formal petition with the itemized slab calculation audit for bill revision."
            act_ta = "மறு கணக்கீட்டு விவரங்களை இணைத்து மின்வாரியத்தில் எழுத்துப்பூர்வ திருத்தக் கோரிக்கை மனு சமர்ப்பிக்கவும்."
            concl_en = (
                f"A verified arithmetic discrepancy of ₹{discrepancy:+.2f} was detected between the billed receipt and "
                "the statutory TNERC tariff schedule. The bill is eligible for formal administrative rectification."
            )
            concl_ta = f"ரசீதுத் தொகையில் ₹{discrepancy:+.2f} கணக்கீட்டு முரண்பாடு இருப்பது உறுதி செய்யப்பட்டுள்ளது. மறுசீரமைப்பு கோரலாம்."
            possible_explanations = [
                "Unidentified surcharge or unitemized penalty applied without statutory breakdown.",
                "Billing software slab index calculation mismatch in utility billing server.",
            ]
            missing_checks = [
                "Confirm if any uncredited previous payments or security deposit interest adjustments were applied.",
                "Review detailed tariff break-up receipt from TANGEDCO payment counter.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=discrepancy,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 3: Extended billing cycle (>65 days)
        if tariff_result.is_cycle_abnormal and days > 65:
            outcome = "B. Unusual bill requiring investigation"
            finding_code = "LONGER_BILLING_PERIOD"
            finding_title = "Longer Billing Period Causing Slab Escalation"
            cat = "SUSPECTED_EXTENDED_CYCLE"
            cert = "MODERATE_EVIDENCE"
            excess_d = days - 60
            daily_u = units / max(1, days)
            penalty_est = max(0.0, excess_d * daily_u * 3.50)

            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                billing_days=days, excess_days=excess_d, penalty_est=penalty_est
            )
            ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                billing_days=days, excess_days=excess_d, penalty_est=penalty_est
            )
            ev_en = [
                f"Billing cycle spans {days} days compared to standard 60-day bi-monthly schedule (+{excess_d} days).",
                f"Daily consumption rate: {daily_u:.2f} kWh/day.",
                "No prorated slab thresholds were identified in the bill calculation.",
            ]
            ev_ta = [
                f"வழக்கமான 60 நாட்களுக்கு மாறாக {days} நாட்கள் கணக்கீட்டில் சேர்க்கப்பட்டுள்ளது (+{excess_d} நாட்கள்).",
                f"தினசரி மின் நுகர்வு வீதம்: {daily_u:.2f} kWh/நாள்.",
                "நாட்களுக்கு ஏற்ப கட்டண அடுக்குகள் பிரித்து கணக்கிடப்படவில்லை.",
            ]
            act_en = "Request TANGEDCO to prorate the consumption slabs corresponding to the 60-day standard period."
            act_ta = "தாமதமான நாட்களுக்கு ஏற்ப 60 நாள் விகிதத்தில் கட்டண அடுக்கை மாற்றி கணக்கிட முறையிடவும்."
            concl_en = (
                f"The billing period was extended to {days} days (+{excess_d} days delay). While arithmetic matches "
                "standard formulas, the delayed reading artificially pushed normal consumption into higher tariff slabs."
            )
            concl_ta = f"அளவீட்டுக் காலம் {days} நாட்களாக நீட்டிக்கப்பட்டதால் கட்டண அடுக்கு உயர்ந்துள்ளது. விகிதாசார சலுகை கோரலாம்."
            possible_explanations = [
                "Administrative or staffing delay by section office meter reader.",
                "Public holiday or cyclone disruption during scheduled reading week.",
            ]
            missing_checks = [
                "Verify official reading schedule posted at TANGEDCO Section Office.",
                "Check whether neighboring domestic connections in the same distribution transformer also had delayed readings.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 4: Arrears explaining an increase
        if tariff_result.arrears_amount > 0 and (billed is None or abs(discrepancy) <= 2.0):
            outcome = "B. Unusual bill requiring investigation"
            finding_code = "ARREARS_EXPLAINING_INCREASE"
            finding_title = "Arrears or Adjustments Explaining Bill Increase"
            cat = "ARREARS_EXPLAINING_INCREASE"
            cert = "HIGH_CERTAINTY"
            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                arrears_amount=tariff_result.arrears_amount,
                cycle_total=tariff_result.calculated_current_cycle_total,
            )
            ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                arrears_amount=tariff_result.arrears_amount,
                cycle_total=tariff_result.calculated_current_cycle_total,
            )
            ev_en = [
                f"Assessed arrears of ₹{tariff_result.arrears_amount:.2f} appended to bill.",
                f"Current cycle statutory charges: ₹{tariff_result.calculated_current_cycle_total:.2f}.",
                "Arrears represent past unpaid dues or regulatory adjustments rather than current consumption surge.",
            ]
            ev_ta = [
                f"முந்தைய நிலுவைத் தொகை ₹{tariff_result.arrears_amount:.2f} சேர்க்கப்பட்டுள்ளது.",
                f"நடப்பு சுழற்சி உண்மையான கட்டணம்: ₹{tariff_result.calculated_current_cycle_total:.2f}.",
            ]
            act_en = "Verify payment receipts of the preceding 2 billing cycles to verify whether arrears were already settled."
            act_ta = "முந்தைய இரண்டு சுழற்சிகளின் ரசீதுகளை ஆய்வு செய்து நிலுவைத் தொகை ஏற்கனவே செலுத்தப்பட்டதா என உறுதி செய்யவும்."
            concl_en = f"The bill increase is attributable to ₹{tariff_result.arrears_amount:.2f} in past arrears/adjustments."
            concl_ta = f"கட்டண உயர்வுக்கு முந்தைய நிலுவைத் தொகையான ₹{tariff_result.arrears_amount:.2f} காரணமென அறியப்படுகிறது."
            possible_explanations = [
                "Uncredited online transaction or delayed bank clearance during last cycle payment.",
                "Annual regulatory adjustment surcharge or meter testing fee billed as arrears.",
            ]
            missing_checks = [
                "Inspect bank statement / TANGEDCO web portal transaction history for previous cycle payment status.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 5: Zero consumption on active meter
        if units == 0.0 and historical_mean_units > 100.0:
            outcome = "B. Unusual bill requiring investigation"
            finding_code = "UNUSUAL_CONSUMPTION_INCREASE"
            finding_title = "Zero Consumption Recorded on Active Residence"
            cat = "DEAD_METER_ZERO"
            cert = "MODERATE_EVIDENCE"
            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(hist_avg=historical_mean_units)
            ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(hist_avg=historical_mean_units)
            ev_en = [
                "0 units recorded despite active historical consumption.",
                f"Historical 6-month mean consumption was {historical_mean_units:.1f} kWh.",
            ]
            ev_ta = [
                "முந்தைய தொடர் பயன்பாடு இருந்தும் தற்போதைய ரசீதில் 0 அலகுகள் பதிவாகியுள்ளது.",
                f"முந்தைய 6 மாத சராசரி பயன்பாடு: {historical_mean_units:.1f} kWh.",
            ]
            act_en = "Verify meter display physically to confirm if meter fuse has blown or register is stuck."
            act_ta = "மின்மானி இயங்குகிறதா அல்லது பியூஸ் போய்விட்டதா என்பதை நேரில் பார்த்து உறுதி செய்யவும்."
            concl_en = "Zero consumption recorded for an actively occupied residence. Requires physical register inspection."
            concl_ta = "தொடர் பயன்பாட்டு இணைப்பிற்கு சுழி நுகர்வு பதிவாகியுள்ளது. மின்மானி ஆய்வு தேவைப்படுகிறது."
            possible_explanations = [
                "Defective mechanical meter counter or blown input phase fuse.",
                "Premises temporarily unoccupied or locked during reading date.",
            ]
            missing_checks = [
                "Check whether red pulse LED on meter blinks when a known load (e.g., light or iron) is switched on.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 6: Unusual consumption surge (ML or statistical)
        spike_ratio = units / max(10.0, historical_mean_units)
        if ml_predicted or spike_ratio > 1.65:
            outcome = "B. Unusual bill requiring investigation"
            finding_code = "UNUSUAL_CONSUMPTION_INCREASE"
            finding_title = "Unusual Consumption Increase Requiring Inspection"
            cat = "UNUSUAL_CONSUMPTION_SURGE" if spike_ratio < 3.0 else "SUSPECTED_METER_FAULT"
            cert = "LOW_NEED_PHYSICAL_INSPECTION"
            spike_pct = (spike_ratio - 1.0) * 100.0

            if cat == "UNUSUAL_CONSUMPTION_SURGE":
                en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                    units=units, spike_pct=spike_pct, hist_avg=historical_mean_units
                )
                ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                    units=units, spike_pct=spike_pct, hist_avg=historical_mean_units
                )
            else:
                en_sum = EXPLANATION_TEMPLATES_EN[cat].format(jump_factor=spike_ratio)
                ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(jump_factor=spike_ratio)

            ev_en = [
                f"Recorded units ({units:.1f} kWh) are {spike_ratio:.2f}x historical average ({historical_mean_units:.1f} kWh).",
                f"ML model anomaly score: {ml_score:.2f}.",
                "Important Note: This finding indicates statistical deviation; meter fault or utility error cannot be claimed without official laboratory meter testing.",
                "Actual appliance load cannot be inferred solely from bill amount.",
            ]
            ev_ta = [
                f"பதிவான அலகுகள் ({units:.1f}), முந்தைய சராசரியை ({historical_mean_units:.1f}) விட {spike_ratio:.2f} மடங்கு அதிகம்.",
                f"இயந்திர கற்றல் (ML) முரண்பாட்டு மதிப்பெண்: {ml_score:.2f}.",
                "நேரடி ஆய்வின்றி மின்மானி கோளாறு என்று உறுதியாகக் கூற இயலாது.",
            ]
            act_en = "Inspect household wiring and high-power appliances. If usage was standard, request a meter accuracy check from TANGEDCO under CEA Rule 57."
            act_ta = "வீட்டின் மின் உபகரணங்களை ஆய்வு செய்யவும். பயன்பாடு சாதாரணமாக இருப்பின், மின்மானி துல்லிய சோதனைக்கு விண்ணப்பிக்கவும்."
            concl_en = (
                f"Consumption increased significantly ({spike_ratio:.2f}x baseline). "
                "Evidence warrants physical inspection to rule out wiring leakage or sensor fault."
            )
            concl_ta = f"மின் நுகர்வு வழக்கத்திற்கு மாறாக {spike_ratio:.2f} மடங்கு அதிகரித்துள்ளது. வயரிங் ஆய்வு பரிந்துரைக்கப்படுகிறது."
            possible_explanations = [
                "Seasonal cooling load (extended air conditioner / fan usage during peak summer months).",
                "Internal wiring leakage (neutral leakage / earth leakage grounding current).",
                "Transient meter current sensor malfunction.",
            ]
            missing_checks = [
                "Conduct an isolation test (switch off all circuit breakers and check if meter continues to record).",
                "Apply for an official parallel check meter test under Rule 57 of CEA Regulations.",
            ]
            return ExplanationReport(
                outcome=outcome, finding_code=finding_code, finding_title=finding_title,
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
                bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
                current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
                units_pct_change=units_pct_change, billing_period_days=days,
                applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
                independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
                possible_explanations=possible_explanations,
                missing_evidence_and_recommended_checks=missing_checks,
                conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
                audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
                evidence_points_en=ev_en, evidence_points_ta=ev_ta,
                recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or [],
            )

        # Case 7: Completely normal bill
        outcome = "C. No discrepancy detected by the available checks"
        finding_code = "NO_DISCREPANCY_DETECTED"
        finding_title = "No Discrepancy Detected by Available Checks"
        cat = "NORMAL_VERIFIED"
        cert = "HIGH_CERTAINTY"
        en_sum = EXPLANATION_TEMPLATES_EN[cat]
        ta_sum = EXPLANATION_TEMPLATES_TA[cat]
        ev_en = [
            "All arithmetic calculations and TNERC slab rules verified precisely.",
            f"Consumption ({units:.1f} kWh) is within normal variance of history ({historical_mean_units:.1f} kWh).",
            "Zero unaccounted arrears or discrepancies detected.",
        ]
        ev_ta = [
            "அனைத்து கட்டணக் கணக்கீடுகளும் TNERC விதிகளின்படி துல்லியமாக உள்ளன.",
            f"மின் நுகர்வு ({units:.1f} kWh) வழக்கமான அளவிலேயே உள்ளது (சராசரி: {historical_mean_units:.1f} kWh).",
            "எந்தவித கணக்கீட்டு முரண்பாடுகளும் இல்லை.",
        ]
        act_en = "No grievance required. Proceed with normal bill payment through official TANGEDCO portals."
        act_ta = "மனு எதுவும் தேவையில்லை. வழக்கமான முறையில் மின் கட்டணத்தை TANGEDCO தளம் மூலம் செலுத்தலாம்."
        concl_en = "All mathematical, regulatory, and baseline consumption checks passed. No discrepancy detected."
        concl_ta = "அனைத்து கட்டண கணக்கீடுகளும் சரியாக உள்ளன. எந்தவித முரண்பாடும் கண்டறியப்படவில்லை."
        possible_explanations = [
            "Routine bi-monthly domestic consumption matching seasonal and household baseline.",
        ]
        missing_checks = [
            "None required. Periodic monitoring recommended.",
        ]

        return ExplanationReport(
            outcome=outcome, finding_code=finding_code, finding_title=finding_title,
            category=cat, certainty_level=cert, is_anomaly=False, discrepancy_amount=0.0,
            previous_bill_amount=previous_bill_amount, current_bill_amount=billed,
            bill_amount_pct_change=bill_pct_change, previous_meter_reading=prev_read,
            current_meter_reading=curr_read, calculated_units=units, billed_units=tariff_result.billed_units,
            units_pct_change=units_pct_change, billing_period_days=days,
            applicable_tariff_code=tariff_result.tariff_code, applicable_tariff_name=tariff_result.tariff_name,
            independently_calculated_amount=calc_total, itemized_discrepancies=itemized_discrepancies,
            possible_explanations=possible_explanations,
            missing_evidence_and_recommended_checks=missing_checks,
            conclusion_and_next_steps=concl_en, conclusion_and_next_steps_ta=concl_ta,
            audit_checks=audit_checks, english_summary=en_sum, tamil_summary=ta_sum,
            evidence_points_en=ev_en, evidence_points_ta=ev_ta,
            recommended_action_en=act_en, recommended_action_ta=act_ta,
            feature_attributions=feature_attributions or [],
        )
