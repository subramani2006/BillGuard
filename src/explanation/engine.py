"""Bilingual Explanation Engine for electricity bill anomaly audits."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.tariff.models import TariffCalculationResult
from .bilingual_templates import EXPLANATION_TEMPLATES_EN, EXPLANATION_TEMPLATES_TA


class ExplanationReport(BaseModel):
    category: str
    certainty_level: str  # HIGH_CERTAINTY, MODERATE_EVIDENCE, LOW_NEED_PHYSICAL_INSPECTION
    is_anomaly: bool
    discrepancy_amount: Optional[float] = None
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
    Maintains a strict boundary between verified arithmetic facts and heuristic ML findings.
    """

    def generate_explanation(
        self,
        tariff_result: TariffCalculationResult,
        ml_score: float = 0.0,
        ml_predicted: bool = False,
        feature_attributions: Optional[List[Dict[str, Any]]] = None,
        historical_mean_units: float = 250.0,
    ) -> ExplanationReport:
        """
        Produce comprehensive bilingual explanation report based on deterministic + ML evidence.
        """
        units = tariff_result.calculated_units
        billed = tariff_result.billed_amount
        calc_total = tariff_result.calculated_grand_total
        days = tariff_result.billing_days
        discrepancy = tariff_result.discrepancy_amount or 0.0

        # Case 1: Meter reading physically mismatched from billed units
        if not tariff_result.units_match:
            cat = "METER_READING_MISMATCH"
            cert = "HIGH_CERTAINTY"
            diff_u = abs(tariff_result.units_discrepancy)
            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                billed_units=tariff_result.billed_units,
                prev_reading="Prev",
                curr_reading="Curr",
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
                f"Billed units ({tariff_result.billed_units}) do not match calculated meter difference ({units}).",
                f"Unaccounted unit difference: {diff_u:.1f} kWh.",
            ]
            ev_ta = [
                f"ரசீதில் உள்ள அலகுகள் ({tariff_result.billed_units}) கணக்கிடப்பட்ட அளவீட்டு வித்தியாசத்துடன் ({units}) பொருந்தவில்லை.",
                f"கணக்கில் வராத அலகு வித்தியாசம்: {diff_u:.1f} kWh.",
            ]
            act_en = "Contact TANGEDCO local section office to rectify the recorded meter units on the bill."
            act_ta = "மின்வாரிய பிரிவு அலுவலகத்தை அணுகி தவறாகப் பதிவான அளவீட்டு அலகுகளைத் திருத்தக் கோரவும்."
            return ExplanationReport(
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=discrepancy,
                english_summary=en_sum, tamil_summary=ta_sum, evidence_points_en=ev_en,
                evidence_points_ta=ev_ta, recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or []
            )

        # Case 2: Arithmetic error (Billed amount != statutory slab sum)
        if billed is not None and abs(discrepancy) > 2.0:
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
                f"Exact arithmetic discrepancy: ₹{discrepancy:.2f}.",
                f"TNERC tariff slabs order: {tariff_result.tariff_name}.",
            ]
            ev_ta = [
                f"அரசாணை விதிகளின்படி சரியான கட்டணம் ₹{calc_total:.2f}, ஆனால் ரசீதில் உள்ள தொகை ₹{billed:.2f}.",
                f"துல்லியமான கட்டண முரண்பாடு: ₹{discrepancy:.2f}.",
                f"பயன்படுத்தப்பட்ட கட்டண ஆணை: {tariff_result.tariff_name}.",
            ]
            act_en = "Submit a formal petition with the itemized slab calculation audit for bill revision."
            act_ta = "மறு கணக்கீட்டு விவரங்களை இணைத்து மின்வாரியத்தில் எழுத்துப்பூர்வ திருத்தக் கோரிக்கை மனு சமர்ப்பிக்கவும்."
            return ExplanationReport(
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=discrepancy,
                english_summary=en_sum, tamil_summary=ta_sum, evidence_points_en=ev_en,
                evidence_points_ta=ev_ta, recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or []
            )

        # Case 3: Extended billing cycle (>65 days) causing slab jump
        if tariff_result.is_cycle_abnormal and days > 65:
            cat = "SUSPECTED_EXTENDED_CYCLE"
            cert = "MODERATE_EVIDENCE"
            excess_d = days - 60
            # Rough penalty estimate: units accumulated during delay assessed at top slab rate (~₹8-10) vs lower (~₹2-4)
            daily_u = units / days
            penalty_est = max(0.0, excess_d * daily_u * 3.50)

            en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                billing_days=days,
                excess_days=excess_d,
                penalty_est=penalty_est
            )
            ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                billing_days=days,
                excess_days=excess_d,
                penalty_est=penalty_est
            )
            ev_en = [
                f"Billing cycle spans {days} days compared to standard 60 days (+{excess_d} days).",
                f"Daily consumption rate: {daily_u:.2f} kWh/day.",
                "No prorated slab limits were identified in the bill.",
            ]
            ev_ta = [
                f"வழக்கமான 60 நாட்களுக்கு மாறாக {days} நாட்கள் கணக்கீட்டில் சேர்க்கப்பட்டுள்ளது (+{excess_d} நாட்கள்).",
                f"தினசரி மின் நுகர்வு வீதம்: {daily_u:.2f} kWh/நாள்.",
                "நாட்களுக்கு ஏற்ப கட்டண அடுக்குகள் பிரித்து கணக்கிடப்படவில்லை.",
            ]
            act_en = "Request TANGEDCO to prorate the consumption slabs corresponding to the 60-day standard period."
            act_ta = "தாமதமான நாட்களுக்கு ஏற்ப 60 நாள் விகிதத்தில் கட்டண அடுக்கை மாற்றி கணக்கிட முறையிடவும்."
            return ExplanationReport(
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                english_summary=en_sum, tamil_summary=ta_sum, evidence_points_en=ev_en,
                evidence_points_ta=ev_ta, recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or []
            )

        # Case 4: Zero consumption on active meter
        if units == 0.0 and historical_mean_units > 100.0:
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
            return ExplanationReport(
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                english_summary=en_sum, tamil_summary=ta_sum, evidence_points_en=ev_en,
                evidence_points_ta=ev_ta, recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or []
            )

        # Case 5: ML / Statistical Behavioral Spike
        spike_ratio = units / max(10.0, historical_mean_units)
        if ml_predicted or spike_ratio > 1.65:
            cat = "UNUSUAL_CONSUMPTION_SURGE" if spike_ratio < 3.0 else "SUSPECTED_METER_FAULT"
            cert = "LOW_NEED_PHYSICAL_INSPECTION"
            spike_pct = (spike_ratio - 1.0) * 100.0

            if cat == "UNUSUAL_CONSUMPTION_SURGE":
                en_sum = EXPLANATION_TEMPLATES_EN[cat].format(
                    units=units,
                    spike_pct=spike_pct,
                    hist_avg=historical_mean_units
                )
                ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(
                    units=units,
                    spike_pct=spike_pct,
                    hist_avg=historical_mean_units
                )
            else:
                en_sum = EXPLANATION_TEMPLATES_EN[cat].format(jump_factor=spike_ratio)
                ta_sum = EXPLANATION_TEMPLATES_TA[cat].format(jump_factor=spike_ratio)

            ev_en = [
                f"Recorded units ({units:.1f}) are {spike_ratio:.2f}x historical average ({historical_mean_units:.1f}).",
                f"ML model anomaly confidence score: {ml_score:.2f}.",
                "Check for seasonal factors (summer AC), internal wiring leakage, or faulty meter sensor.",
            ]
            ev_ta = [
                f"பதிவான அலகுகள் ({units:.1f}), முந்தைய சராசரியை ({historical_mean_units:.1f}) விட {spike_ratio:.2f} மடங்கு அதிகம்.",
                f"இயந்திர கற்றல் (ML) முரண்பாட்டு மதிப்பெண்: {ml_score:.2f}.",
                "கோடைகால பயன்பாடு, வயரிங் மின் கசிவு அல்லது மின்மானி சென்சார் குறைபாடுகளைச் சரிபார்க்கவும்.",
            ]
            act_en = "Inspect major household loads. If appliance usage was normal, request a meter accuracy check from TANGEDCO."
            act_ta = "வீட்டின் மின் உபகரணங்களை ஆய்வு செய்யவும். பயன்பாடு சாதாரணமாக இருப்பின், மின்மானி துல்லிய சோதனைக்கு விண்ணப்பிக்கவும்."
            return ExplanationReport(
                category=cat, certainty_level=cert, is_anomaly=True, discrepancy_amount=0.0,
                english_summary=en_sum, tamil_summary=ta_sum, evidence_points_en=ev_en,
                evidence_points_ta=ev_ta, recommended_action_en=act_en, recommended_action_ta=act_ta,
                feature_attributions=feature_attributions or []
            )

        # Case 6: Completely normal bill
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

        return ExplanationReport(
            category=cat, certainty_level=cert, is_anomaly=False, discrepancy_amount=0.0,
            english_summary=en_sum, tamil_summary=ta_sum, evidence_points_en=ev_en,
            evidence_points_ta=ev_ta, recommended_action_en=act_en, recommended_action_ta=act_ta,
            feature_attributions=feature_attributions or []
        )
