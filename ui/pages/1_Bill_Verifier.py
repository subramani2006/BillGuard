import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for cloud deployment
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import io
import joblib
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from src.explanation.engine import ExplanationEngine
from src.explanation.grievance_generator import GrievanceGenerator
from src.ocr.extractor import BillOcrExtractor
from src.data_gen.ocr_image_generator import BillImageGenerator
from src.ml.features import BillFeatureExtractor
from src.ml.isolation_forest import BillIsolationForest
from src.ml.autoencoder import BillAutoencoderDetector
from src.mlops.registry import ModelRegistry
from src.mlops.audit_logger import AuditLogger, AuditEntry
from ui.utils import (
    format_currency,
    get_verdict_badge,
    get_certainty_badge,
    get_outcome_badge,
    get_check_status_badge,
    inject_custom_theme,
    render_hero_banner,
)

st.set_page_config(page_title="Bill Verifier | BillGuard", page_icon="📄", layout="wide")
inject_custom_theme()


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert any OCR or user field to float without throwing."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# Singletons
@st.cache_resource
def load_components():
    t_engine = TariffEngine()
    e_engine = ExplanationEngine()
    g_gen = GrievanceGenerator()
    reg = ModelRegistry()
    aud = AuditLogger()
    
    # Load trained feature extractor with fitted normalizer
    f_ext_p = "./models/feature_extractor.joblib"
    if os.path.exists(f_ext_p):
        try:
            f_ext = joblib.load(f_ext_p)
        except Exception:
            f_ext = BillFeatureExtractor()
    else:
        f_ext = BillFeatureExtractor()

    # Load OCR extractor
    ocr_p = "./models/ocr_crnn_scratch.pth"
    ocr = BillOcrExtractor(model_path=ocr_p if os.path.exists(ocr_p) else None)

    # Load Production ML Model
    prod = reg.get_production_model()
    ml_mod = None
    if prod and os.path.exists(prod.artifact_path):
        try:
            if "Autoencoder" in prod.model_name:
                ml_mod = BillAutoencoderDetector()
                ml_mod.load(prod.artifact_path)
            else:
                ml_mod = BillIsolationForest()
                ml_mod.load(prod.artifact_path)
        except Exception:
            ml_mod = None

    return t_engine, e_engine, g_gen, reg, aud, f_ext, ocr, ml_mod


tariff_engine, exp_engine, grievance_gen, registry, audit_logger, feat_extractor, ocr_extractor, ml_model = load_components()

# Grand Hero Banner
render_hero_banner(
    title="📄 Electricity Bill Verifier & Evidence Audit",
    subtitle="Statutory TNERC LT-1A Tariff Verification, from-scratch CRNN OCR reading extraction, and explainable multi-tier anomaly detection across 3 official regulatory outcomes.",
    tag_left="<span class='status-dot'></span> 8-STAGE AUDIT ENGINE ACTIVE",
    tag_right="🏛️ TNERC GAZETTE COMPLIANT",
)

# Available tariffs separated into Official and Simulated
tariffs = tariff_engine.list_available_tariffs()
official_tariffs = [t for t in tariffs if t.get("verification_status") == "VERIFIED_OFFICIAL"]
simulated_tariffs = [t for t in tariffs if t.get("verification_status") != "VERIFIED_OFFICIAL"]

tariff_options = {}
for t in official_tariffs:
    tariff_options[t["tariff_code"]] = f"🏛️ {t['tariff_name']} (Verified Official)"
for t in simulated_tariffs:
    tariff_options[t["tariff_code"]] = f"🔬 {t['tariff_name']} (Simulated Research)"

# Prominent Demonstration Presets Bar
st.markdown("### 🎯 Quick Demonstration Scenarios (Select to Test Instantly)")
preset = st.selectbox(
    "Choose a verified test case to demonstrate the audit engine:",
    [
        "Scenario A: Confirmed Arithmetic Overcharge (₹875 Error)",
        "Scenario B: Extended Billing Cycle (78 Days Delay)",
        "Scenario C: Verified Normal Lifeline (85 Units - 100% Free)",
        "Scenario D: Uncharacteristic Summer Spike (5x Surge)",
        "Scenario E: Inverted Reading / Typo (Data Validation Error)",
        "None (Custom Entry / OCR Upload)",
    ],
    index=0
)

# Preset defaults
defaults = {
    "consumer_no": "09-245-001-042",
    "name": "M. Senthilkumar",
    "section": "Anna Nagar West, Chennai",
    "prev_date": "2024-06-15",
    "curr_date": "2024-08-15",
    "prev_read": 4500.0,
    "curr_read": 4850.0,
    "billed_units": 350.0,
    "billed_amt": 1250.0,
    "arrears": 0.0,
    "tariff_code": "TN_DOMESTIC_LT1A_2024",
    "prev_bill_amt": 1180.0,
    "hist_units": [280.0, 310.0, 320.0, 290.0, 330.0],
}

if "Scenario A" in preset:
    defaults.update({
        "prev_read": 2000.0, "curr_read": 2300.0, "billed_units": 300.0,
        "billed_amt": 1450.0, "prev_bill_amt": 750.0,
        "hist_units": [280.0, 310.0, 320.0, 290.0, 330.0],
    })
elif "Scenario B" in preset:
    defaults.update({
        "prev_date": "2024-05-10", "curr_date": "2024-07-27",  # 78 days
        "prev_read": 3000.0, "curr_read": 3550.0, "billed_units": 550.0,
        "billed_amt": 2477.0, "prev_bill_amt": 1800.0,
        "hist_units": [400.0, 420.0, 410.0, 390.0, 430.0],
    })
elif "Scenario C" in preset:
    defaults.update({
        "prev_read": 1100.0, "curr_read": 1185.0, "billed_units": 85.0,
        "billed_amt": 0.0, "prev_bill_amt": 0.0,
        "hist_units": [75.0, 80.0, 85.0, 82.0, 88.0],
    })
elif "Scenario D" in preset:
    defaults.update({
        "prev_read": 5000.0, "curr_read": 6800.0, "billed_units": 1800.0,
        "billed_amt": 17177.0, "prev_bill_amt": 1180.0,
        "hist_units": [300.0, 320.0, 290.0, 310.0, 305.0],
    })
elif "Scenario E" in preset:
    defaults.update({
        "prev_read": 4850.0, "curr_read": 4200.0, "billed_units": 0.0,
        "billed_amt": 500.0, "prev_bill_amt": 1100.0,
        "hist_units": [280.0, 310.0, 320.0, 290.0, 330.0],
    })

tab1, tab2 = st.tabs(["📤 Step 1: Upload Bill Card Image (OCR)", "✍️ Step 1 Alternative: Manual Entry & Field Review"])

extracted_data = {}
ocr_res = None

with tab1:
    st.subheader("Upload Bill Image for From-Scratch OCR Extraction")
    st.info(
        "ℹ️ **From-Scratch CRNN OCR Architecture**: This OCR neural network is trained strictly from random weight "
        "initialization on locally synthesized TANGEDCO bill crops (no pretrained EasyOCR, Tesseract, or hosted AI APIs). "
        "All extracted fields are fully editable in the review form below before running the audit."
    )

    col_u1, col_u2 = st.columns([1, 1])

    with col_u1:
        uploaded_file = st.file_uploader("Upload Bill Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        btn_sample = st.button("🖼️ Generate & Load Sample TANGEDCO Bill")

        bill_img = None
        if uploaded_file is not None:
            bill_img = Image.open(uploaded_file).convert("RGB")
        elif btn_sample:
            img_gen = BillImageGenerator(seed=42)
            bill_img, _ = img_gen.generate_full_bill_image(
                consumer_no=defaults["consumer_no"],
                consumer_name=defaults["name"],
                bill_date=defaults["curr_date"],
                prev_date=defaults["prev_date"],
                prev_reading=defaults["prev_read"],
                curr_reading=defaults["curr_read"],
                units=defaults["billed_units"],
                net_amount=defaults["billed_amt"],
            )
            st.session_state["sample_loaded"] = True

        if bill_img is not None:
            st.image(bill_img, caption="Loaded Electricity Bill Card", use_container_width=True)

    with col_u2:
        if bill_img is not None:
            with st.spinner("Extracting bill fields with randomly-initialized CRNN model..."):
                ocr_res = ocr_extractor.extract_from_bill_image(bill_img)
                extracted_data = ocr_res["fields"]

            if ocr_res["needs_manual_verification"]:
                st.warning(
                    "⚠️ **Manual Review Required**: One or more fields had confidence below threshold (0.70) or were unreadable. "
                    "In accordance with project governance, BillGuard never guesses values silently. "
                    "Please verify or manually fill the fields below."
                )
            else:
                st.success("✅ **OCR Extraction Complete**: Please review the extracted values below.")

            # Display field confidence table
            conf_rows = []
            for k, val in extracted_data.items():
                conf = ocr_res["confidences"].get(k, 0.0)
                status_txt = "⚠️ Needs Verification" if conf < 0.70 or val is None else "✅ Recognized"
                conf_rows.append({
                    "Field": k.replace("_", " ").title(),
                    "Extracted Value": str(val) if val is not None else "[Unreadable - Enter Manually]",
                    "Confidence": f"{conf:.2f}",
                    "Status": status_txt
                })
            st.dataframe(pd.DataFrame(conf_rows), hide_index=True, use_container_width=True)

            # Crop previews for low-confidence fields
            if ocr_res.get("low_confidence_fields"):
                st.markdown("**Inspect Extracted Field Crops:**")
                crop_cols = st.columns(min(3, len(ocr_res["low_confidence_fields"])))
                for idx, f_name in enumerate(ocr_res["low_confidence_fields"][:3]):
                    with crop_cols[idx % 3]:
                        crop_img = ocr_res["crop_images"].get(f_name)
                        if crop_img:
                            st.image(crop_img, caption=f"{f_name} (conf: {ocr_res['confidences'].get(f_name, 0.0):.2f})", width=160)

with tab2:
    st.subheader("Step 2: Verify & Edit Bill Parameters")
    st.caption("Verify and confirm every parameter before submitting to the audit engine. You may override any extracted value.")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("##### 👤 Consumer Information")
    consumer_no = st.text_input("Consumer Number", value=str(extracted_data.get("consumer_number") or defaults["consumer_no"]))
    consumer_name = st.text_input("Consumer Name", value=defaults["name"])
    section_office = st.text_input("Section Office / Division", value=defaults["section"])
    prev_bill_amount = st.number_input("Previous Bill Amount (₹, for % change)", value=float(defaults["prev_bill_amt"]), min_value=0.0, step=50.0)

with col_f2:
    st.markdown("##### 📅 Billing Period & Tariff")
    prev_date = st.text_input("Previous Reading Date (YYYY-MM-DD)", value=str(extracted_data.get("previous_reading_date") or defaults["prev_date"]))
    curr_date = st.text_input("Current Reading Date (YYYY-MM-DD)", value=str(extracted_data.get("current_reading_date") or defaults["curr_date"]))
    tariff_selected = st.selectbox(
        "Applicable Tariff Schedule",
        list(tariff_options.keys()),
        format_func=lambda x: tariff_options[x],
        index=list(tariff_options.keys()).index(defaults["tariff_code"]) if defaults["tariff_code"] in tariff_options else 0
    )

    tariff_obj = tariff_engine.get_tariff(tariff_selected)
    t_status = getattr(tariff_obj, "verification_status", "VERIFIED_OFFICIAL")
    if t_status != "VERIFIED_OFFICIAL":
        disc_msg = getattr(tariff_obj, "disclaimer", "Simulation only — actual charges may differ. Verify the applicable official tariff.")
        st.caption(f"⚠️ *{disc_msg}*")
    else:
        ord_ref = getattr(tariff_obj, "order_reference", "TNERC Statutory Order")
        st.caption(f"🏛️ *TNERC Official Order: {ord_ref}*")

with col_f3:
    st.markdown("##### ⚡ Meter Readings & Billed Values")
    default_prev_r = safe_float(extracted_data.get("previous_meter_reading"), defaults["prev_read"])
    default_curr_r = safe_float(extracted_data.get("current_meter_reading"), defaults["curr_read"])
    default_b_units = safe_float(extracted_data.get("billed_units"), defaults["billed_units"])
    default_b_amt = safe_float(extracted_data.get("billed_amount"), defaults["billed_amt"])
    default_arrears = safe_float(extracted_data.get("arrears_amount"), defaults["arrears"])

    prev_read = st.number_input("Previous Meter Reading", value=float(default_prev_r), min_value=0.0, step=10.0)
    curr_read = st.number_input("Current Meter Reading", value=float(default_curr_r), min_value=0.0, step=10.0)
    billed_units = st.number_input("Billed Units on Receipt (kWh)", value=float(default_b_units), min_value=0.0, step=1.0)
    billed_amt = st.number_input("Billed Amount on Receipt (₹)", value=float(default_b_amt), min_value=0.0, step=10.0)
    arrears_amt = st.number_input("Assessed Past Arrears (₹)", value=float(default_arrears), min_value=0.0, step=10.0)

# Edge Cases & Special Conditions Expander
with st.expander("⚙️ Special Conditions & Adjustments (Estimated Readings, Meter Replacement, Surcharges)"):
    c_sp1, c_sp2 = st.columns(2)
    with c_sp1:
        is_estimated = st.checkbox("Estimated Reading / Door Locked (DL Assessment)", value=False)
        meter_replaced = st.checkbox("Meter Replacement mid-cycle", value=False)
        old_meter_final = None
        new_meter_initial = 0.0
        if meter_replaced:
            old_meter_final = st.number_input("Final Reading on Old Meter", value=5200.0, min_value=0.0)
            new_meter_initial = st.number_input("Initial Reading on New Meter", value=0.0, min_value=0.0)
    with c_sp2:
        additional_charges = st.number_input("Additional Regulatory Charges / Cess (₹)", value=0.0, min_value=0.0)
        adjustments = st.number_input("Adjustments / Credits / Rebates (₹)", value=0.0, min_value=0.0)

st.markdown("---")
btn_verify = st.button("🚀 Step 3: Run Deterministic Audit & ML Screening", type="primary", use_container_width=True)

# Session state management to keep audit results visible across tab clicks and PDF downloads
if "audit_executed" not in st.session_state:
    st.session_state["audit_executed"] = False

if btn_verify:
    st.session_state["audit_executed"] = True

if "last_preset" not in st.session_state:
    st.session_state["last_preset"] = preset
elif st.session_state["last_preset"] != preset:
    st.session_state["last_preset"] = preset
    if "None" not in preset:
        st.session_state["audit_executed"] = True
    else:
        st.session_state["audit_executed"] = False

should_run_audit = st.session_state["audit_executed"] or ("None" not in preset)

if should_run_audit:
    # 1. Run deterministic tariff verification
    bill_input = BillInput(
        consumer_number=consumer_no,
        consumer_name=consumer_name,
        previous_reading_date=prev_date,
        current_reading_date=curr_date,
        previous_meter_reading=prev_read,
        current_meter_reading=curr_read,
        billed_units=billed_units,
        billed_amount=billed_amt,
        arrears_amount=arrears_amt,
        additional_charges=additional_charges,
        adjustments=adjustments,
        tariff_code=tariff_selected,
        is_estimated_reading=is_estimated,
        meter_replaced=meter_replaced,
        old_meter_final_reading=old_meter_final,
        new_meter_initial_reading=new_meter_initial,
    )
    t_result = tariff_engine.calculate_bill(bill_input)

    # 2. Run ML Anomaly Screening
    hist_units = defaults.get("hist_units", [280.0, 310.0, 320.0, 290.0, 330.0])
    hist_mean = float(np.mean(hist_units))
    hist_records = [{"recorded_units": u, "billing_days": 60} for u in hist_units]
    curr_record = {
        "recorded_units": t_result.calculated_units,
        "billed_units": billed_units,
        "billing_days": t_result.billing_days,
        "cycle_in_year": 3,
        "calculated_grand_total": t_result.calculated_grand_total,
        "discrepancy_amount": t_result.discrepancy_amount or 0.0,
        "arrears_amount": arrears_amt,
    }
    feat_vec = feat_extractor.extract_features_from_history(curr_record, hist_records)
    feat_vec_norm = feat_extractor.transform(feat_vec.reshape(1, -1))

    ml_score = 0.15
    is_ml_anom = False
    attributions = []

    if ml_model is not None:
        try:
            scores = ml_model.score_samples(feat_vec_norm)
            preds = ml_model.predict(feat_vec_norm)
            ml_score = float(scores[0])
            is_ml_anom = bool(preds[0] == 1)
            attributions = ml_model.explain_instance(feat_vec_norm.flatten())
        except Exception:
            pass

    # 3. Generate Evidence-Based Investigation Report
    report = exp_engine.generate_explanation(
        tariff_result=t_result,
        ml_score=ml_score,
        ml_predicted=is_ml_anom,
        feature_attributions=attributions,
        historical_mean_units=hist_mean,
        previous_bill_amount=prev_bill_amount,
        previous_meter_reading=prev_read,
        current_meter_reading=curr_read,
    )

    # Persist in immutable audit log
    audit_logger.log_prediction(AuditEntry(
        consumer_number=consumer_no,
        billing_date=curr_date,
        previous_reading=prev_read,
        current_reading=curr_read,
        calculated_units=t_result.calculated_units,
        billed_units=billed_units,
        calculated_total=t_result.calculated_grand_total,
        billed_amount=billed_amt,
        discrepancy_amount=t_result.discrepancy_amount,
        is_anomaly=report.is_anomaly,
        anomaly_category=report.category,
        certainty_level=report.certainty_level,
        ml_score=ml_score,
    ))

    # =========================================================================
    # AUDIT RESULTS & INVESTIGATION REPORT
    # =========================================================================
    st.markdown("## 📊 Investigation Audit Report")

    # If input validation errors occurred
    if not t_result.is_valid:
        st.error("🚨 **Input Validation Failed**: One or more parameters are physically or chronologically invalid.")
        for err in t_result.validation_errors:
            st.markdown(f"- ❌ {err}")

    # Standard Outcome Banner (A, B, or C) with Matching Cyber Elevation
    outcome_box_class = (
        "outcome-box-a" if "A." in report.outcome or "arithmetic" in report.outcome.lower()
        else ("outcome-box-b" if "B." in report.outcome or "investigation" in report.outcome.lower() else "outcome-box-c")
    )
    outcome_badge = get_outcome_badge(report.outcome)
    cert_badge = get_certainty_badge(report.certainty_level)

    st.markdown(f"""
    <div class="{outcome_box_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
            <div>{outcome_badge}</div>
            <div>{cert_badge}</div>
        </div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
            Specific Finding: {report.finding_title}
        </div>
        <div style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.5;">
            {report.english_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Calculated Consumption",
            f"{t_result.calculated_units:.1f} kWh",
            delta=f"{t_result.units_discrepancy:+.1f} kWh vs Billed" if not t_result.units_match else "Exact Units Match"
        )
    with m2:
        st.metric("Statutory Calculated Total", format_currency(t_result.calculated_grand_total))
    with m3:
        pct_label = f"{report.bill_amount_pct_change:+.1f}% vs Prev" if report.bill_amount_pct_change is not None else None
        st.metric("Billed Amount", format_currency(billed_amt), delta=pct_label)
    with m4:
        disc = t_result.discrepancy_amount or 0.0
        st.metric(
            "Audit Discrepancy",
            format_currency(disc),
            delta=f"{disc:+.2f} ₹" if abs(disc) > 0.01 else "0.00 ₹",
            delta_color="inverse" if abs(disc) > 2.0 else "normal"
        )

    # Plain-English Executive Summary
    if abs(disc) > 2.0:
        st.error(
            f"🚨 **Overcharge Detected**: You were billed **{format_currency(billed_amt)}**, but statutory TNERC calculations show you should only be charged **{format_currency(t_result.calculated_grand_total)}**. Net overcharge: **{format_currency(abs(disc))}**."
        )
    elif abs(disc) <= 2.0 and t_result.is_valid and not report.is_anomaly:
        st.success(
            f"✅ **Bill Verified Correct**: Your billed amount of **{format_currency(billed_amt)}** precisely matches the official TNERC tariff calculation of **{format_currency(t_result.calculated_grand_total)}**."
        )

    # 3 Clean Core Tabs for Demo
    res_tab1, res_tab2, res_tab3 = st.tabs([
        "🧮 Statutory Slab Calculation",
        "🗣️ Bilingual Findings & Advice",
        "📝 Official Grievance Petition (PDF)"
    ])

    with res_tab1:
        st.markdown("#### 📋 Statutory Slab-by-Slab Tariff Breakdown (TNERC LT-1A)")
        slab_rows = []
        for s in t_result.slab_breakdown:
            slab_rows.append({
                "Slab (Units)": s.slab_label,
                "Units in Slab": s.units_in_slab,
                "Gross Rate (₹)": f"₹{s.gross_rate:.2f}",
                "Gross Total": format_currency(s.gross_amount),
                "GoTN Subsidy Rate": f"₹{s.subsidy_rate:.2f}",
                "Total Subsidy": format_currency(s.subsidy_amount),
                "Net Rate (₹)": f"₹{s.net_rate:.2f}",
                "Net Payable": format_currency(s.net_amount),
            })
        st.dataframe(pd.DataFrame(slab_rows), hide_index=True, use_container_width=True)

        st.markdown("#### 🧮 Itemized Discrepancy Reconciliation")
        disc_df = pd.DataFrame([
            {"Item": "Consumption Unit Discrepancy", "Discrepancy Value": f"{report.itemized_discrepancies.get('units_discrepancy_kwh', 0.0):+.1f} kWh"},
            {"Item": "Tariff Slab Assessment Discrepancy", "Discrepancy Value": format_currency(report.itemized_discrepancies.get('tariff_slab_discrepancy_inr', 0.0))},
            {"Item": "Arrears Appended to Bill", "Discrepancy Value": format_currency(report.itemized_discrepancies.get('arrears_billed_inr', 0.0))},
            {"Item": "Additional Charges / Taxes", "Discrepancy Value": format_currency(report.itemized_discrepancies.get('additional_charges_inr', 0.0))},
            {"Item": "Total Net Discrepancy (billed - calculated)", "Discrepancy Value": format_currency(report.itemized_discrepancies.get('total_net_discrepancy_inr', 0.0))},
        ])
        st.dataframe(disc_df, hide_index=True, use_container_width=True)

    with res_tab2:
        exp_tab_en, exp_tab_ta = st.tabs(["🇬🇧 English Findings", "🇮🇳 தமிழ் விளக்கம்"])
        with exp_tab_en:
            st.markdown(f"**Audit Summary:** {report.english_summary}")
            st.markdown("**Verifiable Evidence Points:**")
            for ev in report.evidence_points_en:
                st.markdown(f"- {ev}")
            st.success(f"📌 **Conclusion & Next Steps:** {report.conclusion_and_next_steps}")
            st.info(f"💡 **Recommended Action:** {report.recommended_action_en}")

        with exp_tab_ta:
            st.markdown(f"**தணிக்கை விளக்கம்:** {report.tamil_summary}")
            st.markdown("**உறுதிப்படுத்தப்பட்ட ஆதாரங்கள்:**")
            for ev in report.evidence_points_ta:
                st.markdown(f"- {ev}")
            st.success(f"📌 **முடிவு மற்றும் அடுத்த கட்ட நடவடிக்கை:** {report.conclusion_and_next_steps_ta}")
            st.info(f"💡 **பரிந்துரைக்கப்படும் நடவடிக்கை:** {report.recommended_action_ta}")

        if report.possible_explanations:
            st.markdown("---")
            st.markdown("##### 💡 Evidence-Supported Possible Explanations")
            for pe in report.possible_explanations:
                st.markdown(f"- {pe}")

    with res_tab3:
        st.markdown("#### 📝 Official TANGEDCO Representation & Grievance Petition")
        st.caption("A legally formal representation petition citing TNERC tariff orders and Tamil Nadu Electricity Distribution Code (Clause 11). Ready to download and submit.")

        petition_txt = grievance_gen.generate_complaint_text(
            consumer_number=consumer_no,
            consumer_name=consumer_name,
            section_office=section_office,
            tariff_result=t_result,
            explanation=report,
        )

        pdf_temp = f"./data/temp_grievance_{consumer_no.replace('-', '_')}.pdf"
        grievance_gen.generate_complaint_pdf(
            output_filepath=pdf_temp,
            consumer_number=consumer_no,
            consumer_name=consumer_name,
            section_office=section_office,
            tariff_result=t_result,
            explanation=report,
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if os.path.exists(pdf_temp):
                with open(pdf_temp, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "📄 Download Complaint-Ready PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"TANGEDCO_Representation_{consumer_no}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
        with col_d2:
            st.download_button(
                "📥 Download Petition as Text (.txt)",
                data=petition_txt,
                file_name=f"TANGEDCO_Representation_{consumer_no}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with st.expander("👁️ Preview Formal Representation Draft", expanded=False):
            st.text_area("Petition Draft", petition_txt, height=220)

        with st.expander("🏛️ Statutory Escalation Channels & Authorities"):
            st.markdown(f"""
            1. **Local Section Office:** Assistant Executive Engineer (AEE) / Section Officer, Section: `{section_office}`.
            2. **Consumer Grievance Redressal Forum (CGRF):** Under Section 42(5) of the Electricity Act 2003 at Superintending Engineer (SE) office.
            3. **Centralized Digital Helpline:** Minnagam 24x7 Customer Care Helpline — Dial `1912` (Toll-Free).
            4. **Check-Meter Testing:** Right to demand parallel check meter testing under CEA Regulations 2006 Rule 57.
            """)

    # Deep-Dive Expander for Technical Examiners
    with st.expander("🔍 Deep-Dive Regulatory & MLOps Verification Checklist (6 Checks)"):
        check_rows = []
        for c in report.audit_checks:
            check_rows.append({
                "Verification Check": c.check_name,
                "Status": get_check_status_badge(c.status),
                "Findings & Audit Details": c.details
            })
        st.write(
            pd.DataFrame(check_rows).to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
        if report.missing_evidence_and_recommended_checks:
            st.markdown("**Recommended Verifications:**")
            for me in report.missing_evidence_and_recommended_checks:
                st.markdown(f"- {me}")
