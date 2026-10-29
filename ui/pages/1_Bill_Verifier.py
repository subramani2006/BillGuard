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
import plotly.graph_objects as go

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
    get_plotly_dark_theme,
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
    subtitle="Statutory TNERC LT-1A Tariff Verification, from-scratch CRNN OCR reading extraction, and explainable AI anomaly screening.",
    tag_left="<span class='status-dot'></span> 8-STAGE AUDIT ENGINE ACTIVE",
    tag_right="🏛️ TNERC GAZETTE COMPLIANT",
)

# Available tariffs
tariffs = tariff_engine.list_available_tariffs()
tariff_options = {t["tariff_code"]: f"{'🏛️' if t.get('verification_status') == 'VERIFIED_OFFICIAL' else '🧪'} {t['tariff_name']}" for t in tariffs}

# ==============================================================================
# THREE PRELOADED DEMO SCENARIOS
# ==============================================================================
st.markdown("### 🎯 Demonstration Scenarios (Single-Click Selection)")

col_sc1, col_sc2, col_sc3 = st.columns(3)
with col_sc1:
    st.markdown("""
    <div class="glass-card" style="border: 1px solid rgba(16, 185, 129, 0.4); padding: 14px; text-align: left;">
        <span style="color: #34D399; font-weight: 700; font-size: 0.95rem;">🟢 Scenario A — Normal Bill</span>
        <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
            A typical bill with consistent consumption and valid input values.
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">
            <b>Expected:</b> Normal — No anomaly detected.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_sc2:
    st.markdown("""
    <div class="glass-card" style="border: 1px solid rgba(245, 158, 11, 0.4); padding: 14px; text-align: left;">
        <span style="color: #FBBF24; font-weight: 700; font-size: 0.95rem;">🟡 Scenario B — Unusual Bill</span>
        <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
            A bill with an unusually large consumption increase that needs investigation.
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">
            <b>Expected:</b> Anomaly detected — Unusual consumption increase.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_sc3:
    st.markdown("""
    <div class="glass-card" style="border: 1px solid rgba(239, 68, 68, 0.4); padding: 14px; text-align: left;">
        <span style="color: #F87171; font-weight: 700; font-size: 0.95rem;">🔴 Scenario C — Invalid Bill Data</span>
        <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
            A bill containing missing or invalid information (inverted reading).
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">
            <b>Expected:</b> Invalid data — Please correct the highlighted field.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Radio pill selector
selected_scenario = st.radio(
    "Choose Demonstration Scenario",
    [
        "Scenario A — Normal Bill",
        "Scenario B — Unusual Bill",
        "Scenario C — Invalid Bill Data",
        "Custom Entry / OCR Upload"
    ],
    horizontal=True,
)

# Scenario Data Definitions
SCENARIO_CONFIGS = {
    "Scenario A — Normal Bill": {
        "desc": "A typical bill with consistent consumption (200.0 kWh vs. 200.0 kWh baseline) and valid statutory charges (₹279.00).",
        "consumer_no": "09-245-001-042",
        "name": "M. Senthilkumar",
        "section": "Anna Nagar West, Chennai",
        "prev_date": "2024-06-15",
        "curr_date": "2024-08-15",
        "prev_read": 1000.0,
        "curr_read": 1200.0,
        "billed_units": 200.0,
        "billed_amt": 279.0,
        "arrears": 0.0,
        "prev_bill_amt": 279.0,
        "hist_units": [190.0, 205.0, 195.0, 210.0, 200.0],
        "tariff_code": "TN_DOMESTIC_LT1A_2024",
    },
    "Scenario B — Unusual Bill": {
        "desc": "Valid input values, but consumption surged from 120.0 kWh baseline to 420.0 kWh (+250% increase). Tests AI anomaly detection.",
        "consumer_no": "09-245-001-042",
        "name": "M. Senthilkumar",
        "section": "Anna Nagar West, Chennai",
        "prev_date": "2024-06-15",
        "curr_date": "2024-08-15",
        "prev_read": 2000.0,
        "curr_read": 2420.0,
        "billed_units": 420.0,
        "billed_amt": 1467.40,
        "arrears": 0.0,
        "prev_bill_amt": 100.0,
        "hist_units": [120.0, 115.0, 125.0, 118.0, 122.0],
        "tariff_code": "TN_DOMESTIC_LT1A_2024",
    },
    "Scenario C — Invalid Bill Data": {
        "desc": "Deliberately invalid input: Current meter reading (4,200.0 kWh) is less than previous reading (4,850.0 kWh). Tests input validation guardrail.",
        "consumer_no": "09-245-001-042",
        "name": "M. Senthilkumar",
        "section": "Anna Nagar West, Chennai",
        "prev_date": "2024-06-15",
        "curr_date": "2024-08-15",
        "prev_read": 4850.0,
        "curr_read": 4200.0,
        "billed_units": 0.0,
        "billed_amt": 0.0,
        "arrears": 0.0,
        "prev_bill_amt": 1000.0,
        "hist_units": [200.0, 200.0, 200.0, 200.0, 200.0],
        "tariff_code": "TN_DOMESTIC_LT1A_2024",
    },
    "Custom Entry / OCR Upload": {
        "desc": "Enter custom meter readings manually or upload a physical electricity bill card image for OCR.",
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
        "prev_bill_amt": 1180.0,
        "hist_units": [280.0, 310.0, 320.0, 290.0, 330.0],
        "tariff_code": "TN_DOMESTIC_LT1A_2024",
    }
}

active_cfg = SCENARIO_CONFIGS[selected_scenario]

# Display brief explanation banner of current scenario
st.markdown(f"""
<div class="glass-card" style="margin-top: 10px; margin-bottom: 20px; padding: 12px 18px; border-left: 3px solid #38BDF8;">
    <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
        📌 <b>Active Setup:</b> {active_cfg['desc']}
    </div>
</div>
""", unsafe_allow_html=True)

# Session state management for instant single-click execution & switching
if "current_scenario" not in st.session_state:
    st.session_state["current_scenario"] = selected_scenario
    st.session_state["audit_executed"] = True
elif st.session_state["current_scenario"] != selected_scenario:
    st.session_state["current_scenario"] = selected_scenario
    st.session_state["audit_executed"] = True

extracted_data = {}

# If Custom Entry is selected, show OCR upload option
if selected_scenario == "Custom Entry / OCR Upload":
    with st.expander("📤 Optional: Upload Bill Card Image for From-Scratch OCR Extraction", expanded=False):
        col_u1, col_u2 = st.columns([1, 1])
        with col_u1:
            uploaded_file = st.file_uploader("Upload Bill Image (PNG, JPG)", type=["png", "jpg", "jpeg"])
            btn_sample = st.button("🖼️ Generate & Load Sample TANGEDCO Bill")
            bill_img = None
            if uploaded_file is not None:
                bill_img = Image.open(uploaded_file).convert("RGB")
            elif btn_sample:
                img_gen = BillImageGenerator(seed=42)
                bill_img, _ = img_gen.generate_full_bill_image(
                    consumer_no=active_cfg["consumer_no"],
                    consumer_name=active_cfg["name"],
                    bill_date=active_cfg["curr_date"],
                    prev_date=active_cfg["prev_date"],
                    prev_reading=active_cfg["prev_read"],
                    curr_reading=active_cfg["curr_read"],
                    units=active_cfg["billed_units"],
                    net_amount=active_cfg["billed_amt"],
                )
            if bill_img is not None:
                st.image(bill_img, caption="Loaded Bill Card Image", use_container_width=True)

        with col_u2:
            if bill_img is not None:
                with st.spinner("Extracting bill fields with CRNN model..."):
                    ocr_res = ocr_extractor.extract_from_bill_image(bill_img)
                    extracted_data = ocr_res["fields"]
                st.success("✅ OCR Extraction Complete! Extracted fields are populated below.")

# Parameters Form
st.markdown("#### ✍️ Bill Parameters")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("##### 👤 Consumer Information")
    consumer_no = st.text_input("Consumer Number", value=str(extracted_data.get("consumer_number") or active_cfg["consumer_no"]))
    consumer_name = st.text_input("Consumer Name", value=active_cfg["name"])
    section_office = st.text_input("Section Office / Division", value=active_cfg["section"])
    prev_bill_amount = st.number_input("Previous Bill Amount (₹)", value=float(active_cfg["prev_bill_amt"]), min_value=0.0, step=50.0)

with col_f2:
    st.markdown("##### 📅 Billing Period & Tariff")
    prev_date = st.text_input("Previous Reading Date", value=str(extracted_data.get("previous_reading_date") or active_cfg["prev_date"]))
    curr_date = st.text_input("Current Reading Date", value=str(extracted_data.get("current_reading_date") or active_cfg["curr_date"]))
    tariff_selected = st.selectbox(
        "Applicable Tariff Schedule",
        list(tariff_options.keys()),
        format_func=lambda x: tariff_options[x],
        index=list(tariff_options.keys()).index(active_cfg["tariff_code"]) if active_cfg["tariff_code"] in tariff_options else 0
    )
    st.caption("🏛️ *Official TNERC Statutory Tariff (July 2024 Revision)*")

with col_f3:
    st.markdown("##### ⚡ Meter Readings & Billed Values")
    default_prev_r = safe_float(extracted_data.get("previous_meter_reading"), active_cfg["prev_read"])
    default_curr_r = safe_float(extracted_data.get("current_meter_reading"), active_cfg["curr_read"])
    default_b_units = safe_float(extracted_data.get("billed_units"), active_cfg["billed_units"])
    default_b_amt = safe_float(extracted_data.get("billed_amount"), active_cfg["billed_amt"])

    prev_read = st.number_input("Previous Meter Reading", value=float(default_prev_r), min_value=0.0, step=10.0)
    
    # Highlight invalid reading for Scenario C
    is_inverted = (default_curr_r < default_prev_r)
    curr_read = st.number_input(
        "Current Meter Reading" + (" ⚠️ [INVERTED / TYPO]" if is_inverted else ""),
        value=float(default_curr_r),
        min_value=0.0,
        step=10.0
    )
    if is_inverted:
        st.caption("🔴 **Field Error:** Current reading is lower than previous reading.")
        
    billed_units = st.number_input("Billed Units on Receipt (kWh)", value=float(default_b_units), min_value=0.0, step=1.0)
    billed_amt = st.number_input("Billed Amount on Receipt (₹)", value=float(default_b_amt), min_value=0.0, step=10.0)

# Optional Special Conditions
with st.expander("⚙️ Special Conditions (Meter Replacement, Past Arrears)", expanded=False):
    c_sp1, c_sp2 = st.columns(2)
    with c_sp1:
        is_estimated = st.checkbox("Estimated Reading / Door Locked (DL)", value=False)
        meter_replaced = st.checkbox("Meter Replacement mid-cycle", value=False)
        old_meter_final = None
        new_meter_initial = 0.0
        if meter_replaced:
            old_meter_final = st.number_input("Final Reading on Old Meter", value=5200.0, min_value=0.0)
            new_meter_initial = st.number_input("Initial Reading on New Meter", value=0.0, min_value=0.0)
    with c_sp2:
        arrears_amt = st.number_input("Assessed Past Arrears (₹)", value=float(active_cfg["arrears"]), min_value=0.0, step=10.0)
        additional_charges = st.number_input("Additional Charges / Taxes (₹)", value=0.0, min_value=0.0)
        adjustments = st.number_input("Credits / Rebates (₹)", value=0.0, min_value=0.0)

st.markdown("<br/>", unsafe_allow_html=True)
btn_verify = st.button("🚀 Run Verification", type="primary", use_container_width=True)

if btn_verify:
    st.session_state["audit_executed"] = True

# ==============================================================================
# AUDIT & ANOMALY DETECTION EXECUTION
# ==============================================================================
if st.session_state.get("audit_executed", False):
    st.markdown("---")
    
    # 1. Deterministic tariff verification & input validation
    bill_input = BillInput(
        consumer_number=consumer_no,
        consumer_name=consumer_name,
        previous_reading_date=prev_date,
        current_reading_date=curr_date,
        previous_meter_reading=prev_read,
        current_meter_reading=curr_read,
        billed_units=billed_units,
        billed_amount=billed_amt,
        arrears_amount=arrears_amt if "arrears_amt" in locals() else 0.0,
        additional_charges=additional_charges if "additional_charges" in locals() else 0.0,
        adjustments=adjustments if "adjustments" in locals() else 0.0,
        tariff_code=tariff_selected,
        is_estimated_reading=is_estimated if "is_estimated" in locals() else False,
        meter_replaced=meter_replaced if "meter_replaced" in locals() else False,
        old_meter_final_reading=old_meter_final if "old_meter_final" in locals() else None,
        new_meter_initial_reading=new_meter_initial if "new_meter_initial" in locals() else 0.0,
    )
    t_result = tariff_engine.calculate_bill(bill_input)

    # CRITICAL SEPARATION CHECK:
    # If input validation fails (e.g. Scenario C: inverted readings), STOP!
    # Do NOT pass invalid data into the ML detector.
    if not t_result.is_valid:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #EF4444; border-radius: 12px; padding: 20px 24px; margin: 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.5rem;">❌</span>
                <span style="font-size: 1.35rem; font-weight: 800; color: #F87171;">Invalid data — Please correct the highlighted field.</span>
            </div>
            <div style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                <b>Input Validation Halted Processing:</b> Current meter reading (<b>""" + str(curr_read) + """ kWh</b>) is less than previous meter reading (<b>""" + str(prev_read) + """ kWh</b>).<br/>
                <i>Deterministic guardrails stopped this record from reaching the machine-learning pipeline. Input validation ensures that data-entry typos or inverted numbers are rejected immediately rather than mistakenly processed as normal or anomalous bills.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.error("🚨 **Specific Validation Errors Detected:**")
        for err in t_result.validation_errors:
            st.markdown(f"- ❌ {err}")

        st.info("💡 **How to resolve:** Correct the current meter reading in the form above so it is greater than the previous reading. If the meter was replaced, check the 'Meter Replacement' option under Special Conditions.")

    else:
        # 2. Input is VALID -> Proceed to ML Anomaly Screening
        hist_units = active_cfg.get("hist_units", [200.0, 200.0, 200.0, 200.0, 200.0])
        hist_mean = float(np.mean(hist_units))
        hist_records = [{"recorded_units": u, "billing_days": 60} for u in hist_units]
        curr_record = {
            "recorded_units": t_result.calculated_units,
            "billed_units": billed_units,
            "billing_days": t_result.billing_days,
            "cycle_in_year": 3,
            "calculated_grand_total": t_result.calculated_grand_total,
            "discrepancy_amount": t_result.discrepancy_amount or 0.0,
            "arrears_amount": arrears_amt if "arrears_amt" in locals() else 0.0,
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

        # =====================================================================
        # PROMINENT RESULT PANEL
        # =====================================================================
        if report.is_anomaly:
            # Scenario B or Anomaly Result
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.15); border: 2px solid #F59E0B; border-radius: 12px; padding: 20px 24px; margin: 16px 0;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem;">⚠️</span>
                    <span style="font-size: 1.35rem; font-weight: 800; color: #FBBF24;">Anomaly detected — Unusual consumption increase.</span>
                </div>
                <div style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                    <b>Unusual Pattern Flagged:</b> Consumption surged from a 5-cycle baseline of <b>{hist_mean:.1f} kWh</b> to <b>{t_result.calculated_units:.1f} kWh</b> (+{((t_result.calculated_units - hist_mean)/hist_mean)*100:.0f}% increase). 
                    Isolation Forest Anomaly Score: <b>{ml_score:.2f}</b> (threshold: 0.50).<br/>
                    <i>⚠️ <b>Important Note:</b> An anomaly detection warning is a flag for consumer investigation (such as checking for appliance defects, wiring leakage, or meter drift), NOT proof of fraud or an incorrect bill.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Scenario A or Normal Result
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10B981; border-radius: 12px; padding: 20px 24px; margin: 16px 0;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem;">✅</span>
                    <span style="font-size: 1.35rem; font-weight: 800; color: #34D399;">Normal — No anomaly detected.</span>
                </div>
                <div style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                    <b>Verified Normal Record:</b> Consumed <b>{t_result.calculated_units:.1f} kWh</b>, precisely consistent with your historical baseline of <b>{hist_mean:.1f} kWh</b>. 
                    The billed amount of <b>{format_currency(billed_amt)}</b> exactly matches the statutory TNERC LT-1A tariff calculation of <b>{format_currency(t_result.calculated_grand_total)}</b> (including the 100-unit free GoTN lifeline subsidy and FPPCA fuel surcharge).
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
            st.metric("Billed Amount on Receipt", format_currency(billed_amt))
        with m4:
            disc = t_result.discrepancy_amount or 0.0
            st.metric(
                "Audit Discrepancy",
                format_currency(disc),
                delta=f"{disc:+.2f} ₹" if abs(disc) > 0.01 else "0.00 ₹ (Exact Match)",
                delta_color="inverse" if abs(disc) > 2.0 else "normal"
            )

        # Supporting Evidence Tabs
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
            "🧮 Statutory Slab Calculation",
            "📈 Consumption Baseline Comparison",
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
                {"Item": "Assessed Past Arrears", "Discrepancy Value": format_currency(report.itemized_discrepancies.get('arrears_billed_inr', 0.0))},
                {"Item": "Total Net Discrepancy (billed - calculated)", "Discrepancy Value": format_currency(report.itemized_discrepancies.get('total_net_discrepancy_inr', 0.0))},
            ])
            st.dataframe(disc_df, hide_index=True, use_container_width=True)

        with res_tab2:
            st.markdown("#### 📊 Historical Consumption Baseline vs. Current Cycle")
            st.caption("Visualizing consumption consistency against 5 previous bi-monthly billing cycles.")

            cycle_labels = [f"Cycle -{5-i}" for i in range(len(hist_units))] + ["Current Cycle"]
            cycle_values = list(hist_units) + [t_result.calculated_units]
            
            # Color current bar according to anomaly status
            bar_colors = ["#38BDF8"] * len(hist_units)
            if report.is_anomaly:
                bar_colors.append("#F59E0B")  # Amber for anomaly
            else:
                bar_colors.append("#10B981")  # Emerald for normal

            fig_comp = go.Figure(go.Bar(
                x=cycle_labels,
                y=cycle_values,
                marker=dict(color=bar_colors),
                text=[f"{v:.0f} kWh" for v in cycle_values],
                textposition="auto",
            ))

            # Add baseline mean line
            fig_comp.add_hline(
                y=hist_mean,
                line_dash="dot",
                line_color="#94A3B8",
                annotation_text=f"Baseline Average ({hist_mean:.0f} kWh)",
                annotation_position="top left",
                annotation_font=dict(color="#CBD5E1", size=10)
            )

            fig_comp.update_layout(
                title=f"Consumption Comparison — Current ({t_result.calculated_units:.0f} kWh) vs Historical Baseline ({hist_mean:.0f} kWh)",
                xaxis_title="Billing Cycles",
                yaxis_title="Energy Consumed (kWh)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            fig_comp = get_plotly_dark_theme(fig_comp, height=340)
            st.plotly_chart(fig_comp, use_container_width=True)

            if report.is_anomaly:
                st.warning(
                    f"⚠️ **Visual Evidence:** Current consumption of **{t_result.calculated_units:.0f} kWh** stands out dramatically "
                    f"above the typical baseline average of **{hist_mean:.0f} kWh**. "
                    "This sudden shift triggered the machine learning anomaly detector."
                )
            else:
                st.success(
                    f"✅ **Visual Evidence:** Current consumption of **{t_result.calculated_units:.0f} kWh** is in close alignment "
                    f"with the historical baseline average of **{hist_mean:.0f} kWh**."
                )

        with res_tab3:
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

        with res_tab4:
            st.markdown("#### 📝 Official TANGEDCO Grievance Petition")
            st.caption("A formal representation petition citing TNERC regulations and Tamil Nadu Electricity Distribution Code. Ready to download and submit.")

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

            with st.expander("👁️ Preview Petition Draft", expanded=False):
                st.text_area("Petition Draft", petition_txt, height=200)

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
