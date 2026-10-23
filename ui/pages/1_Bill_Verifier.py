"""Bill Verifier & OCR Upload Page."""

import io
import os
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
from ui.utils import format_currency, get_verdict_badge, get_certainty_badge

st.set_page_config(page_title="Bill Verifier | BillGuard", page_icon="📄", layout="wide")

# Singletons
@st.cache_resource
def load_components():
    t_engine = TariffEngine()
    e_engine = ExplanationEngine()
    g_gen = GrievanceGenerator()
    reg = ModelRegistry()
    aud = AuditLogger()
    f_ext = BillFeatureExtractor()

    # Load OCR extractor
    ocr_p = "./models/ocr_crnn_scratch.pth"
    ocr = BillOcrExtractor(model_path=ocr_p if os.path.exists(ocr_p) else None)

    # Load Production ML Model
    prod = reg.get_production_model()
    ml_model = None
    if prod and os.path.exists(prod.artifact_path):
        try:
            if "Autoencoder" in prod.model_name:
                ml_model = BillAutoencoderDetector()
                ml_model.load(prod.artifact_path)
            else:
                ml_model = BillIsolationForest()
                ml_model.load(prod.artifact_path)
        except Exception:
            ml_model = None

    return t_engine, e_engine, g_gen, reg, aud, f_ext, ocr, ml_model

tariff_engine, exp_engine, grievance_gen, registry, audit_logger, feat_extractor, ocr_extractor, ml_model = load_components()

st.title("📄 Electricity Bill Verifier & OCR Audit")
st.markdown("Upload a photo or scanned copy of your electricity bill card, or enter meter readings manually to perform statutory tariff verification.")

# Available tariffs
tariffs = tariff_engine.list_available_tariffs()
tariff_options = {t["tariff_code"]: f"{t['tariff_name']} ({t['verification_status']})" for t in tariffs}

# Preset Scenarios
preset = st.sidebar.selectbox(
    "Load Preset Test Case",
    [
        "None (Custom)",
        "Scenario A: Confirmed Arithmetic Overcharge",
        "Scenario B: Extended Billing Cycle (78 Days)",
        "Scenario C: Verified Normal Lifeline (85 Units)",
        "Scenario D: Uncharacteristic Summer Spike (5x Surge)",
    ]
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
}

if preset == "Scenario A: Confirmed Arithmetic Overcharge":
    defaults.update({
        "prev_read": 2000.0, "curr_read": 2300.0, "billed_units": 300.0,
        "billed_amt": 1450.0,  # True should be ~780
    })
elif preset == "Scenario B: Extended Billing Cycle (78 Days)":
    defaults.update({
        "prev_date": "2024-05-10", "curr_date": "2024-07-27",  # 78 days
        "prev_read": 3000.0, "curr_read": 3550.0, "billed_units": 550.0,
        "billed_amt": 2650.0,
    })
elif preset == "Scenario C: Verified Normal Lifeline (85 Units)":
    defaults.update({
        "prev_read": 1100.0, "curr_read": 1185.0, "billed_units": 85.0,
        "billed_amt": 0.0,
    })
elif preset == "Scenario D: Uncharacteristic Summer Spike (5x Surge)":
    defaults.update({
        "prev_read": 5000.0, "curr_read": 6800.0, "billed_units": 1800.0,
        "billed_amt": 18500.0,
    })

tab1, tab2 = st.tabs(["📤 Upload Bill Card Image (OCR)", "✍️ Manual Bill Entry"])

extracted_data = {}

with tab1:
    st.subheader("Upload Bill Image for From-Scratch OCR Extraction")
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
                st.warning("⚠️ **Manual Review Required**: One or more fields had confidence below threshold (0.70). Please inspect the extracted crops and confirm the values below.")
            else:
                st.success("✅ **High-Confidence OCR Extraction**: All fields extracted with high model certainty.")

            # Display field confidence table
            conf_df = pd.DataFrame([
                {"Field": k, "Extracted Value": str(extracted_data.get(k, "")), "OCR Confidence": f"{ocr_res['confidences'].get(k, 0.0):.2f}"}
                for k in ocr_res["fields"].keys()
            ])
            st.dataframe(conf_df, hide_index=True, use_container_width=True)

            # Crop previews for low-confidence fields
            if ocr_res["low_confidence_fields"]:
                st.markdown("**Inspect Extracted Field Crops:**")
                crop_cols = st.columns(min(3, len(ocr_res["low_confidence_fields"])))
                for idx, f_name in enumerate(ocr_res["low_confidence_fields"][:3]):
                    with crop_cols[idx % 3]:
                        crop_img = ocr_res["crop_images"].get(f_name)
                        if crop_img:
                            st.image(crop_img, caption=f"{f_name} (conf: {ocr_res['confidences'][f_name]:.2f})", width=160)

with tab2:
    st.subheader("Manual Bill Parameters")
    st.caption("You can override or verify any field extracted by OCR or enter values directly.")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    consumer_no = st.text_input("Consumer Number", value=extracted_data.get("consumer_number", defaults["consumer_no"]))
    consumer_name = st.text_input("Consumer Name", value=defaults["name"])
    section_office = st.text_input("Section Office / Division", value=defaults["section"])

with col_f2:
    prev_date = st.text_input("Previous Reading Date (YYYY-MM-DD)", value=extracted_data.get("previous_reading_date", defaults["prev_date"]))
    curr_date = st.text_input("Current Reading Date (YYYY-MM-DD)", value=extracted_data.get("current_reading_date", defaults["curr_date"]))
    tariff_selected = st.selectbox("Tariff Version", list(tariff_options.keys()), format_func=lambda x: tariff_options[x], index=list(tariff_options.keys()).index(defaults["tariff_code"]))

with col_f3:
    prev_read = st.number_input("Previous Meter Reading", value=float(extracted_data.get("previous_meter_reading", defaults["prev_read"])), min_value=0.0, step=10.0)
    curr_read = st.number_input("Current Meter Reading", value=float(extracted_data.get("current_meter_reading", defaults["curr_read"])), min_value=0.0, step=10.0)
    billed_units = st.number_input("Billed Units on Receipt (Optional)", value=float(extracted_data.get("billed_units", defaults["billed_units"])), min_value=0.0, step=1.0)
    billed_amt = st.number_input("Billed Amount on Receipt (₹)", value=float(extracted_data.get("billed_amount", defaults["billed_amt"])), min_value=0.0, step=10.0)
    arrears_amt = st.number_input("Arrears / Past Dues (₹)", value=float(extracted_data.get("arrears_amount", defaults["arrears"])), min_value=0.0, step=10.0)

st.markdown("---")
btn_verify = st.button("🚀 Verify Bill & Audit Calculations", type="primary", use_container_width=True)

if btn_verify or preset != "None (Custom)":
    # 1. Run deterministic tariff calculation
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
        tariff_code=tariff_selected,
    )
    t_result = tariff_engine.calculate_bill(bill_input)

    # 2. Run ML Anomaly Detection
    hist_units = [280.0, 310.0, 320.0, 290.0, 330.0]
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

    ml_score = 0.15
    is_ml_anom = False
    attributions = []

    if ml_model is not None:
        try:
            scores = ml_model.score_samples(feat_vec.reshape(1, -1))
            preds = ml_model.predict(feat_vec.reshape(1, -1))
            ml_score = float(scores[0])
            is_ml_anom = bool(preds[0] == 1)
            attributions = ml_model.explain_instance(feat_vec)
        except Exception:
            pass

    # 3. Generate bilingual explanation
    report = exp_engine.generate_explanation(
        tariff_result=t_result,
        ml_score=ml_score,
        ml_predicted=is_ml_anom,
        feature_attributions=attributions,
        historical_mean_units=hist_mean,
    )

    # Persist in audit log
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

    # --- RESULTS DISPLAY ---
    st.markdown("### 📊 Verification Verdict & Audit Results")

    # Verdict Header Card
    v_badge = get_verdict_badge(report.category)
    c_badge = get_certainty_badge(report.certainty_level)
    st.markdown(f"**Verdict Status:** {v_badge} &nbsp;&nbsp; **Evidence Level:** {c_badge}", unsafe_allow_html=True)

    # Key KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Calculated Units", f"{t_result.calculated_units:.1f} kWh", delta=f"{t_result.units_discrepancy:+.1f} kWh" if not t_result.units_match else "Exact Match")
    with m2:
        st.metric("Statutory Calculated Total", format_currency(t_result.calculated_grand_total))
    with m3:
        st.metric("Billed Amount", format_currency(billed_amt))
    with m4:
        disc = t_result.discrepancy_amount or 0.0
        st.metric(
            "Audit Discrepancy",
            format_currency(disc),
            delta=f"{disc:+.2f} ₹",
            delta_color="inverse" if abs(disc) > 2.0 else "normal"
        )

    # Bilingual Explanation Tabs
    st.markdown("### 🗣️ Evidence-Backed Bilingual Explanation")
    exp_tab_en, exp_tab_ta = st.tabs(["🇬🇧 English Explanation", "🇮🇳 தமிழ் விளக்கம்"])

    with exp_tab_en:
        st.markdown(f"**Summary:** {report.english_summary}")
        st.markdown("**Verifiable Evidence Points:**")
        for ev in report.evidence_points_en:
            st.markdown(f"- {ev}")
        st.info(f"💡 **Recommended Action:** {report.recommended_action_en}")

    with exp_tab_ta:
        st.markdown(f"**விளக்கம்:** {report.tamil_summary}")
        st.markdown("**உறுதிப்படுத்தப்பட்ட ஆதாரங்கள்:**")
        for ev in report.evidence_points_ta:
            st.markdown(f"- {ev}")
        st.info(f"💡 **பரிந்துரைக்கப்படும் நடவடிக்கை:** {report.recommended_action_ta}")

    # Itemized Slab Breakdown Table
    st.markdown("### 📋 Statutory Slab-by-Slab Calculation Audit")
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

    # Additional charges summary
    c_sub1, c_sub2, c_sub3, c_sub4 = st.columns(4)
    c_sub1.caption(f"**Net Energy Charges:** {format_currency(t_result.energy_charge_net)}")
    c_sub2.caption(f"**FPPCA Fuel Surcharge:** {format_currency(t_result.fppca_amount)}")
    c_sub3.caption(f"**Assessed Arrears:** {format_currency(t_result.arrears_amount)}")
    c_sub4.caption(f"**Grand Statutory Total:** {format_currency(t_result.calculated_grand_total)}")

    # Formal Grievance Letter Section
    st.markdown("---")
    st.markdown("### 📝 Formal TANGEDCO Representation / Grievance Petition")
    st.caption("A legally formal representation petition citing TNERC tariff orders and Tamil Nadu Electricity Distribution Code (Clause 11). Ready to submit to your local Section Officer.")

    petition_txt = grievance_gen.generate_complaint_text(
        consumer_number=consumer_no,
        consumer_name=consumer_name,
        section_office=section_office,
        tariff_result=t_result,
        explanation=report,
    )

    with st.expander("📄 View Formal Representation Letter", expanded=(report.is_anomaly)):
        st.text_area("Petition Draft", petition_txt, height=280)

        # Generate PDF
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
            st.download_button(
                "📥 Download Petition as Text (.txt)",
                data=petition_txt,
                file_name=f"TANGEDCO_Representation_{consumer_no}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_d2:
            if os.path.exists(pdf_temp):
                with open(pdf_temp, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "📄 Download Complaint-Ready PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"TANGEDCO_Representation_{consumer_no}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
