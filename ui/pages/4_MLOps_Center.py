"""MLOps Monitoring Center, Drift Dashboard, Model Registry & Safe Rollback."""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.mlops.registry import ModelRegistry
from src.mlops.drift_monitor import DriftMonitor
from src.mlops.audit_logger import AuditLogger
from src.ml.train import train_and_benchmark_all_models

st.set_page_config(page_title="MLOps Center | BillGuard", page_icon="🛠️", layout="wide")

@st.cache_resource
def load_mlops_resources():
    reg = ModelRegistry()
    drift = DriftMonitor()
    aud = AuditLogger()
    return reg, drift, aud

registry, drift_monitor, audit_logger = load_mlops_resources()

st.title("🛠️ MLOps Monitoring & Governance Center")
st.markdown("Automated model registry, safety promotion gate, 1-click rollback, continuous data drift monitoring (PSI & KS-test), and live prediction audit logs.")

tab_gov, tab_drift, tab_retrain, tab_audit = st.tabs([
    "🏛️ Model Registry & Safety Gate",
    "📈 Data Drift & Distribution Shifts",
    "🔄 Trigger Retraining from Scratch",
    "📜 Live Prediction Audit Trail"
])

# --- TAB 1: MODEL REGISTRY & PROMOTION ---
with tab_gov:
    st.subheader("Model Registry Lifecycle Stages")
    prod_model = registry.get_production_model()

    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        if prod_model:
            st.success(f"""
            ### 🟢 Active Production Model: v{prod_model.version}
            - **Model Architecture:** `{prod_model.model_name}`
            - **Promoted At:** {prod_model.promoted_at or prod_model.registered_at}
            - **F1 Score:** `{prod_model.metrics.get('f1_score', 0.0):.3f}`
            - **Precision:** `{prod_model.metrics.get('precision', 0.0):.3f}` | **Recall:** `{prod_model.metrics.get('recall', 0.0):.3f}`
            - **Description:** {prod_model.description}
            """)
        else:
            st.warning("No model currently in PRODUCTION.")

    with col_p2:
        st.markdown("#### Model Governance Actions")
        # Rollback Button
        if st.button("⏪ Rollback Production Model to Previous Version", type="secondary"):
            success, msg = registry.rollback_production()
            if success:
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    st.markdown("#### All Registered Model Versions")
    versions_list = []
    for v in registry.versions:
        versions_list.append({
            "Version": f"v{v.version}",
            "Model Name": v.model_name,
            "Stage": v.stage,
            "F1-Score": v.metrics.get("f1_score", "-"),
            "Precision": v.metrics.get("precision", "-"),
            "Recall": v.metrics.get("recall", "-"),
            "Registered At": v.registered_at[:19],
            "Notes": v.notes or v.description,
        })
    st.dataframe(pd.DataFrame(versions_list), hide_index=True, use_container_width=True)

    # Candidate Promotion Section
    candidates = [v for v in registry.versions if v.stage == "CANDIDATE"]
    if candidates:
        st.markdown("#### 🚀 Candidate Model Promotion Gate")
        cand_choice = st.selectbox("Select Candidate to Evaluate", [f"v{c.version}: {c.model_name}" for c in candidates])
        cand_ver = int(cand_choice.split(":")[0].replace("v", ""))

        can_promote, reason = registry.evaluate_promotion_gate(cand_ver)
        if can_promote:
            st.success(f"**Safety Gate Verdict:** {reason}")
            if st.button(f"Promote v{cand_ver} to PRODUCTION"):
                success, p_msg = registry.promote_candidate(cand_ver)
                if success:
                    st.success(f"🎉 {p_msg}")
                    st.rerun()
                else:
                    st.error(p_msg)
        else:
            st.error(f"**Safety Gate Verdict:** {reason}")

# --- TAB 2: DATA DRIFT ---
with tab_drift:
    st.subheader("Statistical Data & Feature Drift Monitoring")
    st.caption("Measures distribution shifts using Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov tests.")

    # Load baseline vs current
    train_p = Path("./data/synthetic/train_bills.csv")
    test_p = Path("./data/synthetic/test_bills.csv")

    if train_p.exists() and test_p.exists():
        df_base = pd.read_csv(train_p)
        df_curr = pd.read_csv(test_p)
        drift_report = drift_monitor.evaluate_feature_drift(df_base, df_curr)
        qual_report = drift_monitor.evaluate_data_quality(df_curr)

        # Overview Metrics
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Overall Drift Status", drift_report["overall_status"])
        with col_d2:
            st.metric("Max Feature PSI", f"{drift_report['max_psi']:.4f}")
        with col_d3:
            st.metric("Data Quality Score", f"{qual_report['data_quality_score'] * 100:.0f}%")

        # Feature Drift Table
        f_rows = []
        for feat, d in drift_report["features"].items():
            f_rows.append({
                "Feature": feat,
                "PSI Score": d["psi"],
                "Severity": d["severity"],
                "KS Statistic": d["ks_statistic"],
                "p-value": d["ks_pvalue"],
                "Drift Detected": "🚨 Yes" if d["drift_detected"] else "✅ No",
                "Baseline Mean": d["baseline_mean"],
                "Current Mean": d["current_mean"],
            })
        st.dataframe(pd.DataFrame(f_rows), hide_index=True, use_container_width=True)

        # PSI Bar Chart
        psi_df = pd.DataFrame({"Feature": list(drift_report["features"].keys()), "PSI": [d["psi"] for d in drift_report["features"].values()]}).set_index("Feature")
        st.bar_chart(psi_df)
    else:
        st.info("Run dataset generation or retraining to initialize drift comparison baseline.")

# --- TAB 3: RETRAINING PIPELINE ---
with tab_retrain:
    st.subheader("Trigger End-to-End Retraining from Scratch")
    st.markdown("""
    **Zero Pretrained Weights Guarantee**:
    Retraining re-initializes all weights randomly, generates fresh independent domestic consumer sequences,
    trains models from scratch, evaluates on isolated test sets, and registers a candidate in the Model Registry.
    """)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        n_consumers = st.number_input("Number of Synthetic Consumers", value=120, min_value=40, max_value=500, step=20)
    with col_r2:
        n_cycles = st.number_input("Cycles per Consumer", value=10, min_value=6, max_value=18, step=2)

    if st.button("🚀 Start Retraining from Scratch", type="primary"):
        with st.spinner("Generating fresh synthetic histories and training models from random initialization..."):
            benchmarks = train_and_benchmark_all_models(
                num_consumers=n_consumers,
                cycles_per_consumer=n_cycles,
                output_models_dir="./models",
            )
            # Register new candidate
            cand = registry.register_candidate(
                model_name="IsolationForest_Retrained",
                artifact_path="./models/isolation_forest.joblib",
                metrics=benchmarks["isolation_forest"],
                description=f"Auto-retrained on {n_consumers} consumers"
            )
            st.success(f"🎉 Retraining complete! Registered Candidate Model **v{cand.version}**.")

            # Metrics Table
            comp_data = [
                {"Model": "Rule-Based Baseline", **benchmarks["baseline"]},
                {"Model": "Isolation Forest (Scratch)", **benchmarks["isolation_forest"]},
                {"Model": "PyTorch Autoencoder (Scratch)", **benchmarks["autoencoder"]},
            ]
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

# --- TAB 4: AUDIT LOGS ---
with tab_audit:
    st.subheader("Prediction Audit Trail")
    st.caption("Immutable SQLite log of all bill audits for regulatory compliance and dispute resolution.")

    stats = audit_logger.get_audit_summary_stats()
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.metric("Total Bills Audited", stats["total_bills_audited"])
    with col_a2:
        st.metric("Anomalies Flagged", stats["anomalies_flagged"])
    with col_a3:
        st.metric("Anomaly Rate", f"{stats['anomaly_rate_pct']:.1f}%")

    recent_logs = audit_logger.fetch_recent_logs(limit=25)
    if recent_logs:
        st.dataframe(pd.DataFrame(recent_logs), hide_index=True, use_container_width=True)
    else:
        st.info("No audit entries recorded yet. Verify bills in the Bill Verifier page to populate.")
