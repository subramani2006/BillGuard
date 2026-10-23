import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for cloud deployment
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.mlops.registry import ModelRegistry
from src.mlops.drift_monitor import DriftMonitor
from src.mlops.audit_logger import AuditLogger
from src.data_gen.bill_generator import SyntheticBillGenerator
from src.ml.train import train_and_benchmark_all_models
from ui.utils import (
    format_currency,
    inject_custom_theme,
    render_hero_banner,
    get_plotly_dark_theme,
)

st.set_page_config(
    page_title="MLOps Center | BillGuard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_custom_theme()

# Custom Grand CSS Styling
st.markdown("""
<style>
    .prod-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 16px rgba(56, 189, 248, 0.1);
    }
    .gate-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(31, 41, 55, 0.60) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-pass { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-fail { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-warn { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_mlops_resources():
    reg = ModelRegistry()
    drift = DriftMonitor()
    aud = AuditLogger()
    return reg, drift, aud

registry, drift_monitor, audit_logger = load_mlops_resources()

# Grand Hero Banner
render_hero_banner(
    title="🛠️ MLOps Governance & Observability Center",
    subtitle="Continuous Model Governance, Automated Safety Promotion Gates, Data Drift Analytics (PSI & KS-Test), and Immutable Audit Trails.",
    tag_left="<span class='status-dot'></span> ACTIVE OBSERVABILITY ENGINE",
    tag_right="🛡️ ZERO PRETRAINED WEIGHTS",
)

# 4 Main Tabs
tab_gov, tab_drift, tab_retrain, tab_audit = st.tabs([
    "🏛️ Model Registry & Safety Gate",
    "📈 Data Drift & Distribution Shifts",
    "🔄 Trigger Retraining from Scratch",
    "📜 Live Prediction Audit Trail"
])


# ==============================================================================
# TAB 1: MODEL REGISTRY & GOVERNANCE
# ==============================================================================
with tab_gov:
    st.markdown("### 🏛️ Model Lifecycle Registry & Automated Safety Gate")
    st.caption("Active models are staged across Development, Candidate, Production, and Archived tiers. Production promotions must satisfy zero-regression safety gates.")

    prod_model = registry.get_production_model()

    col_p1, col_p2 = st.columns([1.2, 0.8])
    with col_p1:
        if prod_model:
            st.markdown(f"""
            <div class="prod-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.3rem; font-weight: 700; color: #38BDF8;">🟢 PRODUCTION ACTIVE: v{prod_model.version}</span>
                    <span style="background: #0284C7; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">ACTIVE INFERENCE</span>
                </div>
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 6px;">{prod_model.model_name}</div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 16px;">{prod_model.description}</div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; border-top: 1px solid #334155; padding-top: 14px;">
                    <div>
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">F1-Score</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #34D399;">{prod_model.metrics.get('f1_score', 0.0):.3f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Precision</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #60A5FA;">{prod_model.metrics.get('precision', 0.0):.3f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Recall</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24;">{prod_model.metrics.get('recall', 0.0):.3f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">ROC-AUC</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #A78BFA;">{prod_model.metrics.get('roc_auc', 0.0):.3f}</div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.8rem; color: #64748B;">
                    Artifact Path: <code>{prod_model.artifact_path}</code> | Promoted At: {prod_model.promoted_at or prod_model.registered_at}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No active model found in PRODUCTION.")

    with col_p2:
        st.markdown("#### 🛡️ Emergency Model Operations")
        st.markdown("If real-world drift or regressions are identified in production inference, the safety gate enables one-click restoration of the previously certified production checkpoint.")
        
        if st.button("⏪ Instant 1-Click Rollback to Previous Version", type="secondary", use_container_width=True):
            success, msg = registry.rollback_production()
            if success:
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

        archived_count = len([v for v in registry.versions if v.stage == "ARCHIVED"])
        candidate_count = len([v for v in registry.versions if v.stage == "CANDIDATE"])
        st.info(f"**Registry Inventory:**\n- 🟢 Production: 1 Active\n- 🟡 Candidates: {candidate_count}\n- 📦 Archived Standby: {archived_count}")

    st.markdown("---")
    st.markdown("#### 📋 Staged Model Inventory Table")
    v_data = []
    for v in registry.versions:
        stage_chip = "🟢 PRODUCTION" if v.stage == "PRODUCTION" else ("🟡 CANDIDATE" if v.stage == "CANDIDATE" else "📦 ARCHIVED")
        v_data.append({
            "Version": f"v{v.version}",
            "Model Name": v.model_name,
            "Stage": stage_chip,
            "F1-Score": f"{v.metrics.get('f1_score', 0.0):.3f}" if v.metrics.get('f1_score') else "-",
            "Precision": f"{v.metrics.get('precision', 0.0):.3f}" if v.metrics.get('precision') else "-",
            "Recall": f"{v.metrics.get('recall', 0.0):.3f}" if v.metrics.get('recall') else "-",
            "Registered Date": v.registered_at[:19].replace("T", " "),
            "Description": v.notes or v.description,
        })
    st.dataframe(pd.DataFrame(v_data), hide_index=True, use_container_width=True)

    # Candidate Evaluation Gate Section
    candidates = [v for v in registry.versions if v.stage == "CANDIDATE"]
    if candidates:
        st.markdown("---")
        st.markdown("### 🚀 Automated Candidate Promotion Gate")
        cand_choice = st.selectbox(
            "Select Candidate Model to Evaluate for Promotion",
            [f"v{c.version}: {c.model_name} (F1: {c.metrics.get('f1_score', 0.0):.3f})" for c in candidates]
        )
        cand_ver = int(cand_choice.split(":")[0].replace("v", ""))
        target_cand = next(c for c in candidates if c.version == cand_ver)

        c_f1 = target_cand.metrics.get("f1_score", 0.0)
        c_prec = target_cand.metrics.get("precision", 0.0)
        p_f1 = prod_model.metrics.get("f1_score", 0.0) if prod_model else 0.0

        can_promote, reason = registry.evaluate_promotion_gate(cand_ver)

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            pass_f1 = c_f1 >= 0.75
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div class="card-label">Gate 1: Min F1 (≥ 0.75)</div>
                <div style="margin-top: 6px;"><span class='metric-badge {'badge-pass' if pass_f1 else 'badge-fail'}'>{'✅ PASS: ' if pass_f1 else '❌ FAIL: '}{c_f1:.3f}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col_g2:
            pass_prec = c_prec >= 0.70
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div class="card-label">Gate 2: Precision (≥ 0.70)</div>
                <div style="margin-top: 6px;"><span class='metric-badge {'badge-pass' if pass_prec else 'badge-fail'}'>{'✅ PASS: ' if pass_prec else '❌ FAIL: '}{c_prec:.3f}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col_g3:
            pass_comp = c_f1 >= p_f1
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div class="card-label">Gate 3: Exceeds Prod F1 ({p_f1:.3f})</div>
                <div style="margin-top: 6px;"><span class='metric-badge {'badge-pass' if pass_comp else 'badge-fail'}'>{'✅ PASS: ' if pass_comp else '❌ FAIL: '}{c_f1:.3f} ≥ {p_f1:.3f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        if can_promote:
            st.markdown(f"""
            <div class="glass-card" style="border: 1.5px solid #10B981; background: rgba(16, 185, 129, 0.12); margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.8rem;">🎉</div>
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #34D399;">Promotion Safety Gate Certified: All Criteria Satisfied</div>
                        <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 2px;">{reason}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"🚀 Promote Certified Candidate v{cand_ver} to PRODUCTION", type="primary", use_container_width=True):
                success, p_msg = registry.promote_candidate(cand_ver)
                if success:
                    st.success(f"✅ {p_msg}")
                    st.rerun()
                else:
                    st.error(p_msg)
        else:
            st.markdown(f"""
            <div class="glass-card" style="border: 1.5px solid #F59E0B; background: rgba(245, 158, 11, 0.12); margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.8rem;">🛡️</div>
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #FBBF24;">MLOps Zero-Regression Guardrail: Promotion Intercepted</div>
                        <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 2px;">
                            <b>Safety Intercept:</b> {reason}<br/>
                            <i>Automated governance ensures models with lower accuracy cannot degrade active production inference.</i>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: DATA DRIFT & DISTRIBUTION SHIFTS (GRAND VISUALS)
# ==============================================================================
with tab_drift:
    st.markdown("### 📈 Continuous Data Drift & Feature Distribution Shifts")
    st.caption("Monitoring real-world consumption variance using Population Stability Index (PSI) and Two-Sample Kolmogorov-Smirnov (KS) statistical tests.")

    train_p = Path("./data/synthetic/train_bills.csv")
    test_p = Path("./data/synthetic/test_bills.csv")

    # Defensive loading: Ensure baseline and current dataframes are always populated
    if not train_p.exists() or not test_p.exists():
        with st.spinner("Generating initial baseline datasets for drift comparison..."):
            gen = SyntheticBillGenerator(seed=42)
            df_base, _, df_curr = gen.generate_full_dataset(num_consumers=40, output_dir="./data/synthetic")
    else:
        df_base = pd.read_csv(train_p)
        df_curr = pd.read_csv(test_p)

    drift_report = drift_monitor.evaluate_feature_drift(df_base, df_curr)
    qual_report = drift_monitor.evaluate_data_quality(df_curr)

    # Top KPI Metrics with Grand Visual Gauges
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(
            "Overall Drift Verdict",
            drift_report["overall_status"].replace("_", " "),
            delta="Healthy" if drift_report["overall_status"] == "NORMAL" else "Attention Required",
            delta_color="normal" if drift_report["overall_status"] == "NORMAL" else "inverse"
        )
    with kpi_col2:
        st.metric("Max Feature PSI", f"{drift_report['max_psi']:.4f}", delta="Threshold: < 0.25")
    with kpi_col3:
        st.metric("Drifted Features", f"{drift_report['drift_detected_features_count']} / {drift_report['total_monitored_features']}")
    with kpi_col4:
        st.metric("Data Quality Score", f"{qual_report['data_quality_score'] * 100:.0f}%", delta="Clean Bounds")

    # Grand Visual: Plotly Gauge for Data Quality & Bar Chart for PSI
    st.markdown("---")
    chart_col1, chart_col2 = st.columns([1.3, 0.7])

    with chart_col1:
        st.markdown("#### 📊 Feature Population Stability Index (PSI)")
        feats = list(drift_report["features"].keys())
        psi_vals = [d["psi"] for d in drift_report["features"].values()]
        colors_list = []
        for p in psi_vals:
            if p >= 0.25:
                colors_list.append("#EF4444")  # Red: Critical
            elif p >= 0.10:
                colors_list.append("#F59E0B")  # Amber: Warning
            else:
                colors_list.append("#10B981")  # Green: Normal

        fig_psi = go.Figure(go.Bar(
            x=psi_vals,
            y=feats,
            orientation='h',
            marker=dict(color=colors_list),
            text=[f"PSI: {p:.3f}" for p in psi_vals],
            textposition='auto',
        ))
        fig_psi.add_vline(x=0.10, line_dash="dash", line_color="#F59E0B", annotation_text="Warning (0.10)", annotation_position="top right")
        fig_psi.add_vline(x=0.25, line_dash="dash", line_color="#EF4444", annotation_text="Critical Drift (0.25)", annotation_position="top right")
        fig_psi.update_layout(
            xaxis_title="Population Stability Index (PSI)",
            yaxis_title="Monitored Feature",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        fig_psi = get_plotly_dark_theme(fig_psi, height=320)
        st.plotly_chart(fig_psi, use_container_width=True)

    with chart_col2:
        st.markdown("#### 🎯 Overall Health Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=qual_report['data_quality_score'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Data Quality Integrity", 'font': {'size': 15, 'color': '#CBD5E1'}},
            number={'font': {'color': '#38BDF8', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#64748B'},
                'bar': {'color': "#38BDF8"},
                'bgcolor': '#111827',
                'bordercolor': '#1E293B',
                'steps': [
                    {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.25)"},
                    {'range': [60, 85], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.25)"},
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0B0F19",
            plot_bgcolor="#111827",
            font=dict(family="Inter, sans-serif", color="#E2E8F0"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=320
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Interactive Distribution Comparison Overlay
    st.markdown("#### 🔬 Interactive Feature Distribution Overlay (Baseline vs. Current)")
    selected_drift_feat = st.selectbox("Select Feature to Inspect Distribution Shift", feats)

    if selected_drift_feat:
        base_vals = df_base[selected_drift_feat].dropna()
        curr_vals = df_curr[selected_drift_feat].dropna()

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=base_vals,
            name="Baseline (Training Data)",
            opacity=0.6,
            marker_color="#38BDF8",
            nbinsx=30,
        ))
        fig_dist.add_trace(go.Histogram(
            x=curr_vals,
            name="Current (Inference Window)",
            opacity=0.6,
            marker_color="#EC4899",
            nbinsx=30,
        ))
        fig_dist.update_layout(
            barmode='overlay',
            xaxis_title=selected_drift_feat,
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_dist = get_plotly_dark_theme(fig_dist, height=280)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Detailed Statistical Audit Table
    st.markdown("#### 📋 Detailed Feature Drift & KS Statistical Metrics")
    f_rows = []
    for feat, d in drift_report["features"].items():
        f_rows.append({
            "Feature Name": feat,
            "PSI Score": f"{d['psi']:.4f}",
            "Severity Zone": f"🔴 {d['severity']}" if d["severity"] == "CRITICAL" else ("🟡 " + d["severity"] if d["severity"] == "WARNING" else "🟢 Healthy"),
            "KS Statistic": f"{d['ks_statistic']:.4f}",
            "KS p-value": f"{d['ks_pvalue']:.4f}",
            "Baseline Mean ± Std": f"{d['baseline_mean']:.1f} ± {d['baseline_std']:.1f}",
            "Current Mean ± Std": f"{d['current_mean']:.1f} ± {d['current_std']:.1f}",
        })
    st.dataframe(pd.DataFrame(f_rows), hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 3: RETRAINING PIPELINE FROM SCRATCH
# ==============================================================================
with tab_retrain:
    st.markdown("### 🔄 Automated End-to-End Retraining Console")
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px; border-left: 3px solid #10B981;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #34D399; margin-bottom: 6px;">
            Strict Zero Pretrained Weights Architecture
        </div>
        <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
            Triggering retraining dynamically samples fresh non-overlapping domestic consumer billing histories,
            re-initializes neural/statistical weights from scratch, computes multi-model benchmarks, and stages a new
            Candidate version in the Model Registry.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        n_consumers = st.number_input("Synthetic Consumers Cohort", value=100, min_value=40, max_value=300, step=20)
    with col_cfg2:
        n_cycles = st.number_input("Bi-Monthly Cycles / Consumer", value=10, min_value=6, max_value=16, step=2)
    with col_cfg3:
        contamination_rate = st.slider("Target Contamination Rate", min_value=0.05, max_value=0.20, value=0.10, step=0.01)

    if st.button("🚀 Trigger Full Retraining Pipeline", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing fresh synthetic consumer cohorts...")
        
        progress_bar.progress(25, text="Generating realistic seasonal histories & injecting scenario anomalies...")
        data_gen = SyntheticBillGenerator(seed=int(np.random.randint(1000, 9999)))
        train_df, val_df, test_df = data_gen.generate_full_dataset(
            num_consumers=n_consumers,
            cycles_per_consumer=n_cycles,
            output_dir="./data/synthetic"
        )

        progress_bar.progress(60, text="Training Isolation Forest & PyTorch Autoencoder from scratch...")
        benchmarks = train_and_benchmark_all_models(
            num_consumers=n_consumers,
            cycles_per_consumer=n_cycles,
            output_models_dir="./models"
        )

        progress_bar.progress(85, text="Evaluating against test benchmark & registering candidate...")
        cand = registry.register_candidate(
            model_name="Autoencoder_Retrained",
            artifact_path="./models/autoencoder_scratch.pth",
            metrics=benchmarks["autoencoder"],
            description=f"Auto-retrained on {n_consumers} consumers cohort across {n_cycles} cycles"
        )
        progress_bar.progress(100, text="Retraining completed successfully!")

        st.success(f"🎉 **Retraining Finished!** New Candidate Model **v{cand.version}** registered in Model Registry.")

        # Grand Visual: Comparison Bar Chart across Models
        st.markdown("#### 📊 Candidate Benchmark Metrics Comparison")
        comp_df = pd.DataFrame([
            {"Model": "Rule Baseline", **benchmarks["baseline"]},
            {"Model": "Isolation Forest (Scratch)", **benchmarks["isolation_forest"]},
            {"Model": "PyTorch Autoencoder (Scratch)", **benchmarks["autoencoder"]},
        ])

        fig_comp = px.bar(
            comp_df.melt(id_vars=["Model"], value_vars=["precision", "recall", "f1_score", "roc_auc"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            title="Model Evaluation Metrics on Isolated Test Benchmark",
            labels={"value": "Score (0 - 1.0)", "variable": "Metric"},
            color_discrete_sequence=["#38BDF8", "#10B981", "#F59E0B", "#8B5CF6"]
        )
        fig_comp = get_plotly_dark_theme(fig_comp, height=340)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 4: PREDICTION AUDIT TRAIL
# ==============================================================================
with tab_audit:
    st.markdown("### 📜 Immutable Prediction Audit Trail")
    st.caption("SQLite-backed regulatory audit log recording every verified bill, discrepancy amount, and certainty verdict.")

    stats = audit_logger.get_audit_summary_stats()

    # Audit KPI Summary Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Bills Audited", stats["total_bills_audited"])
    with col_k2:
        st.metric("Anomalies Flagged", stats["anomalies_flagged"])
    with col_k3:
        st.metric("Anomaly Rate", f"{stats['anomaly_rate_pct']:.1f}%")
    with col_k4:
        st.metric("Average Discrepancy", format_currency(stats["average_discrepancy"]))

    st.markdown("---")

    # Fetch recent logs
    recent_logs = audit_logger.fetch_recent_logs(limit=100)

    if recent_logs:
        audit_df = pd.DataFrame(recent_logs)

        # Filters
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            all_cats = ["All"] + list(audit_df["anomaly_category"].dropna().unique())
            sel_cat = st.selectbox("Filter by Anomaly Category", all_cats)
        with f_col2:
            search_query = st.text_input("Search by Consumer Number", value="")

        filtered_df = audit_df.copy()
        if sel_cat != "All":
            filtered_df = filtered_df[filtered_df["anomaly_category"] == sel_cat]
        if search_query:
            filtered_df = filtered_df[filtered_df["consumer_number"].str.contains(search_query, case=False, na=False)]

        # Visual Table Formatting
        display_audit = pd.DataFrame()
        display_audit["ID"] = filtered_df["id"]
        display_audit["Timestamp"] = filtered_df["timestamp"].str[:19].str.replace("T", " ")
        display_audit["Consumer No"] = filtered_df["consumer_number"]
        display_audit["Units (kWh)"] = filtered_df["calculated_units"]
        display_audit["Calculated Total"] = filtered_df["calculated_total"].apply(format_currency)
        display_audit["Billed Total"] = filtered_df["billed_amount"].apply(format_currency)
        display_audit["Discrepancy"] = filtered_df["discrepancy_amount"].apply(lambda v: f"₹{v:+.2f}" if v is not None else "-")
        display_audit["Anomaly Status"] = filtered_df["is_anomaly"].apply(lambda a: "🚨 Anomaly" if a == 1 else "✅ Verified Normal")
        display_audit["Category"] = filtered_df["anomaly_category"]
        display_audit["Certainty"] = filtered_df["certainty_level"]

        st.dataframe(display_audit, hide_index=True, use_container_width=True)

        # Export CSV Button
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Audit Trail as CSV",
            data=csv_bytes,
            file_name="billguard_audit_trail.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("ℹ️ No audit records found yet. Submit bills in the Bill Verifier page to populate.")
