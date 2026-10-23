"""Tariff Explorer and Proration Impact Simulator."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.tariff.engine import TariffEngine
from src.tariff.models import BillInput
from ui.utils import (
    format_currency,
    inject_custom_theme,
    render_hero_banner,
    get_plotly_dark_theme,
)

st.set_page_config(page_title="Tariff Explorer | BillGuard", page_icon="⚡", layout="wide")
inject_custom_theme()

@st.cache_resource
def get_engine():
    return TariffEngine()

engine = get_engine()

# Grand Hero Banner
render_hero_banner(
    title="⚡ Tamil Nadu Tariff Explorer & Simulator",
    subtitle="Explore official statutory TNERC LT-1A tariff schedules, GoTN 100-unit lifeline subsidies, and simulate proration impact for delayed billing cycles.",
    tag_left="<span class='status-dot'></span> VERSIONED TARIFF INTELLIGENCE",
    tag_right="🏛️ TNERC GAZETTE VERIFIED",
)

tariffs = engine.list_available_tariffs()
official_tariffs = [t for t in tariffs if t.get("verification_status") == "VERIFIED_OFFICIAL"]
simulated_tariffs = [t for t in tariffs if t.get("verification_status") != "VERIFIED_OFFICIAL"]

# Grouped selection
st.markdown("### 🏛️ Select Tariff Configuration")
tariff_type = st.radio(
    "Tariff Source Category",
    ["Verified Official Statutory Tariffs (TNERC)", "Simulated / Experimental Research Tariffs"],
    horizontal=True
)

if "Verified" in tariff_type:
    available_subset = official_tariffs
else:
    available_subset = simulated_tariffs

tariff_map = {t["tariff_code"]: t for t in available_subset}
selected_code = st.selectbox(
    "Select Tariff Version to Inspect",
    list(tariff_map.keys()),
    format_func=lambda x: f"{tariff_map[x]['tariff_name']} ({tariff_map[x]['verification_status']})"
)

tariff_cfg = engine.get_tariff(selected_code)

# Mandatory simulation banner if tariff is not official
if tariff_cfg.verification_status != "VERIFIED_OFFICIAL":
    st.warning("⚠️ **Simulation only — actual charges may differ. Verify the applicable official tariff.**")
    st.caption("Never describe an unverified calculation as a statutory or official bill result.")
else:
    st.success("✅ **Official Statutory Tariff**: Verified against official TNERC tariff orders and Gazette notifications.")

# Comprehensive Metadata Cards
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">Category & Authority</div>
        <div class="card-value" style="font-size: 1.3rem;">{tariff_cfg.tariff_category}</div>
        <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px;">
            State: <b style="color: #E2E8F0;">{tariff_cfg.state}</b><br/>
            Authority: <b style="color: #38BDF8;">{tariff_cfg.authority}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">Statutory Order & Period</div>
        <div class="card-value" style="font-size: 1.15rem; color: #38BDF8;">{tariff_cfg.order_reference[:28]}</div>
        <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px;">
            Effective: <b style="color: #E2E8F0;">{tariff_cfg.effective_start_date or 'N/A'}</b> to <b style="color: #E2E8F0;">{tariff_cfg.effective_end_date or 'Present'}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">Cycle & Lifeline Allowance</div>
        <div class="card-value" style="font-size: 1.3rem; color: #34D399;">{tariff_cfg.billing_cycle_days} Days</div>
        <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px;">
            Schedule: <b style="color: #E2E8F0;">{tariff_cfg.billing_cycle_type}</b><br/>
            Subsidy: <b style="color: #34D399;">{tariff_cfg.free_allowance_units} Units Free (GoTN)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Policy & Calculation Methodology Card
with st.expander("📜 Subsidy Eligibility & Calculation Methodology", expanded=True):
    st.markdown(f"**Subsidy Eligibility & Conditions:**\n{tariff_cfg.subsidy_conditions}")
    st.markdown(f"**Calculation Methodology:**\n{tariff_cfg.calculation_methodology}")
    if tariff_cfg.fppca.enabled:
        st.markdown(f"**FPPCA Fuel Adjustment:** ₹{tariff_cfg.fppca.rate_per_unit:.2f}/kWh ({tariff_cfg.fppca.description})")

st.markdown("### 📋 Versioned Slab Structure")
slabs_data = []
for s in tariff_cfg.slabs:
    slabs_data.append({
        "Min Units": str(s.min_units),
        "Max Units": str(s.max_units) if s.max_units is not None else "Above 1000",
        "Gross Rate (₹/kWh)": f"₹{s.rate_per_unit:.2f}",
        "GoTN Subsidy (₹/kWh)": f"₹{s.subsidy_per_unit:.2f}",
        "Net Consumer Rate (₹/kWh)": f"₹{s.net_rate:.2f}",
        "Description": s.description,
    })
st.dataframe(pd.DataFrame(slabs_data), hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("### 🧮 Interactive Tariff & Billing Cycle Delay Simulator")
st.caption("Delayed meter reading is one of the most common domestic consumer grievances in Tamil Nadu. Simulate how extended billing periods artificially push consumption into higher slabs without proration.")

col_s1, col_s2 = st.columns(2)
with col_s1:
    sim_units = st.slider("Total Units Consumed (kWh)", min_value=0, max_value=1200, value=350, step=10)
with col_s2:
    sim_days = st.slider("Billing Period Duration (Days)", min_value=40, max_value=90, value=60, step=1)

# Compute both un-prorated and prorated bill
b_unprorated = BillInput(
    consumer_number="SIM-001",
    previous_reading_date="2024-06-01",
    current_reading_date="2024-08-01",
    previous_meter_reading=1000.0,
    current_meter_reading=1000.0 + sim_units,
    tariff_code=selected_code,
)
res_unprorated = engine.calculate_bill(b_unprorated, simulate_proration=False)
res_prorated = engine.calculate_bill(b_unprorated, simulate_proration=True)

diff_proration = res_unprorated.calculated_grand_total - res_prorated.calculated_grand_total

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    st.metric("Un-Prorated Bill", format_currency(res_unprorated.calculated_grand_total))
with col_r2:
    st.metric("Prorated Bill (Fair Assessment)", format_currency(res_prorated.calculated_grand_total))
with col_r3:
    st.metric(
        "Proration Penalty / Discrepancy",
        format_currency(diff_proration),
        delta=f"+{diff_proration:.2f} ₹ (Penalty)" if diff_proration > 0 else "0.00 ₹",
        delta_color="inverse" if diff_proration > 0 else "off"
    )
with col_r4:
    effective_rate = res_unprorated.calculated_grand_total / max(1.0, sim_units)
    st.metric("Effective Rate per kWh", f"₹{effective_rate:.2f}")

# Chart of bill progression
st.markdown("### 📈 Cost Curve Progression Across Consumption")
if tariff_cfg.verification_status != "VERIFIED_OFFICIAL":
    st.caption("⚠️ *Simulation only — actual charges may differ. Verify the applicable official tariff.*")

curve_units = list(range(0, 1001, 25))
curve_costs = []
for u in curve_units:
    b_tmp = BillInput(
        consumer_number="SIM-CURVE",
        previous_meter_reading=1000.0,
        current_meter_reading=1000.0 + u,
        tariff_code=selected_code,
    )
    r = engine.calculate_bill(b_tmp)
    curve_costs.append(r.calculated_grand_total)

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(
    x=curve_units,
    y=curve_costs,
    mode="lines+markers",
    name=f"{tariff_cfg.tariff_name}",
    line=dict(color="#38BDF8", width=3),
    marker=dict(size=5, color="#38BDF8"),
    fill="tozeroy",
    fillcolor="rgba(56, 189, 248, 0.08)",
    hovertemplate="Units: %{x} kWh<br>Bill Total: ₹%{y:,.2f}<extra></extra>"
))

# Key threshold vertical lines
fig_curve.add_vline(
    x=100,
    line_dash="dot",
    line_color="#10B981",
    annotation_text="100 kWh (Free Subsidy Limit)",
    annotation_position="top left",
    annotation_font=dict(color="#34D399", size=10)
)
fig_curve.add_vline(
    x=500,
    line_dash="dot",
    line_color="#F59E0B",
    annotation_text="500 kWh (Telescopic Threshold)",
    annotation_position="top left",
    annotation_font=dict(color="#FBBF24", size=10)
)

fig_curve.update_layout(
    title=f"Bimonthly Bill Obligation Curve — {tariff_cfg.tariff_name}",
    xaxis_title="Energy Consumed (kWh)",
    yaxis_title="Total Bill Amount (₹ INR)",
)
fig_curve = get_plotly_dark_theme(fig_curve, height=380)
st.plotly_chart(fig_curve, use_container_width=True)
