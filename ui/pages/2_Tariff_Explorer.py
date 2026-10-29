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
    title="⚡ Tamil Nadu Electricity Tariff Explorer",
    subtitle="Understand statutory TNERC LT-1A tariffs, the GoTN 100-unit free lifeline subsidy, and simulate how billing delays cause unfair slab jumps.",
    tag_left="<span class='status-dot'></span> STATUTORY TARIFF INTELLIGENCE",
    tag_right="🏛️ TNERC GAZETTE VERIFIED",
)

# 1. Simple Tariff Selector
tariffs = engine.list_available_tariffs()
tariff_options = {
    t["tariff_code"]: f"{'🏛️' if t.get('verification_status') == 'VERIFIED_OFFICIAL' else '🧪'} {t['tariff_name']}"
    for t in tariffs
}
# Default to active 2024 order if present
default_code = "TN_DOMESTIC_LT1A_2024" if "TN_DOMESTIC_LT1A_2024" in tariff_options else list(tariff_options.keys())[0]
default_idx = list(tariff_options.keys()).index(default_code)

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_code = st.selectbox(
        "Select Tariff Schedule",
        options=list(tariff_options.keys()),
        index=default_idx,
        format_func=lambda x: tariff_options[x]
    )

tariff_cfg = engine.get_tariff(selected_code)
ver_status = getattr(tariff_cfg, "verification_status", "VERIFIED_OFFICIAL")

with col_sel2:
    if ver_status == "VERIFIED_OFFICIAL":
        st.markdown("""
        <div style="margin-top: 28px; padding: 10px 14px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
            <span style="color: #34D399; font-weight: 700; font-size: 0.9rem;">✅ Official Statutory Tariff</span>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 2px;">TNERC Gazette Order & Escalation Notice</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top: 28px; padding: 10px 14px; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px;">
            <span style="color: #FBBF24; font-weight: 700; font-size: 0.9rem;">🧪 Experimental / Simulation</span>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 2px;">Research model for solar/progressive rates</div>
        </div>
        """, unsafe_allow_html=True)

# 2. Key Tariff Rules at a Glance (3 Simple Visual Cards)
cat_name = getattr(tariff_cfg, "tariff_category", "LT-1A Domestic")
cycle_days = getattr(tariff_cfg, "billing_cycle_days", 60)
free_units = getattr(tariff_cfg, "free_allowance_units", 100)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">Category & Authority</div>
        <div class="card-value" style="font-size: 1.25rem;">{cat_name}</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 6px;">
            Regulator: <b style="color: #38BDF8;">TNERC</b> | State: <b style="color: #E2E8F0;">Tamil Nadu</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">Official Billing Cycle</div>
        <div class="card-value" style="font-size: 1.25rem; color: #38BDF8;">{cycle_days} Days (Bi-Monthly)</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 6px;">
            Standard Schedule: <b style="color: #E2E8F0;">6 Cycles per Year</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">Lifeline Subsidy</div>
        <div class="card-value" style="font-size: 1.25rem; color: #34D399;">100 Units FREE</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 6px;">
            100% Free Quota: <b style="color: #34D399;">GoTN Subsidy (₹0.00)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# 3. Slab Structure - Simple & Clear
st.markdown("### 📋 Official Slab Structure & Subsidy Rates")
st.caption("How your bill is calculated: Tamil Nadu uses a telescopic slab structure where every domestic household gets the first 100 units completely free.")

slabs_data = []
for s in tariff_cfg.slabs:
    slab_range = f"{s.min_units} - {s.max_units} kWh" if s.max_units is not None else f"Above {s.min_units - 1} kWh"
    
    if s.net_rate == 0.0:
        pay_str = "🎉 ₹0.00 (100% FREE)"
        subsidy_str = "100% Free Subsidy (GoTN)"
    elif s.subsidy_per_unit > 0:
        pay_str = f"₹{s.net_rate:.2f} / unit"
        subsidy_str = f"GoTN Pays ₹{s.subsidy_per_unit:.2f} / unit"
    else:
        pay_str = f"₹{s.net_rate:.2f} / unit"
        subsidy_str = "No Subsidy"
        
    slabs_data.append({
        "Consumption Slab": slab_range,
        "Base TNERC Rate": f"₹{s.rate_per_unit:.2f} / unit",
        "Government Subsidy": subsidy_str,
        "What You Pay": pay_str,
    })

st.dataframe(pd.DataFrame(slabs_data), hide_index=True, use_container_width=True)

st.markdown("""
<div class="glass-card" style="margin-top: 8px; margin-bottom: 24px; padding: 12px 18px; border-left: 3px solid #38BDF8;">
    <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
        💡 <b>How Telescopic Billing Works:</b> 
        If you consume <b>250 units</b> in a 60-day cycle:
        • First <b>100 units</b> are <b>₹0.00 (FREE)</b>.<br/>
        • Next <b>100 units</b> (101 to 200) are charged at <b>₹2.47 / unit = ₹247.00</b>.<br/>
        • Remaining <b>50 units</b> (201 to 250) are charged at <b>₹4.93 / unit = ₹246.50</b>.<br/>
        • Total Energy Charge = <b>₹493.50</b> (+ ₹0.32/kWh FPPCA on 150 billable units = ₹48.00) $\\to$ <b>Total: ₹541.50</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Interactive Billing Cycle Delay Simulator
st.markdown("---")
st.markdown("### ⏱️ Delayed Meter Reading Impact Simulator")
st.markdown(
    "In Tamil Nadu, one of the most common consumer grievances is **delayed meter reading** (e.g. meter read after 75 or 80 days instead of 60 days). "
    "Because slabs are fixed at 60 days, delayed readings push regular daily consumption into **expensive higher tiers**, unfairly inflating the bill."
)

col_s1, col_s2 = st.columns(2)
with col_s1:
    sim_units = st.slider("Total Units Consumed (kWh)", min_value=100, max_value=800, value=350, step=10)
with col_s2:
    sim_days = st.slider("Actual Days Between Meter Readings", min_value=50, max_value=90, value=75, step=1,
                         help="Standard cycle is 60 days. Select 70+ days to see the delayed reading penalty.")

# Accurate calculation with date propagation
base_date = pd.to_datetime("2024-06-01")
curr_date_actual = (base_date + pd.Timedelta(days=sim_days)).strftime("%Y-%m-%d")

b_sim = BillInput(
    consumer_number="SIM-DELAY",
    previous_reading_date="2024-06-01",
    current_reading_date=curr_date_actual,
    previous_meter_reading=1000.0,
    current_meter_reading=1000.0 + sim_units,
    tariff_code=selected_code,
)
res_unprorated = engine.calculate_bill(b_sim, simulate_proration=False)
res_prorated = engine.calculate_bill(b_sim, simulate_proration=True)
diff_proration = max(0.0, round(res_unprorated.calculated_grand_total - res_prorated.calculated_grand_total, 2))

# 3 Side-by-Side Comparison Cards
col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    st.markdown(f"""
    <div class="glass-card" style="border: 1px solid rgba(16, 185, 129, 0.4); text-align: center; padding: 18px;">
        <div class="card-label" style="color: #34D399;">🗓️ Fair Prorated Bill (Statutory Right)</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #34D399; margin: 8px 0;">{format_currency(res_prorated.calculated_grand_total)}</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">Adjusts slab quotas proportionally for {sim_days} days</div>
    </div>
    """, unsafe_allow_html=True)

with col_r2:
    st.markdown(f"""
    <div class="glass-card" style="border: 1px solid rgba(239, 68, 68, 0.4); text-align: center; padding: 18px;">
        <div class="card-label" style="color: #F87171;">⏱️ Un-Prorated Bill (Utility Billed)</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #F87171; margin: 8px 0;">{format_currency(res_unprorated.calculated_grand_total)}</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">Raw {sim_days}-day units squeezed into 60-day slabs</div>
    </div>
    """, unsafe_allow_html=True)

with col_r3:
    penalty_color = "#EF4444" if diff_proration > 0 else "#10B981"
    penalty_text = f"+₹{diff_proration:.2f} (OVERCHARGE)" if diff_proration > 0 else "₹0.00 (NO PENALTY)"
    st.markdown(f"""
    <div class="glass-card" style="border: 1.5px solid {penalty_color}; text-align: center; padding: 18px; background: rgba(239, 68, 68, 0.08) if {diff_proration > 0} else 'transparent';">
        <div class="card-label" style="color: {penalty_color};">⚠️ Unfair Delay Surcharge</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {penalty_color}; margin: 8px 0;">{penalty_text}</div>
        <div style="font-size: 0.85rem; color: #CBD5E1;">{'Refund Claimable by Consumer' if diff_proration > 0 else 'On-Time Reading'}</div>
    </div>
    """, unsafe_allow_html=True)

if diff_proration > 0:
    st.warning(
        f"🚨 **Why you are being overcharged by ₹{diff_proration:.2f}:**\n"
        f"Your meter was read **{sim_days - 60} days late** ({sim_days} days instead of 60 days). "
        f"Because TANGEDCO did not expand your slab quotas proportionally, your normal daily electricity usage crossed into higher pricing tiers (₹6.60 or ₹8.80/unit). "
        f"**BillGuard's Grievance Generator generates an official petition citing TNERC regulations to claim this ₹{diff_proration:.2f} refund.**"
    )

# 5. Bill Progression Curve
st.markdown("---")
st.markdown("### 📈 Domestic Electricity Cost Curve (0 to 1,000 kWh)")
st.caption("Visualizing the steepness of electricity costs as household consumption increases under Tamil Nadu's progressive tariff structure.")

curve_units = list(range(0, 1001, 20))
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
    mode="lines",
    name="Total Bimonthly Bill",
    line=dict(color="#38BDF8", width=3),
    fill="tozeroy",
    fillcolor="rgba(56, 189, 248, 0.08)",
    hovertemplate="Consumption: %{x} kWh<br>Bill Amount: ₹%{y:,.2f}<extra></extra>"
))

# Key threshold vertical lines
fig_curve.add_vline(
    x=100,
    line_dash="dot",
    line_color="#10B981",
    annotation_text="100 kWh: 100% FREE ZONE",
    annotation_position="top left",
    annotation_font=dict(color="#34D399", size=11, family="Inter")
)
fig_curve.add_vline(
    x=500,
    line_dash="dot",
    line_color="#F59E0B",
    annotation_text="500 kWh: High Usage Threshold",
    annotation_position="top left",
    annotation_font=dict(color="#FBBF24", size=11, family="Inter")
)

fig_curve.update_layout(
    title=f"Bimonthly Bill Obligation Curve ({tariff_cfg.tariff_name})",
    xaxis_title="Energy Consumed (kWh / 60 Days)",
    yaxis_title="Total Bill Amount (₹ INR)",
    margin=dict(l=20, r=20, t=40, b=20),
)
fig_curve = get_plotly_dark_theme(fig_curve, height=360)
st.plotly_chart(fig_curve, use_container_width=True)

st.markdown("""
<div class="glass-card" style="margin-top: 10px; margin-bottom: 20px; padding: 14px 20px; border-left: 3px solid #10B981;">
    <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.7;">
        <b>📌 Summary of Cost Curve Milestones:</b><br/>
        • <b>0 to 100 kWh:</b> Total bill is <b>₹0.00</b> (Full lifeline subsidy provided by Tamil Nadu Govt).<br/>
        • <b>101 to 400 kWh:</b> Moderate, affordable domestic rate (₹2.47 to ₹4.93 per unit).<br/>
        • <b>Above 500 kWh:</b> Curve steepens sharply (rates jump up to ₹12.10/unit for luxury consumption).
    </div>
</div>
""", unsafe_allow_html=True)
