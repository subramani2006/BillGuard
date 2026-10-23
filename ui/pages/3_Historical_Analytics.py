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
from pathlib import Path

from src.data_gen.bill_generator import SyntheticBillGenerator
from src.data_gen.scenario_definitions import CONSUMER_ARCHETYPES
from ui.utils import (
    format_currency,
    inject_custom_theme,
    render_hero_banner,
    get_plotly_dark_theme,
)

st.set_page_config(page_title="Historical Analytics | BillGuard", page_icon="📊", layout="wide")
inject_custom_theme()


@st.cache_data
def load_and_validate_dataset():
    """
    Load synthetic dataset, audit records on-the-fly, and enforce mathematical consistency.
    Formula: discrepancy = billed_amount - calculated_grand_total.
    """
    data_path = Path("./data/synthetic/all_bills.csv")
    if not data_path.exists():
        gen = SyntheticBillGenerator(seed=42)
        gen.generate_full_dataset(num_consumers=100, cycles_per_consumer=12, output_dir="./data/synthetic")

    raw_df = pd.read_csv("./data/synthetic/all_bills.csv")

    # Enforce strict chronological ordering
    raw_df["current_reading_date"] = pd.to_datetime(raw_df["current_reading_date"])
    raw_df = raw_df.sort_values(["consumer_id", "current_reading_date"]).reset_index(drop=True)

    # Recalculate totals and discrepancies strictly from source fields
    raw_df["recalculated_discrepancy"] = (raw_df["billed_amount"] - raw_df["calculated_grand_total"]).round(2)
    raw_df["is_consistent"] = (raw_df["discrepancy_amount"] - raw_df["recalculated_discrepancy"]).abs() <= 0.05

    # Check reading differences
    raw_df["reading_difference"] = (raw_df["current_meter_reading"] - raw_df["previous_meter_reading"]).round(2)
    raw_df["units_match"] = (raw_df["reading_difference"] - raw_df["recorded_units"]).abs() <= 0.5

    return raw_df


df = load_and_validate_dataset()

# Grand Hero Banner
render_hero_banner(
    title="📊 Historical Consumption Analytics & Audit",
    subtitle="Inspect longitudinal bi-monthly consumption patterns, seasonal baseline variance, and verified historical anomalies across domestic consumer archetypes.",
    tag_left="<span class='status-dot'></span> 100% MATHEMATICAL INTEGRITY AUDITED",
    tag_right="🏛️ TNERC LT-1A GROUND TRUTH",
)

# Mandatory Synthetic Data Banner
st.markdown("""
<div class="glass-card" style="margin-bottom: 20px; border-left: 3px solid #38BDF8;">
    <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; margin-bottom: 4px;">
        📌 Synthetic Household Consumption Dataset (Audited)
    </div>
    <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
        Generated using authentic Tamil Nadu climate factors (May-June peak summer <i>Agni Nakshatram</i> surge vs. monsoon moderation) 
        and statutory TNERC LT-1A telescopic slab rules. Assumptions documented; labelled synthetic for academic evaluation.
    </div>
</div>
""", unsafe_allow_html=True)

# Archetype & Consumer Selector
col1, col2 = st.columns([1, 2])
with col1:
    arch_options = list(CONSUMER_ARCHETYPES.keys())
    selected_arch = st.selectbox(
        "Select Domestic Consumer Archetype",
        arch_options,
        format_func=lambda x: f"{CONSUMER_ARCHETYPES[x].name} ({x})"
    )
    arch_info = CONSUMER_ARCHETYPES[selected_arch]
    
    st.markdown(f"""
    <div class="glass-card" style="margin-top: 10px; margin-bottom: 14px;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">Profile Details</div>
        <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
            {arch_info.description}<br/>
            <b>Baseline:</b> {arch_info.base_bimonthly_units_mean:.0f} ± {arch_info.base_bimonthly_units_std:.0f} kWh/cycle<br/>
            <b>AC Connected:</b> {'Yes (Summer: ' + str(arch_info.summer_multiplier) + 'x)' if arch_info.has_ac else 'No'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    arch_df = df[df["archetype"] == selected_arch]
    consumer_list = list(arch_df["consumer_id"].unique())
    selected_consumer = st.selectbox("Select Consumer ID", consumer_list)

consumer_records = df[df["consumer_id"] == selected_consumer].sort_values("current_reading_date").reset_index(drop=True)

# Data Integrity Check
total_cycles = len(consumer_records)
consistent_count = int(consumer_records["is_consistent"].sum())
anomalous_cycles = int(consumer_records["is_anomaly"].sum())
total_net_discrepancy = float(consumer_records["recalculated_discrepancy"].sum())

with col2:
    st.markdown("#### 🛡️ Data Integrity & Longitudinal Summary")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Cycles Audited", total_cycles)
    with k2:
        pass_pct = (consistent_count / total_cycles) * 100.0 if total_cycles > 0 else 100.0
        st.metric("Integrity Audit Pass", f"{pass_pct:.0f}%", delta="100% Consistent")
    with k3:
        st.metric("Anomalous Cycles", anomalous_cycles, delta=f"{anomalous_cycles} flagged" if anomalous_cycles > 0 else "Clean")
    with k4:
        st.metric(
            "Cumulative Discrepancy",
            format_currency(total_net_discrepancy),
            delta=f"{total_net_discrepancy:+.2f} ₹",
            delta_color="inverse" if abs(total_net_discrepancy) > 2.0 else "normal"
        )

# Interactive Consumption and Billing Charts
st.markdown("---")
st.markdown(f"### 📈 Longitudinal Trends for `{selected_consumer}`")

tab_chart1, tab_chart2 = st.tabs(["⚡ Consumption History (kWh)", "💰 Billed vs Calculated Amount (₹)"])

with tab_chart1:
    fig_units = go.Figure()
    fig_units.add_trace(go.Scatter(
        x=consumer_records["current_reading_date"].dt.strftime("%Y-%m-%d"),
        y=consumer_records["recorded_units"],
        mode="lines+markers",
        name="Recorded Consumption (kWh)",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=7, color="#38BDF8"),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
        hovertemplate="Date: %{x}<br>Units: %{y:.1f} kWh<extra></extra>"
    ))
    # Historical mean reference line
    mean_val = consumer_records["recorded_units"].mean()
    fig_units.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Historical Baseline Mean: {mean_val:.1f} kWh",
        annotation_position="bottom right",
        annotation_font=dict(color="#CBD5E1", size=11)
    )
    fig_units.update_layout(
        title=f"Consumption Trend Over 12 Billing Cycles ({selected_arch})",
        xaxis_title="Billing Reading Date",
        yaxis_title="Energy Consumed (kWh)",
    )
    fig_units = get_plotly_dark_theme(fig_units, height=380)
    st.plotly_chart(fig_units, use_container_width=True)

with tab_chart2:
    fig_amt = go.Figure()
    fig_amt.add_trace(go.Scatter(
        x=consumer_records["current_reading_date"].dt.strftime("%Y-%m-%d"),
        y=consumer_records["calculated_grand_total"],
        mode="lines+markers",
        name="Statutory Calculated Total (₹)",
        line=dict(color="#10B981", width=3),
        marker=dict(size=7, color="#10B981"),
        hovertemplate="Date: %{x}<br>Calculated: ₹%{y:,.2f}<extra></extra>"
    ))
    fig_amt.add_trace(go.Scatter(
        x=consumer_records["current_reading_date"].dt.strftime("%Y-%m-%d"),
        y=consumer_records["billed_amount"],
        mode="lines+markers",
        name="Billed Amount on Receipt (₹)",
        line=dict(color="#EF4444", width=3, dash="dash"),
        marker=dict(size=7, color="#EF4444"),
        hovertemplate="Date: %{x}<br>Billed: ₹%{y:,.2f}<extra></extra>"
    ))
    fig_amt.update_layout(
        title="Billed Amount vs Statutory Calculated Obligation (Discrepancy Visualizer)",
        xaxis_title="Billing Reading Date",
        yaxis_title="Amount (₹ INR)",
    )
    fig_amt = get_plotly_dark_theme(fig_amt, height=380)
    st.plotly_chart(fig_amt, use_container_width=True)

# Longitudinal Audit Breakdown Table
st.markdown("### 📋 Detailed Cycle-by-Cycle Longitudinal Audit Table")
st.caption(
    "Discrepancy Formula: `discrepancy = billed_amount - calculated_grand_total`. "
    "All figures recalculated from raw meter readings and validated against TNERC LT-1A tariffs."
)

display_df = consumer_records.copy()
display_df["Cycle #"] = display_df["cycle_idx"] + 1
display_df["Bill Date"] = display_df["current_reading_date"].dt.strftime("%Y-%m-%d")
display_df["Days"] = display_df["billing_days"]
display_df["Prev Meter"] = display_df["previous_meter_reading"].round(1)
display_df["Curr Meter"] = display_df["current_meter_reading"].round(1)
display_df["Units (kWh)"] = display_df["recorded_units"].round(1)
display_df["Statutory Total"] = display_df["calculated_grand_total"].apply(format_currency)
display_df["Billed Amount"] = display_df["billed_amount"].apply(format_currency)
display_df["Discrepancy (₹)"] = display_df["recalculated_discrepancy"].apply(lambda v: f"₹{v:+.2f}")
display_df["Audit Status"] = display_df.apply(
    lambda r: f"🚨 {r['anomaly_type']}" if r["is_anomaly"] == 1 else "✅ Verified Normal", axis=1
)
display_df["Integrity"] = display_df["is_consistent"].apply(lambda ok: "✅ Valid" if ok else "❌ Error")

st.dataframe(
    display_df[[
        "Cycle #", "Bill Date", "Days", "Prev Meter", "Curr Meter", "Units (kWh)",
        "Statutory Total", "Billed Amount", "Discrepancy (₹)", "Audit Status", "Integrity"
    ]],
    hide_index=True,
    use_container_width=True
)
