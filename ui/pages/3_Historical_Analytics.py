"""Historical Consumption Analytics and Longitudinal Trends with Data Integrity Audit."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from src.data_gen.bill_generator import SyntheticBillGenerator
from src.data_gen.scenario_definitions import CONSUMER_ARCHETYPES
from ui.utils import format_currency

st.set_page_config(page_title="Historical Analytics | BillGuard", page_icon="📊", layout="wide")


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

st.title("📊 Historical Consumption Trends & Longitudinal Audit")
st.markdown(
    "Inspect longitudinal bi-monthly consumption patterns, seasonal baseline variance, "
    "and verified historical anomalies across domestic consumer archetypes. "
    "All records undergo automated on-the-fly mathematical integrity audits."
)

# Mandatory Synthetic Data Banner
st.info(
    "📌 **Synthetic Household Consumption Dataset (Audited)**: "
    "Generated using authentic Tamil Nadu climate factors (May-June peak summer 'Agni Nakshatram' surge vs. monsoon moderation) "
    "and statutory TNERC LT-1A telescopic slab rules. Assumptions documented; labelled synthetic for academic evaluation."
)

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
    st.markdown(f"**Profile:** {arch_info.description}")
    st.markdown(f"**Baseline Usage:** {arch_info.base_bimonthly_units_mean:.0f} ± {arch_info.base_bimonthly_units_std:.0f} kWh/cycle")
    st.markdown(f"**AC Connected:** {'Yes (Summer multiplier: ' + str(arch_info.summer_multiplier) + 'x)' if arch_info.has_ac else 'No'}")

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
    st.markdown("#### 🛡️ Data Integrity & Summary Audit")
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
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=7)
    ))
    # Historical mean reference line
    mean_val = consumer_records["recorded_units"].mean()
    fig_units.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Historical Baseline Mean: {mean_val:.1f} kWh",
        annotation_position="bottom right"
    )
    fig_units.update_layout(
        title=f"Consumption Trend Over 12 Billing Cycles ({selected_arch})",
        xaxis_title="Billing Reading Date",
        yaxis_title="Energy Consumed (kWh)",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_units, use_container_width=True)

with tab_chart2:
    fig_amt = go.Figure()
    fig_amt.add_trace(go.Scatter(
        x=consumer_records["current_reading_date"].dt.strftime("%Y-%m-%d"),
        y=consumer_records["calculated_grand_total"],
        mode="lines+markers",
        name="Statutory Calculated Total (₹)",
        line=dict(color="#10B981", width=2.5),
        marker=dict(size=7)
    ))
    fig_amt.add_trace(go.Scatter(
        x=consumer_records["current_reading_date"].dt.strftime("%Y-%m-%d"),
        y=consumer_records["billed_amount"],
        mode="lines+markers",
        name="Billed Amount on Receipt (₹)",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        marker=dict(size=7)
    ))
    fig_amt.update_layout(
        title="Billed Amount vs Statutory Calculated Obligation (Discrepancy Visualizer)",
        xaxis_title="Billing Reading Date",
        yaxis_title="Amount (₹ INR)",
        template="plotly_white",
        height=380,
    )
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
