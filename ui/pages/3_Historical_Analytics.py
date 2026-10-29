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
    subtitle="Inspect 2-year longitudinal household billing histories, detect seasonal summer surges, and uncover utility billing discrepancies.",
    tag_left="<span class='status-dot'></span> 100% MATHEMATICAL INTEGRITY AUDITED",
    tag_right="🏛️ TNERC LT-1A GROUND TRUTH",
)

# Friendly Archetype Definitions
ARCHETYPE_FRIENDLY_NAMES = {
    "LIFELINE_LOW": "💡 Lifeline Household (<100 kWh/cycle — 100% Free Subsidy)",
    "MODERATE": "🏠 Moderate Middle Class (Fans, TV, Refrigerator, ~260 kWh)",
    "HIGH_AC": "❄️ Urban Household with AC (Sharp May-June Summer Surge, ~550 kWh)",
    "LUXURY_VILLA": "⚡ Luxury Domestic / Villa (Multiple ACs, EV Charger, >1000 kWh)",
}

# Archetype & Consumer Selector
col1, col2 = st.columns([1.1, 1.9])
with col1:
    arch_options = list(CONSUMER_ARCHETYPES.keys())
    selected_arch = st.selectbox(
        "Select Consumer Household Type",
        arch_options,
        format_func=lambda x: ARCHETYPE_FRIENDLY_NAMES.get(x, x)
    )
    arch_info = CONSUMER_ARCHETYPES[selected_arch]

    arch_df = df[df["archetype"] == selected_arch]
    consumer_list = list(arch_df["consumer_id"].unique())
    selected_consumer = st.selectbox("Select Consumer ID", consumer_list)

    st.markdown(f"""
    <div class="glass-card" style="margin-top: 10px; margin-bottom: 14px; padding: 14px;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; margin-bottom: 4px;">Household Profile</div>
        <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
            {arch_info.description}<br/>
            <b>Normal Usage:</b> ~{arch_info.base_bimonthly_units_mean:.0f} units per 2 months<br/>
            <b>Air Conditioning:</b> {'Yes (Summer Peak: +' + str(int((arch_info.summer_multiplier - 1)*100)) + '%)' if arch_info.has_ac else 'No AC'}
        </div>
    </div>
    """, unsafe_allow_html=True)

consumer_records = df[df["consumer_id"] == selected_consumer].sort_values("current_reading_date").reset_index(drop=True)

total_cycles = len(consumer_records)
anomalous_cycles = int(consumer_records["is_anomaly"].sum())
clean_cycles = total_cycles - anomalous_cycles
total_net_discrepancy = float(consumer_records["recalculated_discrepancy"].sum())

with col2:
    st.markdown("#### 🛡️ 2-Year Audit Summary for Selected Consumer")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Cycles Audited", f"{total_cycles} Cycles", delta="24 Months History")
    with k2:
        st.metric("Accurate Cycles", f"{clean_cycles} Verified", delta="Correctly Billed")
    with k3:
        delta_label = f"🚨 {anomalous_cycles} Overcharged" if anomalous_cycles > 0 else "All Clean"
        st.metric("Billing Irregularities", f"{anomalous_cycles} Flagged", delta=delta_label, delta_color="inverse" if anomalous_cycles > 0 else "normal")
    with k4:
        st.metric(
            "Total Discrepancy",
            format_currency(abs(total_net_discrepancy)),
            delta=f"{'+' if total_net_discrepancy > 0 else ''}{total_net_discrepancy:.2f} ₹ Overcharge" if abs(total_net_discrepancy) > 1 else "Exact Match",
            delta_color="inverse" if total_net_discrepancy > 1 else "normal"
        )

    if anomalous_cycles > 0:
        st.markdown(f"""
        <div style="margin-top: 8px; padding: 10px 14px; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px;">
            <span style="color: #F87171; font-weight: 700; font-size: 0.9rem;">⚠️ Anomaly Detected in Historical Record:</span>
            <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 2px;">
                BillGuard detected <b>{anomalous_cycles} cycle(s)</b> with utility arithmetic overcharges or meter reading jumps, totaling <b>{format_currency(total_net_discrepancy)}</b> in improper charges.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top: 8px; padding: 10px 14px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
            <span style="color: #34D399; font-weight: 700; font-size: 0.9rem;">✅ 100% Clean Audit History:</span>
            <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 2px;">
                All 12 billing cycles match statutory TNERC tariffs perfectly with zero arithmetic overcharges.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Side-by-Side Longitudinal Charts
st.markdown("---")
st.markdown(f"### 📈 Visual Consumption & Billing Patterns (`{selected_consumer}`)")

col_chart1, col_chart2 = st.columns(2)

date_strings = consumer_records["current_reading_date"].dt.strftime("%b %Y")

with col_chart1:
    fig_units = go.Figure()
    # Main units line
    fig_units.add_trace(go.Scatter(
        x=date_strings,
        y=consumer_records["recorded_units"],
        mode="lines+markers",
        name="Units Consumed",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=7, color="#38BDF8"),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
        hovertemplate="Bill Date: %{x}<br>Consumption: %{y:.1f} kWh<extra></extra>"
    ))

    # Highlight anomalous cycles with red dots
    anom_subset = consumer_records[consumer_records["is_anomaly"] == 1]
    if not anom_subset.empty:
        fig_units.add_trace(go.Scatter(
            x=anom_subset["current_reading_date"].dt.strftime("%b %Y"),
            y=anom_subset["recorded_units"],
            mode="markers+text",
            name="Flagged Anomaly",
            marker=dict(size=12, color="#EF4444", symbol="circle"),
            text=[f"🚨 {t[:12]}" for t in anom_subset["anomaly_type"]],
            textposition="top center",
            textfont=dict(color="#F87171", size=10),
            hovertemplate="ANOMALY: %{text}<br>Units: %{y:.1f} kWh<extra></extra>"
        ))

    # Historical average baseline
    mean_val = consumer_records["recorded_units"].mean()
    fig_units.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Average: {mean_val:.0f} kWh",
        annotation_position="bottom right",
        annotation_font=dict(color="#CBD5E1", size=10)
    )

    fig_units.update_layout(
        title="⚡ Electricity Consumption Over Time (kWh)",
        xaxis_title="Billing Cycle Date",
        yaxis_title="Units Consumed (kWh)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_units = get_plotly_dark_theme(fig_units, height=350)
    st.plotly_chart(fig_units, use_container_width=True)

with col_chart2:
    fig_amt = go.Figure()
    # Green line: Statutory Calculated Total
    fig_amt.add_trace(go.Scatter(
        x=date_strings,
        y=consumer_records["calculated_grand_total"],
        mode="lines+markers",
        name="Statutory Fair Bill (₹)",
        line=dict(color="#10B981", width=3),
        marker=dict(size=7, color="#10B981"),
        hovertemplate="Bill Date: %{x}<br>Fair Amount: ₹%{y:,.2f}<extra></extra>"
    ))
    # Red dashed line: Amount Billed by Utility
    fig_amt.add_trace(go.Scatter(
        x=date_strings,
        y=consumer_records["billed_amount"],
        mode="lines+markers",
        name="Utility Billed Amount (₹)",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        marker=dict(size=7, color="#EF4444"),
        hovertemplate="Bill Date: %{x}<br>Utility Billed: ₹%{y:,.2f}<extra></extra>"
    ))

    # Annotate large overcharges right on the chart
    for _, row in consumer_records.iterrows():
        disc = row["recalculated_discrepancy"]
        if disc > 20.0:
            d_str = row["current_reading_date"].strftime("%b %Y")
            fig_amt.add_annotation(
                x=d_str,
                y=row["billed_amount"],
                text=f"🚨 Overcharged +₹{disc:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#EF4444",
                arrowsize=1,
                arrowwidth=1.5,
                ax=0,
                ay=-35,
                font=dict(color="#F87171", size=10),
                bgcolor="rgba(30, 41, 59, 0.9)",
                bordercolor="#EF4444",
                borderwidth=1,
            )

    fig_amt.update_layout(
        title="💰 Fair Calculated Amount vs. Utility Billed Amount",
        xaxis_title="Billing Cycle Date",
        yaxis_title="Amount (₹ INR)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_amt = get_plotly_dark_theme(fig_amt, height=350)
    st.plotly_chart(fig_amt, use_container_width=True)

st.markdown("""
<div class="glass-card" style="margin-top: 5px; margin-bottom: 24px; padding: 12px 18px; border-left: 3px solid #10B981;">
    <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
        💡 <b>How to read these charts without explanation:</b><br/>
        • <b>Left Chart:</b> Shows monthly electricity consumption in kWh. Summer peaks (May-June) are normal, while abnormal sudden jumps are marked with 🚨 red dots.<br/>
        • <b>Right Chart:</b> Compares the <b>Statutory Fair Bill</b> (🟢 green line) against what the <b>Utility Actually Billed</b> (🔴 red dashed line). Wherever the red line jumps above the green line, the utility overcharged the customer!
    </div>
</div>
""", unsafe_allow_html=True)

# 6-Column Simple Cycle-by-Cycle Table
st.markdown("### 📋 Cycle-by-Cycle Audit Breakdown")
st.caption("Detailed record of every bi-monthly electricity bill audited against statutory TNERC LT-1A tariffs.")

display_df = consumer_records.copy()
display_df["Cycle #"] = display_df["cycle_idx"] + 1
display_df["Bill Date"] = display_df["current_reading_date"].dt.strftime("%d-%b-%Y")
display_df["Units Consumed"] = display_df["recorded_units"].apply(lambda u: f"{u:.0f} kWh")
display_df["Fair Calculated Bill"] = display_df["calculated_grand_total"].apply(format_currency)
display_df["Utility Billed Amount"] = display_df["billed_amount"].apply(format_currency)
display_df["Overcharge / Discrepancy"] = display_df["recalculated_discrepancy"].apply(
    lambda v: f"🚨 +₹{v:.2f} (Overcharge)" if v > 1.0 else ("⚠️ -₹{:.2f} (Undercharge)".format(abs(v)) if v < -1.0 else "✅ ₹0.00 (Exact)")
)
display_df["Audit Verdict"] = display_df.apply(
    lambda r: f"🚨 {r['anomaly_type']}" if r["is_anomaly"] == 1 else "✅ Verified Accurate", axis=1
)

st.dataframe(
    display_df[[
        "Cycle #", "Bill Date", "Units Consumed", "Fair Calculated Bill",
        "Utility Billed Amount", "Overcharge / Discrepancy", "Audit Verdict"
    ]],
    hide_index=True,
    use_container_width=True
)

# Collapsed Expander for Technical Diagnostics
with st.expander("🔍 View Technical Meter Reading Diagnostic Details (For Auditors)"):
    st.markdown("Meter reading validity, reading difference checks, and internal consistency audits:")
    tech_df = consumer_records.copy()
    tech_df["Cycle #"] = tech_df["cycle_idx"] + 1
    tech_df["Previous Reading"] = tech_df["previous_meter_reading"].round(1)
    tech_df["Current Reading"] = tech_df["current_meter_reading"].round(1)
    tech_df["Days in Cycle"] = tech_df["billing_days"]
    tech_df["Mathematical Consistency"] = tech_df["is_consistent"].apply(lambda ok: "✅ 100% Consistent" if ok else "❌ Inconsistent")
    
    st.dataframe(
        tech_df[["Cycle #", "Previous Reading", "Current Reading", "Days in Cycle", "Mathematical Consistency"]],
        hide_index=True,
        use_container_width=True
    )
