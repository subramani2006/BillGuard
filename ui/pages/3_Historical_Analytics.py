"""Historical Consumption Analytics and Longitudinal Trends."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_gen.bill_generator import SyntheticBillGenerator
from src.data_gen.scenario_definitions import CONSUMER_ARCHETYPES
from ui.utils import format_currency

st.set_page_config(page_title="Historical Analytics | BillGuard", page_icon="📊", layout="wide")

@st.cache_data
def load_dataset():
    data_path = Path("./data/synthetic/all_bills.csv")
    if not data_path.exists():
        gen = SyntheticBillGenerator(seed=42)
        gen.generate_full_dataset(num_consumers=40, cycles_per_consumer=12, output_dir="./data/synthetic")
    return pd.read_csv("./data/synthetic/all_bills.csv")

df = load_dataset()

st.title("📊 Historical Consumption Trends & Anomaly Analytics")
st.markdown("Inspect multi-cycle consumption patterns, seasonal baseline variance, and historical anomalies across domestic consumer archetypes.")

# Archetype Filter
col1, col2 = st.columns([1, 2])
with col1:
    arch_options = list(CONSUMER_ARCHETYPES.keys())
    selected_arch = st.selectbox(
        "Select Domestic Archetype",
        arch_options,
        format_func=lambda x: f"{CONSUMER_ARCHETYPES[x].name} ({x})"
    )
    arch_info = CONSUMER_ARCHETYPES[selected_arch]
    st.info(f"**Profile:** {arch_info.description}\n\n**Base Usage:** {arch_info.base_bimonthly_units_mean:.0f} ± {arch_info.base_bimonthly_units_std:.0f} kWh/cycle")

    arch_df = df[df["archetype"] == selected_arch]
    consumer_list = list(arch_df["consumer_id"].unique())
    selected_consumer = st.selectbox("Select Consumer ID", consumer_list)

consumer_records = df[df["consumer_id"] == selected_consumer].sort_values("cycle_idx").reset_index(drop=True)

with col2:
    st.markdown(f"#### Consumption History for `{selected_consumer}`")
    chart_data = pd.DataFrame({
        "Cycle": [f"C{r['cycle_idx']+1} ({r['current_reading_date'][:7]})" for _, r in consumer_records.iterrows()],
        "Recorded Units (kWh)": consumer_records["recorded_units"],
        "Calculated Total (₹)": consumer_records["calculated_grand_total"],
        "Billed Total (₹)": consumer_records["billed_amount"],
    }).set_index("Cycle")
    st.line_chart(chart_data[["Recorded Units (kWh)"]])

# Timeline Cards
st.markdown("### 🗓️ Longitudinal Cycle Audit Breakdown")

display_df = consumer_records[[
    "cycle_idx", "current_reading_date", "billing_days", "recorded_units",
    "calculated_grand_total", "billed_amount", "discrepancy_amount", "anomaly_type", "is_anomaly"
]].copy()

display_df["Status"] = display_df.apply(
    lambda r: f"🚨 {r['anomaly_type']}" if r["is_anomaly"] == 1 else "✅ Normal", axis=1
)
display_df["Discrepancy (₹)"] = display_df["discrepancy_amount"].apply(lambda v: f"₹{v:+.2f}")
display_df["Calculated Total"] = display_df["calculated_grand_total"].apply(format_currency)
display_df["Billed Amount"] = display_df["billed_amount"].apply(format_currency)

st.dataframe(
    display_df[[
        "cycle_idx", "current_reading_date", "billing_days", "recorded_units",
        "Calculated Total", "Billed Amount", "Discrepancy (₹)", "Status"
    ]].rename(columns={"cycle_idx": "Cycle #", "current_reading_date": "Bill Date", "recorded_units": "Units (kWh)"}),
    hide_index=True,
    use_container_width=True
)
