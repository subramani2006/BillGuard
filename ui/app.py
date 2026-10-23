"""BillGuard: Main Streamlit Application."""

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="BillGuard | Electricity Bill Verification & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0A2540;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-info {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ BillGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System for Indian Domestic Consumers</div>', unsafe_allow_html=True)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Jurisdiction & Tariff</div>
        <div class="metric-value">Tamil Nadu</div>
        <span class="badge-info">TNERC LT-1A 2024</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ML Models Initialized</div>
        <div class="metric-value">From Scratch</div>
        <span class="badge-info">Zero Pretrained Weights</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Bilingual Reasoning</div>
        <div class="metric-value">English + தமிழ்</div>
        <span class="badge-info">Legal Evidence Audit</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MLOps Lifecycle</div>
        <div class="metric-value">Active Gate</div>
        <span class="badge-info">Drift, Registry & Rollback</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Navigation & Features
st.markdown("### 🔍 System Capabilities & Navigation")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    #### 📄 [1. Bill Verifier & OCR Upload](Bill_Verifier)
    - **Dual Input Modes**: Manual entry or upload physical bill card images.
    - **From-Scratch CRNN OCR**: Field recognition with per-field confidence scoring.
    - **Human-in-the-Loop Fallback**: Low-confidence crop preview with inline verification.
    - **Deterministic Mathematical Audit**: Slabs, subsidies (GoTN 100 free units), FPPCA, and arrears.
    - **Evidence-Based Bilingual Explanations**: In English and Tamil (தமிழ்).
    - **One-Click Formal Grievance Letter**: Export complaint petition in text or PDF.
    """)

    st.markdown("""
    #### ⚡ [2. Tariff Explorer & Simulator](Tariff_Explorer)
    - Versioned TANGEDCO LT-1A Tariff Orders (2022, 2023, 2024, and experimental progressive).
    - Interactive slab price calculator and visual curve.
    - Billing period extension & proration impact simulator.
    """)

with c2:
    st.markdown("""
    #### 📊 [3. Historical Consumption Analytics](Historical_Analytics)
    - Multi-cycle longitudinal trends across domestic archetypes.
    - Seasonal decomposition (South Indian summer peak vs monsoon low).
    - Anomaly pattern visualizer (surge, dead meter, arithmetic error).
    """)

    st.markdown("""
    #### 🛠️ [4. MLOps Monitoring Center](MLOps_Center)
    - **Data Drift Detection**: Population Stability Index (PSI) & Kolmogorov-Smirnov test.
    - **Model Registry & Staging**: Candidate vs Production evaluation gates.
    - **Safe Promotion & 1-Click Rollback**: Zero-downtime model lifecycle control.
    - **Automated Retraining**: Train new candidate models strictly from scratch.
    """)

st.markdown("---")

# Architectural Integrity Statement
st.info("""
**Auditing Integrity & Academic Compliance Statement**:
- **Zero Pretrained Weights**: Every neural and machine learning model (CRNN OCR, Isolation Forest, PyTorch Autoencoder) is trained strictly from random initialization on locally generated synthetic data. No pretrained APIs, checkpoints, or external weights are used.
- **Evidence-Backed Verdicts**: A high bill amount alone is never attributed to fraud or meter failure. Only verified mathematical discrepancies, meter physical difference mismatches, or statistical behavioral shifts with explicit uncertainty levels are reported.
- **Official Tariff Alignment**: Slabs, free subsidies, and billing rules reflect verified Tamil Nadu Electricity Regulatory Commission (TNERC) tariff orders.
""")
