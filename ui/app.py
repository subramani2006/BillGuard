import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for cloud deployment
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from PIL import Image
from ui.utils import inject_custom_theme

st.set_page_config(
    page_title="BillGuard | Electricity Bill Verification & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified high-tech cyber-slate theme
inject_custom_theme()

# Grand Hero Banner
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
        <span class="card-chip chip-cyan">
            <span class="status-dot"></span> TNERC LT-1A AUDIT ENGINE ACTIVE
        </span>
        <span class="card-chip chip-emerald">
            🛡️ ZERO PRETRAINED WEIGHTS GUARANTEE
        </span>
    </div>
    <div class="hero-title">⚡ BillGuard Enterprise</div>
    <p class="hero-sub">
        An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System designed for 
        Indian domestic electricity consumers. Combines deterministic tariff mathematics, 
        randomly-initialized deep learning OCR, unsupervised anomaly detection, and bilingual legal reasoning in English & தமிழ்.
    </p>
</div>
""", unsafe_allow_html=True)

# Overview Metric Grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Statutory Tariff</div>
        <div class="card-value">Tamil Nadu</div>
        <span class="card-chip chip-cyan">🏛️ TNERC LT-1A 2024</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Model Weights</div>
        <div class="card-value">From Scratch</div>
        <span class="card-chip chip-emerald">🧠 Zero Pretrained Weights</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Explainability</div>
        <div class="card-value">English + தமிழ்</div>
        <span class="card-chip chip-amber">⚖️ 3 Regulatory Outcomes</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">MLOps Platform</div>
        <div class="card-value">Active Gate</div>
        <span class="card-chip chip-rose">🔄 Drift & 1-Click Rollback</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# System Navigation Cards
st.markdown("### 🎛️ System Capabilities & Verification Modules")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="font-size: 1.8rem; background: rgba(56, 189, 248, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">📄</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.2rem;">1. Bill Verifier & OCR Upload</h4>
                <span style="font-size: 0.8rem; color: #94A3B8;">8-Step Verification Workflow</span>
            </div>
        </div>
        <ul style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.7; margin-bottom: 0;">
            <li><b>Dual Input Modes:</b> Manual bill parameter entry or physical bill card image upload.</li>
            <li><b>From-Scratch CRNN OCR:</b> Character-level confidence estimation with interactive crop review.</li>
            <li><b>Deterministic Audit:</b> Slabs, GoTN 100 free units subsidy, FPPCA fuel surcharge, and arrears.</li>
            <li><b>Evidence-Based Report:</b> 10-point investigation breakdown into 3 standard regulatory outcomes.</li>
            <li><b>Legal Petition Export:</b> One-click generation of TANGEDCO grievance letters in plain text or PDF.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="font-size: 1.8rem; background: rgba(245, 158, 11, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.3);">⚡</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.2rem;">2. Tariff Explorer & Simulator</h4>
                <span style="font-size: 0.8rem; color: #94A3B8;">Versioned Tariff Intelligence</span>
            </div>
        </div>
        <ul style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.7; margin-bottom: 0;">
            <li><b>Official Statutory Tariffs:</b> TNERC LT-1A orders (2022, 2023, 2024 revisions) with source citations.</li>
            <li><b>Simulated Research Tariffs:</b> Clearly labeled with simulation disclaimers.</li>
            <li><b>Interactive Billing Delay Simulator:</b> Visualizes the financial impact of delayed 75+ day cycles.</li>
            <li><b>Dynamic Cost Curve:</b> Real-time telescopic slab progression chart.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="font-size: 1.8rem; background: rgba(16, 185, 129, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">📊</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.2rem;">3. Historical Consumption Analytics</h4>
                <span style="font-size: 0.8rem; color: #94A3B8;">Longitudinal Data Integrity Audit</span>
            </div>
        </div>
        <ul style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.7; margin-bottom: 0;">
            <li><b>Automated Data Integrity Audit:</b> Enforces <code>discrepancy = billed - calculated</code> on every cycle.</li>
            <li><b>Archetype Behavioral Profiles:</b> Lifeline, Moderate, High AC, and Luxury Villa.</li>
            <li><b>Seasonal Decomposition:</b> May-June peak summer (Agni Nakshatram) surge vs monsoon moderation.</li>
            <li><b>Dual Interactive Charts:</b> Consumption trends and billed vs calculated divergences.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="font-size: 1.8rem; background: rgba(99, 102, 241, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3);">🛠️</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.2rem;">4. MLOps Monitoring Center</h4>
                <span style="font-size: 0.8rem; color: #94A3B8;">Governance, Safety Gate & Drift</span>
            </div>
        </div>
        <ul style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.7; margin-bottom: 0;">
            <li><b>Real-Time Drift Monitoring:</b> Population Stability Index (PSI) and Kolmogorov-Smirnov test.</li>
            <li><b>Automated Promotion Gate:</b> Rejects candidates failing F1 ≥ 0.75 or Precision ≥ 0.70.</li>
            <li><b>Safe 1-Click Rollback:</b> Immediate production restoration to previous checkpoint.</li>
            <li><b>Immutable Audit Trail:</b> SQLite and append-only JSONL prediction records.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Architectural Integrity Statement Card
st.markdown("""
<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-left: 4px solid #38BDF8; border-radius: 12px; padding: 20px 24px; margin-top: 10px;">
    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #38BDF8; font-size: 1rem; margin-bottom: 8px;">
        🛡️ Architectural Governance & Academic Principles
    </div>
    <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
        <b>1. Zero Pretrained Weights:</b> All neural architectures (CRNN OCR, PyTorch Autoencoder) and statistical models (Isolation Forest) are initialized from random weights and trained strictly on locally generated synthetic datasets.<br/>
        <b>2. Deterministic vs. Heuristic Separation:</b> Arithmetic calculations and statutory slab tariffs are audited with mathematical certainty, while behavioral anomalies are graded with explicit uncertainty levels.<br/>
        <b>3. Presumption of Innocence:</b> An uncharacteristic spike alone is never termed fraud or meter fault without official laboratory testing pursuant to Rule 57 of the CEA Regulations.
    </div>
</div>
""", unsafe_allow_html=True)
