import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for cloud deployment
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from ui.utils import inject_custom_theme, render_hero_banner

st.set_page_config(
    page_title="BillGuard | Electricity Bill Verification & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_theme()

# Grand Hero Banner
render_hero_banner(
    title="⚡ BillGuard",
    subtitle="An Explainable, MLOps-Driven Electricity Bill Verification & Anomaly Detection System for Indian Domestic Consumers.",
    tag_left="<span class='status-dot'></span> AUDIT ENGINE ONLINE",
    tag_right="🛡️ STRICT ZERO PRETRAINED WEIGHTS",
)

# Quick Demonstration Guide Card
st.markdown("""
<div class="glass-card" style="margin-bottom: 24px; padding: 18px 24px; border-left: 4px solid #38BDF8;">
    <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
        🚀 College Demonstration Guide — 3 Steps to Audit:
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 10px;">
        <div>
            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.92rem;">1️⃣ Select Preloaded Scenario</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 3px;">
                Choose <b>Scenario A (Normal)</b>, <b>Scenario B (Unusual)</b>, or <b>Scenario C (Invalid Data)</b> in the Bill Verifier.
            </div>
        </div>
        <div>
            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.92rem;">2️⃣ Run Instant Audit</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 3px;">
                BillGuard audits every slab, GoTN 100 free units subsidy, and runs AI anomaly screening with zero delay.
            </div>
        </div>
        <div>
            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.92rem;">3️⃣ Review & Download Petition</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 3px;">
                View bilingual explanations (English & தமிழ்) and download an official complaint-ready PDF petition.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3 Primary Dashboard Sections
st.markdown("### 🎛️ Primary System Capabilities")

col_sec1, col_sec2, col_sec3 = st.columns(3)

with col_sec1:
    st.markdown("""
    <div class="glass-card" style="padding: 22px; height: 100%; border: 1px solid rgba(56, 189, 248, 0.35);">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
            <div style="font-size: 1.8rem; background: rgba(56, 189, 248, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">📄</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.15rem;">1. Bill Verification</h4>
                <span style="font-size: 0.8rem; color: #38BDF8; font-weight: 600;">MAIN DEMO FEATURE</span>
            </div>
        </div>
        <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;">
            The primary engine for consumer bill audit and anomaly detection:
        </p>
        <ul style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
            <li><b>3 Preloaded Scenarios:</b> Normal Bill, Unusual Bill, and Invalid Bill Data.</li>
            <li><b>From-Scratch CRNN OCR:</b> Character recognition trained strictly on synthetic bills.</li>
            <li><b>Deterministic Audit:</b> Slabs, GoTN 100 free units subsidy, and FPPCA.</li>
            <li><b>Explainable AI:</b> Plain-English & தமிழ் findings with 1-click PDF petition.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_sec2:
    st.markdown("""
    <div class="glass-card" style="padding: 22px; height: 100%; border: 1px solid rgba(245, 158, 11, 0.35);">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
            <div style="font-size: 1.8rem; background: rgba(245, 158, 11, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.3);">⚡</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.15rem;">2. Tariff Explorer</h4>
                <span style="font-size: 0.8rem; color: #FBBF24; font-weight: 600;">STATUTORY TARIFF INTELLIGENCE</span>
            </div>
        </div>
        <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;">
            Statutory tariff schedules and delay penalty simulations:
        </p>
        <ul style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
            <li><b>TNERC LT-1A Schedules:</b> Gazette-verified Tamil Nadu domestic tariff orders.</li>
            <li><b>GoTN 100-Unit Subsidy:</b> Full explanation of telescopic domestic quotas.</li>
            <li><b>Billing Delay Simulator:</b> Visualizes the financial impact of delayed 75+ day cycles.</li>
            <li><b>Dynamic Cost Curve:</b> Interactive progressive tariff trajectory up to 1,000 kWh.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_sec3:
    st.markdown("""
    <div class="glass-card" style="padding: 22px; height: 100%; border: 1px solid rgba(16, 185, 129, 0.35);">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
            <div style="font-size: 1.8rem; background: rgba(16, 185, 129, 0.15); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">🛠️</div>
            <div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.15rem;">3. ML Monitoring</h4>
                <span style="font-size: 0.8rem; color: #34D399; font-weight: 600;">MLOPS & AUDIT TRAILS</span>
            </div>
        </div>
        <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;">
            Model governance, safety gates, and data drift observability:
        </p>
        <ul style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
            <li><b>Model Registry:</b> Active production model status, candidate staging, and 1-click rollback.</li>
            <li><b>Data Drift (PSI):</b> Continuous monitoring of seasonal consumption variances.</li>
            <li><b>Automated Safety Gate:</b> Enforces zero-regression before candidate promotion.</li>
            <li><b>Immutable Audit Trail:</b> SQLite-backed log of every verified bill.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Live Status Strip
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Statutory Tariff</div>
        <div class="card-value" style="font-size: 1.3rem;">Tamil Nadu</div>
        <span class="card-chip chip-cyan">🏛️ TNERC LT-1A 2024</span>
    </div>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Active Model</div>
        <div class="card-value" style="font-size: 1.3rem;">Isolation Forest</div>
        <span class="card-chip chip-emerald">🧠 Zero Pretrained Weights</span>
    </div>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Explainability</div>
        <div class="card-value" style="font-size: 1.3rem;">English + தமிழ்</div>
        <span class="card-chip chip-amber">⚖️ 3 Regulatory Outcomes</span>
    </div>
    """, unsafe_allow_html=True)

with col_k4:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">Safety Gate</div>
        <div class="card-value" style="font-size: 1.3rem;">Zero Regression</div>
        <span class="card-chip chip-rose">🔄 1-Click Rollback Enabled</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Academic Integrity Principles
st.markdown("""
<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-left: 4px solid #38BDF8; border-radius: 12px; padding: 18px 22px;">
    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #38BDF8; font-size: 0.95rem; margin-bottom: 6px;">
        🛡️ Academic Principles & Critical Distinctions
    </div>
    <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.6;">
        • <b>Normal vs. Anomaly vs. Invalid:</b> Input validation halts corrupted records before they reach the ML model. An anomaly is a flag for consumer investigation, NOT proof of fraud.<br/>
        • <b>Deterministic Audit vs. ML Screening:</b> Tariff calculations are verified with 100% mathematical certainty against TNERC regulations, while behavioral spikes are graded with explicit confidence levels.<br/>
        • <b>Strict Zero Pretrained Weights:</b> All models (CRNN OCR, Isolation Forest, PyTorch Autoencoder) are initialized from random weights and trained solely on synthetic Tamil Nadu domestic consumer billing histories.
    </div>
</div>
""", unsafe_allow_html=True)
