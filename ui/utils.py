"""Utility helpers, styling engines, and theme providers for BillGuard Streamlit Dashboard."""

import io
import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = os.getenv("BILLGUARD_API_URL", "http://localhost:8000/api")


def format_currency(val: Optional[float]) -> str:
    """Format numeric amount into Indian Rupee currency string."""
    if val is None:
        return "₹0.00"
    return f"₹{val:,.2f}"


def inject_custom_theme():
    """
    Injects enterprise-grade, high-contrast dark cyber-slate design system
    with matching electric sapphire accents, glassmorphic elevations, and luminous state indicators.
    """
    st.markdown("""
    <style>
        /* Import Google Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F8FAFC;
        }

        /* Ambient app background smoothing */
        .stApp {
            background-color: #0B0F19;
            background-image: 
                radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.08) 0px, transparent 50%);
        }

        /* Sleek modern scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0B0F19;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E293B;
            border-radius: 4px;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #38BDF8;
        }

        /* Glassmorphic Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.70) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.6), 0 0 24px -6px rgba(56, 189, 248, 0.15);
            backdrop-filter: blur(16px);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #38BDF8, #6366F1, #10B981);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #FFFFFF 20%, #E2E8F0 60%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .hero-sub {
            font-size: 1.05rem;
            color: #94A3B8;
            line-height: 1.6;
            margin-bottom: 0;
            max-width: 950px;
        }

        /* Pulsing Status Dot */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 10px #10B981;
            margin-right: 6px;
            animation: pulse-dot 2s infinite ease-in-out;
        }
        @keyframes pulse-dot {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Glassmorphic Metric Cards */
        .glass-card {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(31, 41, 55, 0.60) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(12px);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            height: 100%;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 12px 32px -4px rgba(0, 0, 0, 0.6), 0 0 16px rgba(56, 189, 248, 0.15);
        }
        .card-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }
        .card-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }
        .card-chip {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 6px;
        }
        .chip-cyan { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .chip-emerald { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .chip-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .chip-rose { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }

        /* Grand Outcome Callouts */
        .outcome-box-a {
            background: linear-gradient(135deg, rgba(69, 10, 10, 0.5) 0%, rgba(127, 29, 29, 0.3) 100%);
            border: 1.5px solid #EF4444;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 0 24px rgba(239, 68, 68, 0.18);
        }
        .outcome-box-b {
            background: linear-gradient(135deg, rgba(69, 26, 3, 0.5) 0%, rgba(120, 53, 15, 0.3) 100%);
            border: 1.5px solid #F59E0B;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 0 24px rgba(245, 158, 11, 0.18);
        }
        .outcome-box-c {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(6, 95, 70, 0.3) 100%);
            border: 1.5px solid #10B981;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 0 24px rgba(16, 185, 129, 0.18);
        }

        /* Custom Streamlit Metric Styling */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(31, 41, 55, 0.60) 100%) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
            transition: border-color 0.2s ease, transform 0.2s ease !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: rgba(56, 189, 248, 0.35) !important;
            transform: translateY(-2px);
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-weight: 800 !important;
            font-size: 1.65rem !important;
            letter-spacing: -0.02em !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        /* Sidebar Sleek Styling */
        [data-testid="stSidebar"] {
            background-color: #0D1322 !important;
            border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #E2E8F0;
        }

        /* Expanders Customization */
        .streamlit-expanderHeader {
            background-color: #111827 !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
        .streamlit-expanderHeader:hover {
            border-color: #38BDF8 !important;
        }

        /* Streamlit Input field sleek overrides */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
            font-size: 0.95rem !important;
        }
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
        }

        /* Selectboxes Customization */
        [data-baseweb="select"] > div {
            background-color: #111827 !important;
            border-color: #334155 !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }

        /* Buttons Grand Elevation */
        button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #0284C7 100%) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 24px rgba(56, 189, 248, 0.45) !important;
        }
        button[kind="secondary"] {
            background: #1E293B !important;
            color: #E2E8F0 !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        button[kind="secondary"]:hover {
            border-color: #38BDF8 !important;
            color: white !important;
        }

        /* Streamlit Tabs Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #111827;
            padding: 6px;
            border-radius: 12px;
            border: 1px solid #1E293B;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 18px;
            color: #94A3B8;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1E293B 0%, #334155 100%) !important;
            color: #38BDF8 !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #1E293B;
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, tag_left: str = "", tag_right: str = ""):
    """Renders a consistent, grand cyber-slate hero banner at the top of a dashboard page."""
    chips_html = ""
    if tag_left or tag_right:
        left_html = f'<span class="card-chip chip-cyan">{tag_left}</span>' if tag_left else ""
        right_html = f'<span class="card-chip chip-emerald">{tag_right}</span>' if tag_right else ""
        chips_html = f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">{left_html}{right_html}</div>'

    st.markdown(f"""
    <div class="hero-banner">
        {chips_html}
        <div class="hero-title">{title}</div>
        <p class="hero-sub">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def get_plotly_dark_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply matching enterprise dark aesthetics to Plotly charts."""
    fig.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#111827",
        font=dict(family="Inter, sans-serif", color="#E2E8F0", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.10)",
            linecolor="rgba(148, 163, 184, 0.20)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.10)",
            linecolor="rgba(148, 163, 184, 0.20)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


def get_verdict_badge(category: str) -> str:
    """Return glowing HTML badge corresponding to anomaly category."""
    color_map = {
        "CONFIRMED_ARITHMETIC_ERROR": ("#EF4444", "rgba(239, 68, 68, 0.15)", "Arithmetic Overcharge / Error"),
        "METER_READING_MISMATCH": ("#F97316", "rgba(249, 115, 22, 0.15)", "Meter Reading Inconsistency"),
        "SUSPECTED_EXTENDED_CYCLE": ("#F59E0B", "rgba(245, 158, 11, 0.15)", "Extended Billing Cycle"),
        "UNUSUAL_CONSUMPTION_SURGE": ("#EAB308", "rgba(234, 179, 8, 0.15)", "Unusual High Consumption"),
        "SUSPECTED_METER_FAULT": ("#EC4899", "rgba(236, 72, 153, 0.15)", "Suspected Meter Anomaly"),
        "DEAD_METER_ZERO": ("#818CF8", "rgba(129, 140, 248, 0.15)", "Zero Consumption Warning"),
        "NORMAL_VERIFIED": ("#10B981", "rgba(16, 185, 129, 0.15)", "Verified Statutory Normal"),
    }
    fg, bg, label = color_map.get(category, ("#94A3B8", "rgba(148, 163, 184, 0.15)", category.replace("_", " ")))
    return f'<span style="background-color: {bg}; color: {fg}; border: 1px solid {fg}; padding: 4px 14px; border-radius: 9999px; font-weight: 700; font-size: 13px; letter-spacing: 0.02em;">{label}</span>'


def get_certainty_badge(level: str) -> str:
    """Return badge representing legal/mathematical evidence grading."""
    color_map = {
        "HIGH_CERTAINTY": ("#10B981", "rgba(16, 185, 129, 0.15)", "High Certainty (Mathematical Proof)"),
        "MODERATE_EVIDENCE": ("#F59E0B", "rgba(245, 158, 11, 0.15)", "Moderate Evidence (Cycle Shift)"),
        "LOW_NEED_PHYSICAL_INSPECTION": ("#EF4444", "rgba(239, 68, 68, 0.15)", "Inspection Required (Behavioral Spike)"),
    }
    fg, bg, label = color_map.get(level, ("#94A3B8", "rgba(148, 163, 184, 0.15)", level))
    return f'<span style="background-color: {bg}; color: {fg}; border: 1px solid {fg}; padding: 4px 12px; border-radius: 8px; font-size: 12px; font-weight: 600;">{label}</span>'


def get_outcome_badge(outcome: str) -> str:
    """Return grand colored badge for the 3 standard investigation outcomes."""
    if "A." in outcome or "arithmetic discrepancy" in outcome.lower():
        bg = "rgba(239, 68, 68, 0.15)"
        border = "#EF4444"
        fg = "#F87171"
        icon = "🚨"
    elif "B." in outcome or "investigation" in outcome.lower():
        bg = "rgba(245, 158, 11, 0.15)"
        border = "#F59E0B"
        fg = "#FBBF24"
        icon = "🔍"
    else:
        bg = "rgba(16, 185, 129, 0.15)"
        border = "#10B981"
        fg = "#34D399"
        icon = "🛡️"
    return (
        f'<span style="background-color: {bg}; border: 1.5px solid {border}; color: {fg}; '
        f'padding: 7px 18px; border-radius: 9999px; font-weight: 800; font-size: 14.5px; '
        f'box-shadow: 0 0 16px {bg};">{icon} {outcome}</span>'
    )


def get_check_status_badge(status: str) -> str:
    """Return glowing badge for audit check statuses."""
    s = status.upper()
    if s == "PASSED":
        return '<span style="background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid #10B981; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">✅ PASSED</span>'
    elif s == "FAILED":
        return '<span style="background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">❌ FAILED</span>'
    elif s == "WARNING":
        return '<span style="background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid #F59E0B; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">⚠️ WARNING</span>'
    else:
        return '<span style="background-color: rgba(148, 163, 184, 0.12); color: #94A3B8; border: 1px solid #475569; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 11.5px;">⚪ NOT PERFORMED</span>'
