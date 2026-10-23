"""Utility helpers and API clients for Streamlit Dashboard."""

import io
import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

API_BASE_URL = os.getenv("BILLGUARD_API_URL", "http://localhost:8000/api")


def format_currency(val: Optional[float]) -> str:
    if val is None:
        return "₹0.00"
    return f"₹{val:,.2f}"


def get_verdict_badge(category: str) -> str:
    """Return colored HTML badge corresponding to anomaly category."""
    color_map = {
        "CONFIRMED_ARITHMETIC_ERROR": ("#DC2626", "Arithmetic Overcharge / Error"),
        "METER_READING_MISMATCH": ("#EA580C", "Meter Reading Inconsistency"),
        "SUSPECTED_EXTENDED_CYCLE": ("#D97706", "Extended Billing Cycle"),
        "UNUSUAL_CONSUMPTION_SURGE": ("#CA8A04", "Unusual High Consumption"),
        "SUSPECTED_METER_FAULT": ("#E11D48", "Suspected Meter Anomaly"),
        "DEAD_METER_ZERO": ("#4F46E5", "Zero Consumption Warning"),
        "NORMAL_VERIFIED": ("#16A34A", "Verified Normal"),
    }
    bg, label = color_map.get(category, ("#6B7280", category.replace("_", " ")))
    return f'<span style="background-color: {bg}; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 13px;">{label}</span>'


def get_certainty_badge(level: str) -> str:
    color_map = {
        "HIGH_CERTAINTY": ("#059669", "High Certainty (Mathematical Proof)"),
        "MODERATE_EVIDENCE": ("#D97706", "Moderate Evidence (Cycle Shift)"),
        "LOW_NEED_PHYSICAL_INSPECTION": ("#DC2626", "Inspection Required (Behavioral Spike)"),
    }
    bg, label = color_map.get(level, ("#6B7280", level))
    return f'<span style="background-color: {bg}; color: white; padding: 3px 10px; border-radius: 6px; font-size: 12px;">{label}</span>'


def get_outcome_badge(outcome: str) -> str:
    """Return colored badge for the 3 standard investigation outcomes."""
    if "A." in outcome or "arithmetic discrepancy" in outcome.lower():
        bg = "#DC2626"
        icon = "🚨"
    elif "B." in outcome or "investigation" in outcome.lower():
        bg = "#D97706"
        icon = "🔍"
    else:
        bg = "#16A34A"
        icon = "✅"
    return f'<span style="background-color: {bg}; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 14px;">{icon} {outcome}</span>'


def get_check_status_badge(status: str) -> str:
    """Return badge for audit check statuses."""
    s = status.upper()
    if s == "PASSED":
        return '<span style="background-color: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">✅ PASSED</span>'
    elif s == "FAILED":
        return '<span style="background-color: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">❌ FAILED</span>'
    elif s == "WARNING":
        return '<span style="background-color: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">⚠️ WARNING</span>'
    else:
        return '<span style="background-color: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">⚪ NOT PERFORMED</span>'

