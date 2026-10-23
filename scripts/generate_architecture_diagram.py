"""Generate publication-grade architecture diagram for BillGuard Final-Year Project Report."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path("./project_report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "architecture_diagram.png"

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(15, 10.5), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color tokens
    c_blue = '#2563EB'
    c_cyan = '#38BDF8'
    c_emerald = '#10B981'
    c_amber = '#F59E0B'
    c_purple = '#8B5CF6'
    c_rose = '#EF4444'
    c_slate = '#1E293B'
    c_dark = '#111827'
    c_text = '#F8FAFC'
    c_sub = '#94A3B8'

    # Title
    ax.text(50, 96.5, "BillGuard: System Architecture & Data Flow", 
            fontsize=20, fontweight='bold', color=c_text, ha='center', va='center')
    ax.text(50, 93.8, "Explainable, MLOps-Driven Electricity Bill Verification & Anomaly Detection Pipeline", 
            fontsize=11, color=c_cyan, ha='center', va='center')

    # Box drawing helper
    def draw_box(x, y, w, h, title, subtitle="", items=None, border_color=c_cyan, bg_color=c_dark, header_color=c_cyan):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.2",
                                      linewidth=1.8, edgecolor=border_color, facecolor=bg_color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.0, title, fontsize=10.5, fontweight='bold', color=header_color, ha='center', va='top')
        if subtitle:
            ax.text(x + w/2, y + h - 4.2, subtitle, fontsize=8.2, color=c_sub, ha='center', va='top', style='italic')
        if items:
            start_y = y + h - (5.8 if subtitle else 4.2)
            for i, itm in enumerate(items):
                ax.text(x + 1.2, start_y - (i * 2.3), f"• {itm}", fontsize=8.0, color=c_text, ha='left', va='top')

    # Arrow drawing helper
    def draw_arrow(x1, y1, x2, y2, color=c_cyan, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=14))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.8, label, fontsize=7.5, color=color, ha='center', va='bottom', fontweight='semibold')

    # ROW 1: USER INTERFACE & INGESTION (Y: 76 - 90)
    draw_box(4, 76, 28, 14, "1. Presentation Layer (UI)", "Dual-Interface Access",
             ["Streamlit Enterprise Dashboard", "FastAPI RESTful Endpoints", "Preset Demo Scenarios (A to E)", "Responsive Cyber-Slate UI"],
             border_color=c_cyan, header_color=c_cyan)

    draw_box(36, 76, 28, 14, "2. Dual Input & Ingestion", "Raw Bill Data Ingestion",
             ["Manual Parameter Review Form", "Physical Bill Card Image Upload", "Synthetic TANGEDCO Generator", "Edge Case & Delay Injection"],
             border_color=c_blue, header_color=c_cyan)

    draw_box(68, 76, 28, 14, "3. From-Scratch CRNN OCR", "Zero Pretrained Weights",
             ["Convolutional Feature Extractor", "Bidirectional LSTM Sequence Model", "CTC Beam Search Decoder", "Confidence Check & Human Review (<0.7)"],
             border_color=c_purple, header_color=c_purple)

    # ROW 2: CORE VERIFICATION & ANOMALY DETECTION (Y: 48 - 66)
    draw_box(4, 48, 28, 21, "4. Deterministic Tariff Engine", "TNERC Gazette Ground Truth",
             ["Date & Chronology Validation", "Meter Rollover & Mid-Cycle Replace", "LT-1A Progressive Telescopic Slabs", "GoTN 100-Unit Lifeline Subsidy", "FPPCA Fuel Surcharge Computation", "Past Arrears & Regulatory Cess", "Strict Discrepancy = Billed - Calc"],
             border_color=c_emerald, header_color=c_emerald)

    draw_box(36, 48, 28, 21, "5. ML Anomaly Screening", "Unsupervised Behavior Baselines",
             ["10-Dim Longitudinal Feature Vector", "From-Scratch Isolation Forest", "From-Scratch Deep Autoencoder", "Tree-Path & Reconstruction Attribs", "Summer (Agni Nakshatram) Multipliers", "Certainty Level Grading (High/Mod/Low)", "Distinguishes Math Error vs Behavioral"],
             border_color=c_amber, header_color=c_amber)

    draw_box(68, 48, 28, 21, "6. Evidence Investigation", "3 Regulatory Outcomes",
             ["Outcome A: Confirmed Arithmetic Error", "Outcome B: Unusual Spike (Need Test)", "Outcome C: Verified Statutory Normal", "10-Point Investigation Report", "Bilingual Explanations (EN & TA)", "Statutory Petition Draft (Clause 11)", "Court-Ready PDF Grievance Export"],
             border_color=c_rose, header_color=c_rose)

    # ROW 3: MLOPS GOVERNANCE & LIFECYCLE (Y: 18 - 40)
    draw_box(4, 18, 28, 22, "7. Model Registry & Gates", "Lifecycle & Zero-Regression",
             ["4-Tier Staging: Dev, Cand, Prod, Arch", "Automated Safety Gatekeeper:", "  - Minimum F1 Threshold (>= 0.75)", "  - Precision Threshold (>= 0.70)", "  - Non-Regression vs Active Prod", "1-Click Certified Promotion", "1-Click Rollback to Previous Checkpoint"],
             border_color=c_cyan, header_color=c_cyan)

    draw_box(36, 18, 28, 22, "8. Continuous Drift Monitor", "Statistical Distribution Shift",
             ["Population Stability Index (PSI)", "Two-Sample Kolmogorov-Smirnov (KS)", "Threshold Zones: Normal, Warning, Crit", "Real-Time Feature Distribution Shift", "Data Quality & Bounds Score", "Defensive Baseline Dataset Generator", "Automated Alerting Trigger"],
             border_color=c_purple, header_color=c_purple)

    draw_box(68, 18, 28, 22, "9. Retraining & Audit Trail", "Zero-Pretrained Weight Governance",
             ["Automated Retraining Console", "Fresh Synthetic Cohort Sampling", "Model Training from Scratch", "Multi-Model Benchmark Evaluation", "Candidate Registration in Registry", "SQLite Immutable Prediction Log", "Append-Only JSONL Audit Stream"],
             border_color=c_emerald, header_color=c_emerald)

    # STORAGE & PERSISTENCE BAR (Y: 5 - 12)
    draw_box(4, 5, 92, 7.5, "10. Storage & Artifact Layer: YAML Tariffs (config/) | Model Checkpoints (.pth / .joblib) | SQLite Audit DB | CSV/JSON Datasets",
             "", None, border_color=c_slate, bg_color='#0F172A', header_color=c_sub)

    # CONNECTING ARROWS & DATA FLOW
    draw_arrow(18, 76, 18, 69, color=c_cyan, label="")
    draw_arrow(32, 83, 36, 83, color=c_blue, label="Params")
    draw_arrow(64, 83, 68, 83, color=c_purple, label="Bill Crop")
    draw_arrow(68, 80, 64, 80, color=c_emerald, label="Fields")
    draw_arrow(50, 76, 50, 69, color=c_amber, label="Features")
    draw_arrow(32, 58, 36, 58, color=c_emerald, label="Units/Totals")
    draw_arrow(64, 58, 68, 58, color=c_rose, label="Scores/Attribs")
    draw_arrow(82, 48, 82, 40, color=c_rose, label="Audit Log")
    draw_arrow(18, 48, 18, 40, color=c_cyan, label="Production Model")
    draw_arrow(50, 48, 50, 40, color=c_purple, label="Live Inferences")
    draw_arrow(64, 29, 68, 29, color=c_emerald, label="Drift Trigger")
    draw_arrow(36, 29, 32, 29, color=c_cyan, label="Metrics")
    draw_arrow(50, 18, 50, 12.5, color=c_sub, label="Persist")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved successfully to: {OUTPUT_IMG}")

if __name__ == "__main__":
    create_architecture_diagram()
