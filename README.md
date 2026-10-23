# ⚡ BillGuard: An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Scratch--Trained-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/Tests-34%20Passed-brightgreen.svg)]()

> **Final-Year Engineering Capstone Project**  
> An explainable, full-stack, production-grade electricity bill verification and anomaly detection system designed for Indian domestic consumers. Initially supporting Tamil Nadu (TANGEDCO / TNERC LT-1A tariff) through a versioned tariff engine, from-scratch deep learning OCR, unsupervised anomaly detection, bilingual reasoning in English and Tamil (தமிழ்), and an end-to-end MLOps lifecycle.

---

## 🌟 Core System Highlights

1. **Strict Zero Pretrained Weights Guarantee**:
   - Every neural network and machine learning model is trained strictly from random initialization on locally generated synthetic data.
   - **Zero** pretrained models, **zero** pretrained weights, **zero** pretrained OCR checkpoints, and **zero** external AI cloud APIs.
2. **Deterministic vs. Heuristic Separation**:
   - Exact statutory tariff slabs, GoTN 100-unit free subsidies, FPPCA fuel surcharges, and meter reading arithmetic are audited deterministically down to the exact paise.
   - Behavioral, seasonal, and statistical shifts are flagged using trained ML anomaly models with explicit uncertainty bounds.
   - High bills are **never** falsely attributed to fraud or meter failure without physical inspection evidence.
3. **From-Scratch CRNN OCR with Human-in-the-Loop Fallback**:
   - Custom Convolutional Recurrent Neural Network (CNN + Bidirectional GRU + CTC Loss) trained on synthetic bill card text crops.
   - Character-level confidence estimation. Fields with confidence `< 0.70` trigger an interactive crop-preview fallback for consumer review before calculation.
4. **Bilingual Explainability & Grievance Generation**:
   - Explanations generated in both **English** and **Tamil (தமிழ்)**.
   - Generates formal, legally compliant grievance petition letters citing the **Tamil Nadu Electricity Distribution Code (Clause 11)** and **CEA Meter Testing Regulations (Rule 57)**, exportable to formatted text or **PDF**.
5. **Complete MLOps Platform**:
   - Experiment tracking with **MLflow**.
   - Continuous Data and Feature Drift Monitoring using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests.
   - **Model Registry** with safety promotion gates ($F1 \ge 0.75$, Precision $\ge 0.70$, $F1_{cand} \ge F1_{prod}$) and **1-Click Safe Rollback**.
   - Immutable **SQLite** and **JSONL** audit trails for regulatory compliance.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Input Ingestion Layer"]
        A1[Physical Bill Card Image] --> B1[From-Scratch CRNN OCR]
        A2[Manual Entry / Portal Input] --> B2[Bill Input Schema]
        B1 --> B3{Confidence > 0.70?}
        B3 -- No --> B4[Human-in-the-Loop Fallback Verification]
        B3 -- Yes --> B2
        B4 --> B2
    end

    subgraph Verification["Deterministic Tariff & Audit Engine"]
        B2 --> C1[Meter Reading Difference Check]
        B2 --> C2[Billing Cycle Duration & Proration Check]
        B2 --> C3[Versioned TNERC Slab Breakdown Engine]
        C1 --> D1[Calculated Units]
        C2 --> D2[Proration Penalty Assessment]
        C3 --> D3[Statutory Grand Total]
        D1 & D3 --> E1[Arithmetic Discrepancy Evaluation]
    end

    subgraph ML_Layer["Machine Learning & Feature Engine (Zero Pretrained Weights)"]
        B2 --> F1[Domain Feature Engineering]
        F1 --> G1[Statistical Baseline Model]
        F1 --> G2[Isolation Forest]
        F1 --> G3[Deep PyTorch Autoencoder]
        G1 & G2 & G3 --> H1[Multi-Model Anomaly Consensus & Attribution]
    end

    subgraph Explainability["Bilingual Explainability & Grievance Engine"]
        E1 & H1 --> I1[Certainty Level Classifier]
        I1 --> J1[English Explanation Report]
        I1 --> J2[தமிழ் விளக்கம் (Tamil Report)]
        I1 --> J3[Statutory Grievance Petition Generator]
        J3 --> K1[Complaint PDF & Text Ready for TANGEDCO Submission]
    end

    subgraph MLOps["MLOps, Monitoring & Governance"]
        E1 & H1 --> L1[SQLite Audit Logger]
        L1 --> M1[PSI & KS Drift Monitor]
        M1 --> N1{Drift Alert?}
        N1 -- Yes --> O1[Auto-Retraining Trigger]
        O1 --> P1[Safety Promotion Gate]
        P1 --> Q1[Model Registry: Staging & Rollback]
    end
```

---

## 📁 Repository Directory Structure

```
billguard/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── config/
│   ├── config.yaml                # Master configuration
│   └── tariffs/                   # Versioned official TNERC tariff rules
│       ├── tn_domestic_lt1a_2022.yaml
│       ├── tn_domestic_lt1a_2023.yaml
│       ├── tn_domestic_lt1a_2024.yaml
│       └── simulated_progressive_solar.yaml
├── data/
│   ├── synthetic/                 # Isolated train/val/test consumer datasets
│   ├── sample_bills/              # Rendered TANGEDCO bill cards
│   └── ocr_synth/                 # Field crops and OCR labels
├── src/
│   ├── tariff/                    # Deterministic slab calculation & proration
│   │   ├── models.py
│   │   ├── rules.py
│   │   └── engine.py
│   ├── data_gen/                  # Archetypes, seasonal factors, synthetic generator
│   │   ├── scenario_definitions.py
│   │   ├── bill_generator.py
│   │   └── ocr_image_generator.py
│   ├── ocr/                       # From-scratch CRNN OCR pipeline
│   │   ├── model.py               # CNN + BiGRU + CTC Loss
│   │   ├── decoder.py             # Greedy CTC decoding & confidence scoring
│   │   ├── metrics.py             # CER & WER calculation
│   │   ├── dataset.py             # PyTorch Dataset & CTC Collator
│   │   ├── extractor.py           # Field bounding box cropper & fallback
│   │   └── train.py               # Scratch training script
│   ├── ml/                        # Unsupervised anomaly detection
│   │   ├── features.py            # 8 domain behavioral features
│   │   ├── baselines.py           # 3-sigma Z-score & surge rules
│   │   ├── isolation_forest.py    # Scratch-trained Isolation Forest
│   │   ├── autoencoder.py         # PyTorch bottleneck autoencoder
│   │   └── train.py               # Model benchmarking orchestrator
│   ├── explanation/               # Bilingual reasoning & petition generator
│   │   ├── engine.py              # Evidence certainty classification
│   │   ├── bilingual_templates.py # English and Tamil explanation text
│   │   └── grievance_generator.py # Formal petition text and ReportLab PDF
│   ├── mlops/                     # MLOps lifecycle management
│   │   ├── drift_monitor.py       # PSI and KS-test drift monitor
│   │   ├── registry.py            # Model registry, safety gate, rollback
│   │   ├── audit_logger.py        # SQLite and JSONL immutable audit trail
│   │   └── tracking.py            # MLflow experiment tracking interface
│   └── api/                       # FastAPI REST backend
│       ├── schemas.py
│       ├── routes.py
│       └── app.py
├── ui/                            # Streamlit multi-page frontend
│   ├── app.py                     # Main dashboard hub
│   ├── pages/
│   │   ├── 1_Bill_Verifier.py     # Bill upload, OCR preview, audit & grievance
│   │   ├── 2_Tariff_Explorer.py   # Versioned slabs & delay simulator
│   │   ├── 3_Historical_Analytics.py # Multi-cycle trend visualizer
│   │   └── 4_MLOps_Center.py      # Drift monitor, registry & retrain trigger
│   └── utils.py                   # Formatting helpers
├── tests/                         # Comprehensive pytest test suite (34 tests)
│   ├── test_tariff.py
│   ├── test_data_gen.py
│   ├── test_ocr.py
│   ├── test_ml_anomaly.py
│   ├── test_explanation.py
│   ├── test_mlops.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md            # Architectural specification & math models
│   ├── tariff_methodology.md      # TNERC LT-1A orders & proration formulation
│   └── final_year_project_report_outline.md # University report outline
├── scripts/
│   ├── seed_demo_data.py          # One-command data & model initialization
│   └── run_all_stages.py          # Stage-by-stage automated verification runner
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start & Setup Guide

### 1. Prerequisites
- Python 3.10 or 3.11
- Git

### 2. Local Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/billguard.git
cd billguard

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Synthetic Data & Train Scratch Models
```bash
python scripts/seed_demo_data.py
```
This script automatically:
1. Generates 1,200 longitudinal domestic bills across 100 consumers with realistic Tamil Nadu seasonal consumption.
2. Trains the **CRNN OCR Model** from random initialization on 400 synthetic text crops.
3. Fits the **Isolation Forest** and **PyTorch Autoencoder** from scratch on normal consumption patterns.
4. Initializes the **Model Registry** and populates sample prediction audit records.

### 4. Run the Full Test Suite
```bash
python scripts/run_all_stages.py
```
*All 34 automated unit, integration, OCR, tariff, drift, and API tests will run and report green passes.*

### 5. Launch the Application

#### Option A: Run Streamlit Frontend (Standalone or with Backend)
```bash
streamlit run ui/app.py
```
*Access the dashboard at `http://localhost:8501`.*

#### Option B: Run FastAPI REST API
```bash
uvicorn src.api.app:app --reload --port 8000
```
*Access interactive Swagger API documentation at `http://localhost:8000/docs`.*

#### Option C: Run via Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 🧪 Comprehensive Test Suite (34/34 Passing)

| Stage | Module | Tests | Status | Key Coverage |
|---|---|---|---|---|
| **Stage 1** | Tariff Engine | 7 | `PASSED` | 100-unit free subsidy, tiered slabs, overcharge, rollover, proration |
| **Stage 2** | Synthetic Data | 4 | `PASSED` | Consumer archetypes, summer AC surge, zero cross-set leakage |
| **Stage 3** | From-Scratch OCR | 4 | `PASSED` | CRNN forward shape, CTC tokenizer, CER/WER metrics, manual fallback |
| **Stage 4** | ML Anomaly Models | 4 | `PASSED` | 8-feature extraction, 3-sigma baseline, Isolation Forest, Autoencoder |
| **Stage 5** | Explanation & Grievance | 4 | `PASSED` | Bilingual synthesis, certainty grading, ReportLab PDF petition |
| **Stage 6** | MLOps & Governance | 5 | `PASSED` | PSI drift calculation, KS-test, safety gate promotion, rollback, SQLite audit |
| **Stage 7** | FastAPI Integration | 6 | `PASSED` | Health, tariffs, verify-bill, OCR image upload, drift report, registry |

---

## 🏛️ Official Tamil Nadu Tariff Reference (TNERC LT-1A)

Under official **TNERC Tariff Orders (No. 7 of 2022 and 2024 revision)**:
- **Bi-monthly Billing Cycle**: 60 days.
- **GoTN Subsidy**: First 100 units **100% free** for all domestic consumers.
- **Tiered Net Rates (2024)**:
  - 0 - 100 units: ₹0.00 / unit (Subsidized)
  - 101 - 200 units: ₹2.47 / unit
  - 201 - 400 units: ₹4.93 / unit
  - 401 - 500 units: ₹6.60 / unit
  - 501 - 600 units: ₹8.80 / unit
  - 601 - 800 units: ₹9.90 / unit
  - 801 - 1000 units: ₹11.00 / unit
  - Above 1000 units: ₹12.10 / unit
- **Fuel Adjustment (FPPCA)**: ₹0.32 / billable unit.
- **Billing Delay Detection**: Detects non-prorated 75+ day cycles that unfairly force consumers into higher tariff slabs.

---

## 📜 Legal and Ethical Disclaimer
BillGuard is an academic public-interest engineering project designed to assist consumers in verifying billing calculations. All mathematical checks are based on publicly available TNERC orders. In accordance with the Central Electricity Authority (Installation and Operation of Meters) Regulations (Rule 57) and Tamil Nadu Electricity Distribution Code (Clause 11), a high consumption score does not prove fraud, meter tampering, or utility malice without physical meter laboratory inspection.

---

## 🎓 Academic Capstone Submission Details
- **Project Title:** BillGuard: An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System
- **Documentation:** See [`docs/architecture.md`](docs/architecture.md), [`docs/tariff_methodology.md`](docs/tariff_methodology.md), and [`docs/final_year_project_report_outline.md`](docs/final_year_project_report_outline.md).
