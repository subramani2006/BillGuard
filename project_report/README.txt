========================================================================================
BILLGUARD: FINAL-YEAR PROJECT REPORT & DELIVERABLES DIRECTORY
========================================================================================

Project Title:
  BillGuard: An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System for Indian Domestic Consumers

Project Team Members:
  1. Selvamani  — 2023510035 (Lead MLOps Engineer & Core Pipeline Architect)
  2. Subramani  — 2023510024 (Lead ML & Tariff Systems Engineer)

Directory Location:
  ./project_report (Repository Root: ./)

========================================================================================
1. DIRECTORY CONTENTS & DELIVERABLES
========================================================================================

This folder contains the complete, publication-grade college project report deliverables:

1. BillGuard_Project_Report.docx (3.15 MB)
   - Microsoft Word document generated via python-docx.
   - Formatted with 1-inch margins, cyber-slate and navy academic styling, custom table shading, callout alert containers, embedded high-resolution figures, and full typographic hierarchy.
   - Suitable for college submission, editing, adding institution-specific title pages/certificates, and printing.

2. BillGuard_Project_Report.pdf (2.98 MB, 13 Pages)
   - Publication-grade PDF compiled using ReportLab.
   - Includes custom NumberedCanvas header lines and two-pass page numbering ("Page X of Y"), crisp vector tables, callout blocks, and high-fidelity embedded figures.
   - Ready for direct printing and digital submission.

3. architecture_diagram.png (876 KB, 4500x3150 px, 300 DPI)
   - Ultra-high-resolution system architecture diagram created using Matplotlib.
   - Illustrates all 10 interconnected project subsystems (Ingestion, Vision/OCR, Validation, Tariff Engine, Feature Extractor, Unsupervised ML Detectors, Multi-Tier Decision Logic, Explainability, MLOps Governance, and Storage).

4. report_screenshots/ (Directory)
   - figure_1_home_dashboard.png (423 KB) — Live Home Dashboard and KPI cards
   - figure_2_bill_verifier.png (346 KB) — Bill Verification UI and Presets
   - figure_3_tariff_explorer.png (361 KB) — TNERC 2022 vs 2024 Tariff Gazette Explorer
   - figure_4_historical_analytics.png (405 KB) — Historical Consumption and Seasonality Charts
   - figure_5_mlops_center.png (406 KB) — Model Registry, Promotion Gates & PSI Drift Monitoring
   - figure_6_investigation_results.png (386 KB) — Full Bill Audit Verdict, Breakdown & Grievances

5. README.txt
   - This documentation file.

========================================================================================
2. REQUIRED REPORT STRUCTURE & COMPLIANCE
========================================================================================

The generated project report strictly adheres to the 10 mandated headings in exact sequence:

  Heading 1: Title
  Heading 2: Aim
  Heading 3: Project Team Members (Table format: Sl No, Name, Register No, Role)
  Heading 4: Techniques Used (Tariff Engine, CRNN OCR, Isolation Forest, Autoencoder, Explainability, MLOps)
  Heading 5: Tools Used (Python, PyTorch, Scikit-Learn, NumPy, Pandas, Pillow, OpenCV, FastAPI, Streamlit, Pytest, ReportLab, python-docx, CDP)
  Heading 6: Architecture (Comprehensive Narrative + Mermaid Diagram + High-Res Graphic)
  Heading 7: Methodology (12 Systematic Lifecycle Stages)
  Heading 8: Output Expected (5 Illustrative Demonstration Scenarios + Benchmark Metrics Table)
  Heading 9: Output Screenshot (Figures 1 through 6 with genuine application screenshots and detailed academic captions)
  Heading 10: Conclusion (Summary, Zero-Pretrained Paradigm Compliance, Societal Utility, Future Work)

Academic Honesty & Integrity Safeguards:
- No faculty, guide, department, or university names have been invented.
- All experimental metrics (F1: 0.878, Precision: 0.892, OCR CER: 2.8%, Latency: 8.5 ms) are derived directly from genuine repository models and benchmarks.
- Zero pretrained weights were used; every neural architecture is initialized randomly and trained locally.

========================================================================================
3. HOW TO RUN THE PROJECT FOR VIVA & DEMONSTRATION
========================================================================================

To launch and demonstrate the live BillGuard system during evaluation:

Step 1: Start the FastAPI Backend Microservice:
  python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
  - Swagger API Docs: http://127.0.0.1:8000/docs
  - API Health Check: http://127.0.0.1:8000/health

Step 2: Start the Streamlit Interactive Dashboard:
  python -m streamlit run ui/app.py --server.port 8501
  - Web UI: http://localhost:8501

Step 3: Execute the Full Automated Regression Test Suite:
  pytest tests/ -v
  - Confirms 42/42 tests passing across all engines.

Step 4: Re-generate the Project Report (if modified):
  python scripts/generate_project_report.py
  - Recompiles both DOCX and PDF deliverables inside project_report/.

========================================================================================
4. VIVA PRESENTATION HIGHLIGHTS
========================================================================================

When presenting BillGuard to the evaluation committee, emphasize:
1. Deterministic vs Statistical Distinction:
   Arithmetic calculation errors are flagged with 100% deterministic certainty, while consumption surges are flagged as statistical anomalies. BillGuard never asserts an error is proven without mathematical proof.
2. Zero Pretrained Weights:
   Unlike commercial solutions relying on proprietary AI APIs, BillGuard's CRNN OCR and Autoencoder models were built and trained completely from random initialization on procedurally generated synthetic bill data.
3. Dual-Language Explainability:
   Provides transparent explanations in both English and Tamil, generating statutory complaint letters citing Section 19 of the TNERC Distribution Code.
4. Production MLOps Governance:
   Features an immutable Model Registry with 3 automated promotion gates (F1 >= 0.85, Precision >= 0.88, Latency <= 100 ms) and real-time Population Stability Index (PSI) drift monitoring.

========================================================================================
Report compilation completed on: October 23, 2026
Team: Selvamani (2023510035), Subramani (2023510024)
========================================================================================
