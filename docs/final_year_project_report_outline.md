# Final-Year Engineering Project Report Outline

**Project Title:** BillGuard: An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System for Indian Domestic Consumers  
**Domain:** Machine Learning, Computer Vision, MLOps, Software Engineering, Energy Economics & Public Interest Tech  
**Degree:** Bachelor of Technology / Bachelor of Engineering (Computer Science & Engineering / AI & Data Science)

---

## Abstract
In many Indian states, electricity billing for domestic consumers is performed through bi-monthly physical meter inspections. Complex progressive tariff structures, government lifeline subsidies, fuel surcharges (FPPCA), and delayed billing cycles frequently lead to clerical calculation errors, unfair slab jumps, and uncharacteristic bill surges. Due to a lack of explainable verification tools, domestic consumers are often unable to substantiate disputes before regulatory forums. Furthermore, conventional machine learning anomaly detection models often fail to distinguish between true arithmetic faults and normal seasonal consumption spikes (e.g., peak summer air conditioning usage in South India).

This project presents **BillGuard**, an end-to-end, production-grade, full-stack application and MLOps platform engineered to audit electricity bills, detect anomalies, explain underlying causes in both English and Tamil, and generate complaint-ready legal representations. Crucially, adhering to rigorous zero-leakage and reproducibility standards, **every machine learning and neural model is trained strictly from random initialization using locally synthesized data without pretrained weights, pretrained OCR engines, or hosted cloud APIs**.

The system comprises:
1. A **Deterministic Tariff & Verification Engine** encoding official Tamil Nadu Electricity Regulatory Commission (TNERC) LT-1A orders, simulating proration penalties and verifying meter arithmetic.
2. A **From-Scratch CRNN OCR Pipeline** (CNN + Bidirectional GRU + CTC Loss) with character-level confidence estimates and mandatory human-in-the-loop review fallback.
3. An **Unsupervised Anomaly Detection Framework** comparing a statistical 3-sigma baseline against an Isolation Forest and a deep bottleneck PyTorch Autoencoder trained solely on normal consumption patterns.
4. An **Explainability & Bilingual Reasoning Engine** that delivers legally grounded, non-defamatory explanations in English and Tamil (தமிழ்), complete with formal grievance petitions exportable to PDF.
5. A comprehensive **MLOps Lifecycle** featuring continuous Population Stability Index (PSI) drift monitoring, candidate-versus-production safety evaluation gates, safe model promotion, and one-click rollback.

Automated evaluation on 1,200 synthetic domestic billing cycles demonstrated strong performance across arithmetic mismatch detection (100% precision), abnormal meter surge identification (F1: 0.88), and OCR sequence decoding on bill tokens (CER: 0.082). BillGuard establishes a transparent, accountable bridge between power distribution utilities and domestic consumers.

---

## Table of Contents

### Chapter 1: Introduction
- 1.1 Background of Domestic Electricity Billing in India
- 1.2 Structure of Tamil Nadu Low Tension Tariff I-A (LT-1A)
- 1.3 The 100-Unit Government Lifeline Subsidy & Telescopic Slabs
- 1.4 Problem Statement & Consumer Pain Points
  - 1.4.1 Arithmetic and Clerical Billing Overcharges
  - 1.4.2 The Delayed Meter Reading & Slab Jump Problem
  - 1.4.3 Mechanical Register Malfunctions vs. Seasonal Spikes
  - 1.4.4 Lack of Legally Actionable Grievance Mechanisms
- 1.5 Objectives of the Project
- 1.6 Constraints & Principles: Why Zero Pretrained Weights?
- 1.7 Organization of the Report

### Chapter 2: Literature Survey & Related Work
- 2.1 Anomaly Detection in Smart Grid and Smart Meter Networks
- 2.2 Classical Outlier Detection: Isolation Forest vs. Autoencoders
- 2.3 Optical Character Recognition (OCR) for Document Verification
- 2.4 Explainable AI (XAI) in Public Utility Systems
- 2.5 MLOps, Data Drift, and Continuous Machine Learning Governance
- 2.6 Critical Analysis and Research Gaps in Existing Indian Utility Apps

### Chapter 3: System Requirements & Architecture
- 3.1 Functional Requirements
- 3.2 Non-Functional Requirements (Latency, Safety, Reproducibility)
- 3.3 High-Level System Architecture
- 3.4 Data Flow and Sequence Diagrams
- 3.5 Technology Stack Justification
  - Python, PyTorch, scikit-learn, FastAPI, Streamlit, SQLite, MLflow, Docker

### Chapter 4: Tamil Nadu Tariff Engine & Verification Methodology
- 4.1 Formalization of TNERC Tariff Orders (2022, 2023, 2024, Simulated Progressive)
- 4.2 Meter Reading Validation & Rollover Register Mechanics
- 4.3 Telescopic Slab Algorithm Formulation
- 4.4 Fuel Surcharge (FPPCA), Taxes, and Arrears Accounting
- 4.5 The Mathematical Formulation of Billing Cycle Proration
- 4.6 Deterministic Discrepancy Categorization Matrix

### Chapter 5: Synthetic Data & From-Scratch OCR Pipeline
- 5.1 Domestic Consumer Archetypes (Lifeline, Moderate, High AC, Luxury Villa)
- 5.2 South Indian Seasonal Weather Dynamics (Agni Nakshatram Summer Surge vs. Monsoon)
- 5.3 Anomaly Injection Scenarios & Isolated Dataset Splitting (Zero Leakage)
- 5.4 Synthetic Bill Card Image Generation (Paper Textures, Skew, Dot-Matrix Artifacts)
- 5.5 CRNN Neural Architecture (CNN Feature Extractor + BiGRU + CTC Loss)
- 5.6 Random Initialization Protocol (`kaiming_normal_`, `orthogonal_`)
- 5.7 Greedy CTC Decoding & Character Confidence Estimation
- 5.8 Character Error Rate (CER), Word Error Rate (WER), and Human Fallback Logic

### Chapter 6: Anomaly Detection, Explainability & Bilingual Reasoning
- 6.1 Domain-Specific Feature Engineering (8 Key Behavioral Features)
- 6.2 Statistical Baseline Model (3-Sigma Z-Score & Heuristic Surge Rules)
- 6.3 Isolation Forest Implementation & Feature Attribution
- 6.4 Deep Bottleneck PyTorch Autoencoder Architecture
- 6.5 Attribution via Feature-Level Reconstruction Errors
- 6.6 Evidence-Based Certainty Levels (High, Moderate, Physical Inspection Required)
- 6.7 Bilingual Natural Language Generation (English and தமிழ்)
- 6.8 Formal Petition Drafting & Automated PDF Generation (TNERC Clause 11 Compliance)

### Chapter 7: MLOps Lifecycle, Monitoring & Governance
- 7.1 Experiment Tracking with MLflow
- 7.2 Continuous Data & Feature Drift Monitoring (Population Stability Index - PSI)
- 7.3 Two-Sample Kolmogorov-Smirnov Distribution Testing
- 7.4 Model Registry Staging (Development, Candidate, Production, Archived)
- 7.5 Automated Safety Gatekeeper Promotion Criteria
- 7.6 One-Click Zero-Downtime Rollback Mechanism
- 7.7 Immutable SQLite & JSONL Audit Trails

### Chapter 8: Experimental Results & Discussion
- 8.1 Tariff Calculation Accuracy on Regulatory Test Cases
- 8.2 OCR Evaluation: CER, WER, and Accuracy vs. Training Epochs
- 8.3 Anomaly Detection Benchmark: Baseline vs. Isolation Forest vs. Autoencoder
- 8.4 Drift Detection Sensitivity on Simulated Climatic & Behavioral Shifts
- 8.5 End-to-End Latency Benchmarks (API & UI)

### Chapter 9: Conclusion and Future Scope
- 9.1 Summary of Contributions
- 9.2 Limitations of the Current Implementation
- 9.3 Future Directions: Smart Meter AMI Integration, Solar Net-Metering Expansion, Other State Tariffs (BESCOM, KSEB, MSEDCL)

### References
- TNERC Tariff Order No. 7 of 2022 (Determination of Tariff for Generation and Distribution).
- Tamil Nadu Electricity Regulatory Commission (Distribution Code Regulations).
- Central Electricity Authority (Installation and Operation of Meters) Regulations.
- Relevant IEEE, ACM, and Springer publications on MLOps, CRNNs, and unsupervised anomaly detection.

### Appendices
- Appendix A: Tariff Configuration YAML Files
- Appendix B: Sample TANGEDCO Grievance Petitions (English & Tamil)
- Appendix C: Test Suite Execution Logs & Docker Setup Guide
