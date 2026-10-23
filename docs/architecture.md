# BillGuard: System Architecture & Design Specification

## 1. System Overview
BillGuard is an explainable, MLOps-driven electricity bill verification and anomaly detection system specifically tailored for Indian domestic consumers. The system strictly separates **deterministic mathematical calculation audits** from **heuristic machine learning behavioral anomaly detection**.

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
        B2 --> C1[Meter Register Difference Check]
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

## 2. Mathematical Tariff Model (Tamil Nadu TANGEDCO LT-1A)

### 2.1 Meter Reading Verification
$$\Delta U = (R_{curr} - R_{prev}) \times K_{meter}$$
Where:
- $R_{curr}$ = Current meter reading index
- $R_{prev}$ = Previous meter reading index
- $K_{meter}$ = Meter multiplying constant (1.0 for domestic LT)

If $R_{curr} < R_{prev}$, the system tests for meter register rollover:
$$\Delta U_{rollover} = (10^N - R_{prev}) + R_{curr}$$
where $N \in \{5, 6\}$ digits. If inverted without rollover bounds, an invalid meter reading error is flagged.

### 2.2 Telescopic Slab Calculation
For $k$ tiered slabs where slab $i$ has lower boundary $L_i$, upper boundary $U_i$, gross rate $r_i$, and subsidy rate $s_i$:
$$u_i = \max(0, \min(U_i \cdot \alpha, \Delta U) - L_i \cdot \alpha)$$
$$\text{Gross Energy Charge} = \sum_{i=1}^k u_i \cdot r_i$$
$$\text{Government Subsidy} = \sum_{i=1}^k u_i \cdot s_i$$
$$\text{Net Energy Charge} = \sum_{i=1}^k u_i \cdot (r_i - s_i)$$

Where $\alpha$ represents the proration factor:
$$\alpha = \frac{\text{Actual Billing Days}}{\text{Standard Cycle Days (60)}}$$

### 2.3 Grand Payable Total & Discrepancy
$$\text{Calculated Total} = \text{Net Energy Charge} + \text{Fixed Charges} + \text{FPPCA} + \text{Electricity Tax} + \text{Arrears}$$
$$\text{Discrepancy} = \text{Billed Amount} - \text{Calculated Total}$$

---

## 3. From-Scratch OCR Neural Network
- Architecture: CRNN (Convolutional Recurrent Neural Network).
- CNN Backbone: 4 Conv2d layers with BatchNorm2d, ReLU activations, and MaxPool2d downsampling width by 4 ($256 \to 64$) and height to 1 ($32 \to 1$).
- RNN Backbone: 2-layer Bidirectional GRU (hidden size: 64, output: 128).
- Output: Linear layer mapping to 42 character vocabulary classes.
- Loss: PyTorch Connectionist Temporal Classification (`nn.CTCLoss(blank=0, zero_infinity=True)`).
- Weights initialized strictly randomly (`kaiming_normal_`, `xavier_uniform_`, `orthogonal_`). Zero pretrained weights or external APIs used.

---

## 4. Anomaly Detection Engine (Trained from Scratch)
- **Features Extracted**:
  1. `daily_consumption` = $\Delta U / \text{days}$
  2. `billing_days` = Duration in days
  3. `ratio_to_history_mean` = $\Delta U / \bar{U}_{hist}$
  4. `seasonal_deviation_ratio` = $(\Delta U - \hat{U}_{season}) / \hat{U}_{season}$
  5. `peak_to_average_ratio` = $\Delta U / \max(U_{hist})$
  6. `discrepancy_kwh_rate` = $\text{Discrepancy} / \text{Total}$
  7. `arrears_ratio` = $\text{Arrears} / \text{Total}$
  8. `slab_jump_magnitude` = $\text{Slab}_{curr} - \text{Slab}_{median}$
- **Models**:
  1. **Baseline**: 3-sigma Z-score & heuristic spike rules.
  2. **Isolation Forest**: 100 isolation trees fit on normalized local training set.
  3. **PyTorch Autoencoder**: Bottleneck architecture (8 -> 16 -> 8 -> 4 -> 8 -> 16 -> 8) trained solely on normal consumption patterns; reconstruction error serves as continuous anomaly score and per-feature attribution.

---

## 5. MLOps Lifecycle & Monitoring
- **Drift Monitoring**:
  - Population Stability Index (PSI):
    $$PSI = \sum_{i=1}^k (Actual_i - Baseline_i) \times \ln\left(\frac{Actual_i}{Baseline_i}\right)$$
  - Kolmogorov-Smirnov 2-sample test ($p$-value $< 0.05$).
- **Model Registry & Gatekeeper**:
  - Tracks versions: Development, Candidate, Production, Archived.
  - Candidate promotion requires: $F1_{cand} \ge F1_{prod}$, $F1_{cand} \ge 0.75$, $\text{Precision}_{cand} \ge 0.70$.
  - 1-click safe rollback restores the preceding production model version with zero downtime.
- **Audit Logging**:
  - SQLite table + JSONL append log recording inputs, calculations, model versions, and verdicts.
