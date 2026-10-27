# 🚀 BillGuard — Cloud Deployment Guide

This guide provides step-by-step instructions for deploying **BillGuard: An Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System for Indian Domestic Consumers** to the web using **Streamlit Community Cloud** (recommended) or as a Docker container.

---

## 📋 Table of Contents
1. [Prerequisites & Repository Verification](#1-prerequisites--repository-verification)
2. [Cloud Compatibility & Architecture Decisions](#2-cloud-compatibility--architecture-decisions)
3. [Deploying to Streamlit Community Cloud (Step-by-Step)](#3-deploying-to-streamlit-community-cloud-step-by-step)
4. [Testing & Verifying the Live Deployed Application](#4-testing--verifying-the-live-deployed-application)
5. [Streamlit Cloud Storage & MLOps Considerations](#5-streamlit-cloud-storage--mlops-considerations)
6. [Updating the Deployed Application](#6-updating-the-deployed-application)
7. [Docker / Self-Hosted Containerization](#7-docker--self-hosted-containerization)
8. [Amazon Web Services (AWS) Production Deployment](#8-amazon-web-services-aws-production-deployment)

---

## 1. Prerequisites & Repository Verification

Before initiating deployment, verify that:
1. Your project is uploaded to a GitHub repository (e.g., `https://github.com/<your-username>/BillGuard`).
2. The repository contains the following core cloud assets:
   - `requirements.txt` (includes `opencv-python-headless`, `torch`, `scikit-learn`, `reportlab`, `plotly`, `python-docx`)
   - `packages.txt` (Debian system dependencies: `libglib2.0-0`, `fonts-dejavu-core`)
   - `.streamlit/config.toml` (Theme and server flags)
   - `ui/app.py` (Main Streamlit Dashboard entry point)
   - `models/` directory with trained model checkpoints:
     * `models/isolation_forest.joblib` (Production model v1)
     * `models/autoencoder_scratch.pth` (Candidate model v2)
     * `models/ocr_crnn_scratch.pth` (From-scratch CRNN OCR)
     * `models/registry.json` (Model registry metadata)
   - `config/tariffs/` (Official TNERC 2022 and 2024 YAML tariff orders)
   - `data/synthetic/` (Synthetic billing datasets for analytics)

---

## 2. Cloud Compatibility & Architecture Decisions

During preparation, several architectural refinements were made to ensure zero-downtime cloud operation on Debian Linux (Streamlit Community Cloud):

1. **Headless Computer Vision (`opencv-python-headless`)**:
   - Standard `opencv-python` requires native X11/OpenGL display libraries (`libGL.so.1`, `libXext.so.6`), which are absent on headless cloud containers and cause immediate crashes.
   - Replaced with `opencv-python-headless>=4.8.0` in `requirements.txt`.

2. **Cross-Platform Path Resolution (`sys.path`)**:
   - Streamlit Cloud runs apps from the repository root, but individual pages execute within their respective scopes.
   - `ui/app.py`, `app.py`, and all sub-pages dynamically inject the repository root into `sys.path`, preventing `ModuleNotFoundError: No module named 'src'`.

3. **Dual Entry-Point Redundancy**:
   - **Primary Entry Point:** `ui/app.py`
   - **Fallback Entry Point:** `app.py` (at root)
   - Both entry points support the multi-page structure (`1_Bill_Verifier.py`, `2_Tariff_Explorer.py`, `3_Historical_Analytics.py`, `4_MLOps_Center.py`).

4. **Zero Pretrained Weights Preservation**:
   - All models are lightweight (<2 MB each) and trained strictly from random initialization on procedurally generated synthetic data.
   - Stored directly in Git without requiring Git LFS or external model hubs.

---

## 3. Deploying to Streamlit Community Cloud (Step-by-Step)

Streamlit Community Cloud offers free, permanent hosting for Streamlit applications with automatic Git-integrated CI/CD.

### Step 3.1: Sign In to Streamlit Community Cloud
1. Navigate to: [**https://share.streamlit.io/**](https://share.streamlit.io/)
2. Click **Continue with GitHub** and authorize Streamlit to read your repositories.

### Step 3.2: Create a New App
1. Once logged in to the Streamlit workspace, click the **"New app"** button (top right).
2. Choose **"Deploy an app"** (or **"I already have an app"**).

### Step 3.3: Configure App Settings
Fill in the deployment form with the following exact parameters:
- **Repository:** `<your-github-username>/BillGuard` (select from the dropdown)
- **Branch:** `main` (or `master`)
- **Main file path:** `ui/app.py` *(Note: `app.py` will also work seamlessly)*
- **App URL (optional):** You can customize the subdomain, e.g., `billguard-tnerc.streamlit.app`

### Step 3.4: Advanced Settings (Optional)
- Click **"Advanced settings"**.
- **Python version:** Select **`3.10`** or **`3.11`**.
- **Secrets:** BillGuard does not require private API keys (zero external cloud dependencies). If connecting to a remote FastAPI backend in the future, you can optionally define:
  ```toml
  BILLGUARD_API_URL = "https://your-api-domain.com/api"
  ```
  *(By default, the UI runs fully self-contained in standalone mode).*

### Step 3.5: Deploy and Monitor
1. Click **"Deploy!"**.
2. Streamlit Cloud will spin up a Debian container, read `packages.txt`, install dependencies from `requirements.txt`, and boot the app.
3. Click the bottom-right terminal icon (**"Manage app"**) to observe build logs in real time.
4. Deployment typically takes **2 to 3 minutes**. Once complete, your public URL will be active!

---

## 4. Testing & Verifying the Live Deployed Application

Once the application is live at `https://<your-app-subdomain>.streamlit.app`, perform the following smoke tests:

| Page / Feature | Verification Procedure | Expected Result |
| :--- | :--- | :--- |
| **1. Home Dashboard** | Open the base URL. Inspect metric cards, statutory tariff status, and KPI blocks. | Hero banner renders with cyber-slate theme, all 4 metric cards display without errors. |
| **2. Bill Verifier (Manual)** | Select preset *"Scenario A: Confirmed Arithmetic Overcharge"*, click *"Step 3: Run Audit"*. | Audit detects ₹875 overcharge, displays `CONFIRMED_CALCULATION_DISCREPANCY` badge, itemizes slabs. |
| **3. Bill Verifier (OCR)** | Click *"Sample Bill Image"*, click *"Extract fields"*. | CRNN OCR extracts meter readings and consumer numbers; confidence table renders. |
| **4. Grievance Download** | On the Verifier page, expand the grievance section and click *"Download PDF"*. | PDF downloads cleanly (`TANGEDCO_Representation_*.pdf`) containing formal legal petition text. |
| **5. Tariff Explorer** | Navigate to *Tariff Explorer*. Select *TNERC 2024 LT-1A*. Test the proration slider (e.g. 78 days). | Interactive Plotly slab visualization renders; proration warnings display if extended. |
| **6. Historical Analytics** | Navigate to *Historical Analytics*. Switch between consumer archetypes (Lifeline, Regular, AC Summer Peak). | Plotly longitudinal time-series, moving average bands, and anomaly distribution plots render smoothly. |
| **7. MLOps Center** | Navigate to *MLOps Center*. Review the Model Registry and Promotion Gates. | Both Model v1 (Production) and Model v2 (Candidate) appear; all 3 promotion gates pass in green. |

---

## 5. Streamlit Cloud Storage & MLOps Considerations

Streamlit Community Cloud operates on an **ephemeral container filesystem**:
- **Model Registry & Configurations:** Read from version-controlled files (`models/registry.json`, `config/tariffs/`). These are persistent across reboots because they are tracked in Git.
- **Audit Logs:** In a cloud container, new entries appended to `data/audit_logs.jsonl` persist during the active session. If the container sleeps or reboots after inactivity, filesystem changes revert to the latest Git commit.
- **Enterprise Persistent Storage (Optional Future Extension):** For multi-tenant production deployments requiring durable audit storage, connect the `AuditLogger` to a hosted PostgreSQL/Supabase instance by setting `DATABASE_URL` in Streamlit Secrets.

---

## 6. Updating the Deployed Application

Streamlit Community Cloud continuously monitors your GitHub repository:
1. Make your code or configuration modifications locally.
2. Commit and push your changes to GitHub:
   ```bash
   git add .
   git commit -m "feat: enhance tariff rule parameters"
   git push origin main
   ```
3. Streamlit Cloud automatically detects the push, pulls the latest commit, and redeploys the app within ~30 seconds.

---

## 7. Docker / Self-Hosted Containerization

For on-premises institutional hosting or private cloud servers, the production `Dockerfile` in the root directory can be built and executed locally:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    fonts-dejavu-core \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

To build and run locally:
```bash
docker build -t billguard:latest .
docker run -p 8501:8501 billguard:latest
```

---

## 8. Amazon Web Services (AWS) Production Deployment

BillGuard can be deployed to **Amazon Web Services (AWS)** using a production-grade, highly available, and cost-effective cloud architecture designed specifically for Indian domestic consumer workloads.

### 8.1 AWS Cloud Architecture Overview

```
+---------------------------------------------------------------------------------+
|                                 AMAZON WEB SERVICES (AWS)                       |
|                          Region: ap-south-1 (Mumbai, India)                     |
|                                                                                 |
|  +--------------------+       +-----------------------------------------------+ |
|  | GitHub Repository  | ====> | AWS App Runner (Serverless Container)         | |
|  | subramani2006/     | (Push)|  - Auto-builds from root Dockerfile           | |
|  | BillGuard          |       |  - Port 8501 (Streamlit Engine)               | |
|  +--------------------+       |  - Auto-scales & scales-to-zero               | |
|                               |  - HTTPS / TLS managed certificate            | |
|                               +-----------------------+-----------------------+ |
|                                                       |                         |
|                               +-----------------------+                         |
|                               |                       |                         |
|                               v                       v                         |
|                     +-------------------+   +--------------------+              |
|                     | Amazon S3 Bucket  |   | Amazon CloudWatch  |              |
|                     | (Audit PDFs &     |   | (Logs, Metrics,    |              |
|                     | Synthetic Bills)  |   | Drift Alarms)      |              |
|                     +-------------------+   +--------------------+              |
+---------------------------------------------------------------------------------+
```

| AWS Service | Role in BillGuard | Pricing & Free Tier Safety |
| :--- | :--- | :--- |
| **AWS App Runner** | Managed container hosting for Streamlit web UI | Auto-scales down when idle; builds directly from GitHub without needing Docker Desktop. |
| **Amazon S3** | Durable storage for generated grievance petitions & sanitized bills | **12-Month Free Tier:** 5 GB storage, 20,000 GET, 2,000 PUT requests free/month ($\approx$ **₹0**). |
| **Amazon CloudWatch** | Ingestion of audit telemetry, latency tracking, and error alarms | **Always Free Tier:** 5 GB log ingestion, 10 custom metrics, 3 metric alarms free/month ($\approx$ **₹0**). |

---

### 8.2 Phase A: AWS Account Setup & Zero-Spend Cost Protection

> [!IMPORTANT]
> If you are a beginner, always set up a **Zero-Spend Budget Alert** immediately after creating your account to ensure you are never billed unexpectedly.

1. **Create an AWS Account**:
   - Go to [**https://aws.amazon.com/free/**](https://aws.amazon.com/free/).
   - Click **Create an AWS Account** and complete registration (12 months free tier included).
2. **Select India Region**:
   - In the top-right corner of the AWS Management Console, select **Asia Pacific (Mumbai) `ap-south-1`**. This ensures minimum latency for Indian domestic billing calculations.
3. **Configure AWS Zero-Spend Budget Alert**:
   - Search for **AWS Budgets** in the top search bar.
   - Click **Create budget** $\to$ Select **Zero spend budget** template.
   - Enter your email address to receive immediate alerts if any service exceeds $0.01.
   - Click **Create budget**.

---

### 8.3 Phase B: Deploying BillGuard via AWS App Runner (Recommended)

AWS App Runner connects directly to your GitHub repository and automatically builds and runs the container using the repository's `Dockerfile`. **You do NOT need Docker Desktop installed on your local computer.**

#### Step 1: Open AWS App Runner
- In the AWS Console search bar, type `App Runner` and open the service.
- Click **Create service**.

#### Step 2: Connect Source Code Repository
- **Source repository type**: Choose **Source code repository**.
- Click **Add new** under GitHub connection, authorize your GitHub account (`subramani2006`), and select repository `subramani2006/BillGuard`.
- **Branch**: Select `main`.
- **Deployment trigger**: Select **Automatic** (every push to `main` redeploys the app).
- Click **Next**.

#### Step 3: Configure Build & Container Runtime
- **Runtime**: Select **Dockerfile** (AWS App Runner will automatically detect the production `Dockerfile` in the root).
- **Port**: Enter `8501` (Streamlit standard port).
- Click **Next**.

#### Step 4: Configure Service & Resources
- **Service name**: `billguard`
- **Virtual CPU & Memory**: Select **1 vCPU, 2 GB Memory**.
- **Environment variables**: Add the following key-value pairs (matching `.env.example`):
  | Key | Value |
  | :--- | :--- |
  | `ENVIRONMENT` | `production` |
  | `PORT` | `8501` |
  | `STREAMLIT_SERVER_HEADLESS` | `true` |
  | `AWS_REGION` | `ap-south-1` |
  | `CLOUDWATCH_ENABLED` | `true` |
- Click **Next** $\to$ Review all configurations $\to$ Click **Create & deploy**.

#### Step 5: Verify Live Deployment
- Deployment takes 3 to 5 minutes as AWS builds the image and spins up the container.
- Once the status changes to **Running**, App Runner provides a permanent, secure HTTPS URL:
  `https://<unique-id>.ap-south-1.awsapprunner.com`
- Test the application using the verification test matrix in [Section 4](#4-testing--verifying-the-live-deployed-application).

---

### 8.4 Phase C: Persistent Audit Storage with Amazon S3 (Optional)

By default, BillGuard writes audit trails and generated petition PDFs to local disk (`data/audit_logs.jsonl`). To enable durable cloud storage in S3:

1. In the AWS Console, search for **S3** and click **Create bucket**.
2. **Bucket name**: e.g., `billguard-audit-storage-<your-roll-no>`.
3. **AWS Region**: Select `Asia Pacific (Mumbai) ap-south-1`.
4. Keep **Block all public access** checked (audits and bills must remain private).
5. Enable **Bucket encryption** with standard Amazon S3-managed keys (`SSE-S3`) at zero additional cost.
6. Click **Create bucket**.
7. In App Runner environment variables, set:
   ```bash
   AWS_S3_BUCKET_NAME=billguard-audit-storage-<your-roll-no>
   ```

---

### 8.5 Phase D: Telemetry & Monitoring with Amazon CloudWatch

1. In the AWS Console, search for **CloudWatch** $\to$ click **Log groups**.
2. App Runner automatically streams stdout logs to `/aws/apprunner/billguard`.
3. You can set a CloudWatch metric alarm on `5XX` HTTP response codes or application exceptions:
   - Metric: `AppRunner/HTTPCode_5XX`
   - Threshold: `> 0 for 1 datapoint`
   - Notification: Send alert to your personal email via Amazon SNS.

---

*Authored by: Selvamani (2023510035) & Subramani (2023510024)*  
*BillGuard Capstone Project — Final Year Engineering*

