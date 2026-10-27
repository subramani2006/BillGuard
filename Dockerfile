# Production-Grade Dockerfile for BillGuard on Amazon Web Services (AWS) & Cloud
# Base image: Official lightweight Python 3.10 slim on Debian Linux
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install essential system dependencies:
# - curl: for healthcheck probe
# - libglib2.0-0: for headless image and computer vision processing
# - fonts-dejavu-core: TrueType fonts for synthetic bill generation and OCR verification
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, configs, and assets
COPY ui/ ./ui/
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/
COPY pages/ ./pages/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Expose Streamlit default port
EXPOSE 8501

# Container health probe for AWS App Runner / ECS / Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
