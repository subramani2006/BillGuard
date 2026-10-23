"""FastAPI application entrypoint for BillGuard."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="BillGuard API",
    description="Explainable, MLOps-Driven Electricity Bill Anomaly Detection and Verification System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit / React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Welcome to BillGuard: Explainable Electricity Bill Verification API",
        "documentation": "/docs",
        "health": "/api/health",
        "default_state": "Tamil Nadu (TANGEDCO LT-1A)",
    }
