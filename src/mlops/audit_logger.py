"""Prediction Audit Logger backed by SQLite and append-only JSONL."""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class AuditEntry(BaseModel):
    consumer_number: str
    billing_date: str
    previous_reading: float
    current_reading: float
    calculated_units: float
    billed_units: Optional[float] = None
    calculated_total: float
    billed_amount: Optional[float] = None
    discrepancy_amount: Optional[float] = None
    is_anomaly: bool
    anomaly_category: str
    certainty_level: str
    ml_score: float = 0.0
    model_version: int = 1
    user_feedback: str = "PENDING"  # PENDING, AGREE, DISAGREE


class AuditLogger:
    """
    Persists immutable audit trails of all system calculations and predictions.
    Supports continuous feedback collection and monitoring queries.
    """

    def __init__(self, db_path: str = "./data/billguard_audit.db", jsonl_path: str = "./data/audit_logs.jsonl"):
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Create audit log schema if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                consumer_number TEXT NOT NULL,
                billing_date TEXT,
                previous_reading REAL,
                current_reading REAL,
                calculated_units REAL,
                billed_units REAL,
                calculated_total REAL,
                billed_amount REAL,
                discrepancy_amount REAL,
                is_anomaly INTEGER,
                anomaly_category TEXT,
                certainty_level TEXT,
                ml_score REAL,
                model_version INTEGER,
                user_feedback TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_prediction(self, entry: AuditEntry) -> int:
        """Append log record to SQLite and JSONL."""
        now_ts = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_logs (
                timestamp, consumer_number, billing_date, previous_reading, current_reading,
                calculated_units, billed_units, calculated_total, billed_amount, discrepancy_amount,
                is_anomaly, anomaly_category, certainty_level, ml_score, model_version, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_ts, entry.consumer_number, entry.billing_date, entry.previous_reading, entry.current_reading,
            entry.calculated_units, entry.billed_units, entry.calculated_total, entry.billed_amount,
            entry.discrepancy_amount, 1 if entry.is_anomaly else 0, entry.anomaly_category,
            entry.certainty_level, entry.ml_score, entry.model_version, entry.user_feedback
        ))
        log_id = cur.lastrowid
        conn.commit()
        conn.close()

        # Append to JSONL
        try:
            record_dict = entry.model_dump()
            record_dict["id"] = log_id
            record_dict["timestamp"] = now_ts
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict) + "\n")
        except Exception as e:
            print(f"[Warning] Failed to append to JSONL log: {e}")

        return log_id

    def fetch_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch latest prediction audit records."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_audit_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate metrics from audit log."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(is_anomaly), AVG(discrepancy_amount) FROM audit_logs")
        row = cur.fetchone()
        total = row[0] or 0
        anomalies = row[1] or 0
        avg_discrepancy = row[2] or 0.0
        conn.close()

        return {
            "total_bills_audited": total,
            "anomalies_flagged": anomalies,
            "anomaly_rate_pct": round((anomalies / max(1, total)) * 100.0, 2),
            "average_discrepancy": round(avg_discrepancy, 2),
        }
