"""Executes and verifies all stages of BillGuard."""

import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_stage(title: str, test_file: str):
    print("\n" + "=" * 65)
    print(f"RUNNING: {title}")
    print("=" * 65)
    cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[FAIL] FAILED STAGE: {title}")
        sys.exit(res.returncode)
    print(f"\n[PASS] PASSED STAGE: {title}")


def main():
    print("=" * 70)
    print("BILLGUARD: COMPLETE STAGE-BY-STAGE VERIFICATION RUNNER")
    print("=" * 70)

    stages = [
        ("Stage 1: Tamil Nadu Tariff Engine & Rules", "tests/test_tariff.py"),
        ("Stage 2: Synthetic Data & Scenario Generator", "tests/test_data_gen.py"),
        ("Stage 3: From-Scratch OCR Neural Network", "tests/test_ocr.py"),
        ("Stage 4: ML Anomaly Detection & Baselines", "tests/test_ml_anomaly.py"),
        ("Stage 5: Bilingual Explanations & Grievance Generator", "tests/test_explanation.py"),
        ("Stage 6: MLOps Drift, Registry & Audit Logging", "tests/test_mlops.py"),
        ("Stage 7: FastAPI Endpoints & Integration", "tests/test_api.py"),
    ]

    for title, test_file in stages:
        run_stage(title, test_file)

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 7 STAGES PASSED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
