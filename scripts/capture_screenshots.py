"""Capture genuine application screenshots from running Streamlit using Chrome CDP."""

import os
import time
import json
import base64
import subprocess
import urllib.request
import websocket
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = Path("./project_report/report_screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPTURES = [
    ("Figure 1: BillGuard Home Dashboard", "http://localhost:8501", "figure_1_home_dashboard.png", 5, 0),
    ("Figure 2: Electricity Bill Verification Interface", "http://localhost:8501/Bill_Verifier", "figure_2_bill_verifier.png", 6, 0),
    ("Figure 3: Tamil Nadu Tariff Explorer", "http://localhost:8501/Tariff_Explorer", "figure_3_tariff_explorer.png", 6, 0),
    ("Figure 4: Historical Consumption Analytics", "http://localhost:8501/Historical_Analytics", "figure_4_historical_analytics.png", 6, 0),
    ("Figure 5: MLOps Monitoring and Governance Center", "http://localhost:8501/MLOps_Center", "figure_5_mlops_center.png", 6, 0),
    ("Figure 6: Bill Investigation Results", "http://localhost:8501/Bill_Verifier", "figure_6_investigation_results.png", 7, 750),
]


import tempfile

def capture_screenshot(caption, url, filename, wait_sec=6, scroll_y=0):
    out_file = OUTPUT_DIR / filename
    temp_dir = os.path.join(tempfile.gettempdir(), f"chrome_snap_{int(time.time()*1000)}")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        f"--user-data-dir={temp_dir}",
        "--window-size=1440,960",
        url
    ]
    proc = subprocess.Popen(cmd)
    try:
        time.sleep(3.5)
        req = urllib.request.urlopen("http://127.0.0.1:9222/json")
        targets = json.loads(req.read().decode())
        page_target = next(t for t in targets if t.get("type") == "page")
        ws = websocket.create_connection(page_target["webSocketDebuggerUrl"], timeout=25)
        
        # Wait for Streamlit rendering
        time.sleep(wait_sec)
        
        # Optional scroll if needed to show investigation results
        if scroll_y > 0:
            scroll_expr = f"window.scrollTo(0, {scroll_y});"
            ws.send(json.dumps({"id": 50, "method": "Runtime.evaluate", "params": {"expression": scroll_expr}}))
            time.sleep(1.5)
        
        # Capture screenshot
        ws.send(json.dumps({"id": 100, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        while True:
            raw = ws.recv()
            msg = json.loads(raw)
            if msg.get("id") == 100:
                data = base64.b64decode(msg["result"]["data"])
                with open(out_file, "wb") as f:
                    f.write(data)
                print(f"[OK] {caption} -> {filename} ({len(data):,} bytes)")
                break
        ws.close()
    except Exception as e:
        print(f"[Error] Failed to capture {caption}: {e}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


def main():
    print("Capturing genuine application screenshots...")
    for caption, url, filename, wait_sec, scroll_y in CAPTURES:
        capture_screenshot(caption, url, filename, wait_sec, scroll_y)
    print("Screenshot capture complete!")

if __name__ == "__main__":
    main()
