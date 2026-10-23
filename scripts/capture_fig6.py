"""Capture figure 6 with investigation results after clicking Audit & Screening."""

import os
import tempfile
import time
import json
import base64
import subprocess
import urllib.request
import websocket
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
out_file = Path("./project_report/report_screenshots/figure_6_investigation_results.png")
temp_dir = os.path.join(tempfile.gettempdir(), f"chrome_snap_fig6_{int(time.time()*1000)}")

cmd = [
    CHROME_PATH,
    "--headless=new",
    "--disable-gpu",
    "--remote-debugging-port=9222",
    "--remote-allow-origins=*",
    f"--user-data-dir={temp_dir}",
    "--window-size=1440,1100",
    "http://localhost:8501/Bill_Verifier"
]

proc = subprocess.Popen(cmd)
try:
    time.sleep(4.0)
    req = urllib.request.urlopen("http://127.0.0.1:9222/json")
    targets = json.loads(req.read().decode())
    page_target = next(t for t in targets if t.get("type") == "page")
    ws = websocket.create_connection(page_target["webSocketDebuggerUrl"], timeout=30)
    
    # Wait for page to hydrate
    time.sleep(6.0)
    
    # Click the primary submit button to trigger verification
    click_js = """
    (function() {
        let buttons = Array.from(document.querySelectorAll('button'));
        let targetBtn = buttons.find(b => b.innerText && (b.innerText.includes('Audit') || b.innerText.includes('Step 3')));
        if (targetBtn) {
            targetBtn.click();
            return 'CLICKED';
        }
        return 'NOT_FOUND';
    })();
    """
    ws.send(json.dumps({"id": 51, "method": "Runtime.evaluate", "params": {"expression": click_js}}))
    res = json.loads(ws.recv())
    print("Click result:", res)
    
    # Wait for Streamlit to compute audit and update DOM
    time.sleep(6.0)
    
    # Scroll down to investigation cards
    scroll_js = "window.scrollTo(0, 950);"
    ws.send(json.dumps({"id": 52, "method": "Runtime.evaluate", "params": {"expression": scroll_js}}))
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
            print(f"[OK] Figure 6 Investigation captured -> {out_file} ({len(data):,} bytes)")
            break
    ws.close()
except Exception as e:
    print(f"Error: {e}")
finally:
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        pass
