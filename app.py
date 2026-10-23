"""
BillGuard: Root Streamlit Entry Point.
Provides a redundant, cloud-compatible entry point that delegates to ui/app.py.
Both 'streamlit run ui/app.py' and 'streamlit run app.py' are supported.
"""

import os
import sys
import runpy
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Execute the main dashboard
_target = _ROOT / "ui" / "app.py"
runpy.run_path(str(_target), run_name="__main__")
