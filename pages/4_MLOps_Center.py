import runpy
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(_ROOT / "ui" / "pages" / "4_MLOps_Center.py"), run_name="__main__")
