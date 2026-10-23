import runpy
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(_ROOT / "ui" / "pages" / "2_Tariff_Explorer.py"), run_name="__main__")
