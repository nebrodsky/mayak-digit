import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
