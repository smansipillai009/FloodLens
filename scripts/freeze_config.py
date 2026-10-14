"""Freeze the config BEFORE opening any check-only data (EMSR927).
Writes configs/config.lock = "<sha256>  <UTC time>  <git commit or 'no-git'>".
Commit this file. validation_only/compare_emsr927.py refuses to run if the config changed since."""
import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import LOCK_PATH, config_hash  # noqa: E402

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:  # noqa: BLE001
    commit = "no-git"
LOCK_PATH.write_text(f"{config_hash()}  {datetime.datetime.now(datetime.timezone.utc).isoformat()}  {commit}\n", encoding="utf-8")
print("Config frozen:", LOCK_PATH.read_text(encoding="utf-8").strip(), "\nNow `git add configs/config.lock && git commit`.")
