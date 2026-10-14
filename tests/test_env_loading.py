import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_env_file_is_loaded_by_config(tmp_path):
    """Regression: .env.example told users to create .env, but nothing loaded it."""
    env_file = ROOT / ".env"
    existed = env_file.exists()
    backup = env_file.read_text(encoding="utf-8") if existed else None
    try:
        env_file.write_text("CDSE_CLIENT_ID=test-id-123\nCDSE_CLIENT_SECRET=test-secret\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("CDSE_")}
        out = subprocess.run([sys.executable, "-c",
                              "import os, app.config; print(os.getenv('CDSE_CLIENT_ID'))"],
                             cwd=ROOT, env=env, capture_output=True, text=True)
        assert out.stdout.strip() == "test-id-123", out.stderr
    finally:
        if existed:
            env_file.write_text(backup, encoding="utf-8")
        else:
            env_file.unlink(missing_ok=True)
