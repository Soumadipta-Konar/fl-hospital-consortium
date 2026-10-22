import subprocess
import sys
from pathlib import Path


def test_dashboard_files_exist():
    root = Path(__file__).resolve().parent.parent
    dashboard_dir = root / "dashboard"
    assert dashboard_dir.is_dir()
    assert (dashboard_dir / "index.html").exists()
    assert (dashboard_dir / "style.css").exists()
    assert (dashboard_dir / "app.js").exists()


def test_dashboard_cli_help():
    result = subprocess.run(
        [sys.executable, "dashboard.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Launch FL Hospital Consortium Web Dashboard" in result.stdout
