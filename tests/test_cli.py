import importlib.util
import subprocess
import sys
from pathlib import Path
import pytest

HAS_FLAMBY = importlib.util.find_spec("flamby") is not None
DATA_DIR = Path("data/fed_heart_disease")
HAS_DATA = DATA_DIR.exists() and len(list(DATA_DIR.glob("*.data"))) == 4


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Unified FL Hospital Consortium Experiment Runner" in result.stdout
    assert "--stage" in result.stdout


@pytest.mark.skipif(not (HAS_FLAMBY and HAS_DATA), reason="FLamby library or raw dataset not present on disk")
def test_cli_summary():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--stage", "summary"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "cleveland" in result.stdout.lower()
    assert "hungarian" in result.stdout.lower()


def test_cli_budget(tmp_path):
    out_dir = str(tmp_path / "results")
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--stage", "budget", "--rounds", "5", "--output-dir", out_dir],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (Path(out_dir) / "privacy_budget.csv").exists()


def test_cli_plots():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--stage", "plots"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "[OK] Figures generated in plots/" in result.stdout


def test_cli_flower():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--stage", "flower"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Stage 5: Multi-Process Flower Deployment Parity" in result.stdout
