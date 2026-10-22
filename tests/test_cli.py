import subprocess
import sys
from pathlib import Path


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Unified FL Hospital Consortium Experiment Runner" in result.stdout
    assert "--stage" in result.stdout


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
