from pathlib import Path
from src.plots import generate_all_plots

EXPECTED_PLOTS = [
    "hospital_baselines.png", "consortium_convergence.png", "leave_one_out_generalization.png",
    "hospital_contribution_matrix.png", "client_dropout_resilience.png",
    "straggler_deadlines_convergence.png", "straggler_deadlines_wallclock.png",
    "flower_deployment_parity.png", "differential_privacy_tradeoff.png", "byzantine_attacks_and_defense.png"
]

def test_plot_generation():
    plots_dir = Path("plots")
    generate_all_plots()

    for p in EXPECTED_PLOTS:
        target = plots_dir / p
        assert target.exists(), f"Plot {p} was not generated"
        assert target.stat().st_size > 1000, f"Plot {p} is empty or corrupt"
