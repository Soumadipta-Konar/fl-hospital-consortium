import torch
from src.models import Baseline, get_loss_fn, compute_accuracy, evaluate_model, PARAM_BYTES

def test_model_architecture():
    model = Baseline(input_dim=13, output_dim=1)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 14, f"Baseline model should have 14 parameters, got {total_params}"
    assert PARAM_BYTES == 56, f"Expected 56 bytes, got {PARAM_BYTES}"

def test_model_forward():
    model = Baseline(input_dim=13, output_dim=1)
    x = torch.randn(8, 13)
    y_pred = model(x)
    assert y_pred.shape == (8, 1)
    assert (y_pred >= 0.0).all() and (y_pred <= 1.0).all(), "Sigmoid output must be in [0, 1]"

def test_evaluation_functions():
    model = Baseline(input_dim=13, output_dim=1)
    loss_fn = get_loss_fn()
    x = torch.randn(10, 13)
    y = torch.randint(0, 2, (10, 1)).float()

    acc, loss = evaluate_model(model, x, y, loss_fn)
    assert 0.0 <= acc <= 1.0, f"Accuracy must be between 0 and 1, got {acc}"
    assert loss >= 0.0, f"BCE loss must be non-negative, got {loss}"

    raw_acc = compute_accuracy(y, model(x))
    assert 0.0 <= raw_acc <= 1.0

def test_compute_bootstrap_ci():
    from src.models import compute_bootstrap_ci
    import numpy as np

    # Synthetic small cohort (e.g., Switzerland N=16)
    y_true = np.array([1] * 15 + [0], dtype=np.uint8)
    y_pred = np.array([0.9] * 15 + [0.1], dtype=np.float32)  # 100% correct

    res = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=500, ci=0.95, seed=42)
    assert res["n_samples"] == 16
    assert res["mean"] == 1.0
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]

    # Imbalanced cohort with errors (e.g., VA N=45)
    y_true_va = np.array([1] * 35 + [0] * 10)
    y_pred_va = np.array([0.8] * 25 + [0.2] * 10 + [0.8] * 5 + [0.2] * 5)
    res_va = compute_bootstrap_ci(y_true_va, y_pred_va, n_bootstraps=500, ci=0.95, seed=42)
    assert res_va["n_samples"] == 45
    assert 0.0 <= res_va["ci_lower"] <= res_va["mean"] <= res_va["ci_upper"] <= 1.0
    assert res_va["std_error"] > 0.0

    # Determinism
    res_repeat = compute_bootstrap_ci(y_true_va, y_pred_va, n_bootstraps=500, ci=0.95, seed=42)
    assert res_va["ci_lower"] == res_repeat["ci_lower"]
    assert res_va["ci_upper"] == res_repeat["ci_upper"]
