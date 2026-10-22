from typing import Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np

PARAM_BYTES = 14 * 4

class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def get_loss_fn():
    return nn.BCELoss()

def compute_accuracy(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy().astype(np.uint8).ravel()
    else:
        y_true_np = np.asarray(y_true, dtype=np.uint8).ravel()

    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy().ravel()
    else:
        y_pred_np = np.asarray(y_pred).ravel()

    return float(((y_pred_np > 0.5) == y_true_np).mean())

def compute_bootstrap_ci(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes non-parametric percentile bootstrap confidence intervals for accuracy.

    Crucial for small clinical cohorts (e.g., Switzerland N=16, VA Long Beach N=45)
    where seed variance only captures training stochasticity and fails to account
    for test-set sampling noise.

    Returns:
        dict: {
            "mean": float,
            "ci_lower": float,
            "ci_upper": float,
            "std_error": float,
            "n_samples": int,
        }
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy().astype(np.uint8).ravel()
    else:
        y_true_np = np.asarray(y_true, dtype=np.uint8).ravel()

    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy().ravel()
    else:
        y_pred_np = np.asarray(y_pred).ravel()

    binary_preds = (y_pred_np > 0.5).astype(np.uint8)
    correct = (binary_preds == y_true_np).astype(np.float64)
    n = len(correct)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_error": 0.0, "n_samples": 0}

    point_acc = float(correct.mean())
    rng = np.random.default_rng(seed)
    boot_accs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        boot_accs[b] = correct[boot_idx].mean()

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_accs, lower_pct))
    ci_upper = float(np.percentile(boot_accs, upper_pct))
    std_error = float(np.std(boot_accs))

    return {
        "mean": point_acc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_samples": n,
    }

def evaluate_model(model, x, y, loss_fn=None):
    if loss_fn is None:
        loss_fn = get_loss_fn()
    model.eval()
    with torch.no_grad():
        preds = model(x)
        loss = float(loss_fn(preds, y).item())
        acc = compute_accuracy(y, preds)
    return acc, loss
