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
