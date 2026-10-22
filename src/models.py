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
    y_true_np = y_true.detach().cpu().numpy().astype(np.uint8).ravel()
    y_pred_np = y_pred.detach().cpu().numpy().ravel()
    return float(((y_pred_np > 0.5) == y_true_np).mean())

def evaluate_model(model, x, y, loss_fn=None):
    if loss_fn is None:
        loss_fn = get_loss_fn()
    model.eval()
    with torch.no_grad():
        preds = model(x)
        loss = float(loss_fn(preds, y).item())
        acc = compute_accuracy(y, preds)
    return acc, loss
