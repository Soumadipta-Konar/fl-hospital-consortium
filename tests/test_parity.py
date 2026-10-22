import pandas as pd
import torch
from src.fl import run_federated
from src.privacy import run_privacy_fl

def test_dp_unperturbed_parity():
    # Pure mathematical parity test on synthetic tensors (independent of network/disk)
    torch.manual_seed(42)
    mock_data = {}
    for sid in range(4):
        x_tr = torch.randn(20, 13)
        y_tr = torch.randint(0, 2, (20, 1)).float()
        x_te = torch.randn(10, 13)
        y_te = torch.randint(0, 2, (10, 1)).float()
        mock_data[sid] = {"train": (x_tr, y_tr), "test": (x_te, y_te)}
    mock_pooled_test = (torch.randn(20, 13), torch.randint(0, 2, (20, 1)).float())

    fedavg_recs = run_federated(mock_data, pooled_test=mock_pooled_test, seeds=[42], rounds=3, method="fedavg")
    dp_recs = run_privacy_fl(mock_data, pooled_test=mock_pooled_test, seeds=[42], rounds=3, clip_norm=10000.0, noise_multiplier=0.0)

    df_fed = pd.DataFrame(fedavg_recs)
    df_dp = pd.DataFrame(dp_recs)

    max_diff = (df_fed["metric_value"] - df_dp["metric_value"]).abs().max()
    assert max_diff == 0.0, f"Unperturbed DP-FedAvg must match FedAvg exactly, got max diff {max_diff}"
