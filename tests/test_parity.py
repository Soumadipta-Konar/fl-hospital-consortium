import pandas as pd
from src.data import load_all_sites, load_pooled_data
from src.fl import run_federated
from src.privacy import run_privacy_fl

def test_dp_unperturbed_parity():
    sites_data = load_all_sites()
    pooled_test = load_pooled_data(train=False)

    fedavg_recs = run_federated(sites_data, pooled_test=pooled_test, seeds=[42], rounds=3, method="fedavg")
    dp_recs = run_privacy_fl(sites_data, pooled_test=pooled_test, seeds=[42], rounds=3, clip_norm=10000.0, noise_multiplier=0.0)

    df_fed = pd.DataFrame(fedavg_recs)
    df_dp = pd.DataFrame(dp_recs)

    max_diff = (df_fed["metric_value"] - df_dp["metric_value"]).abs().max()
    assert max_diff == 0.0, f"Unperturbed DP-FedAvg must match FedAvg exactly, got max diff {max_diff}"
