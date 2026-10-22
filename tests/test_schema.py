from pathlib import Path
import pandas as pd

EXPECTED_SCHEMA = [
    "exp", "dataset", "method", "scenario", "site", "heldout_site",
    "seed", "round", "sim_time_s", "bytes_sent", "n_clients_active",
    "metric_name", "metric_value", "loss"
]

def test_results_csv_schema():
    results_dir = Path("results")
    csv_files = sorted(results_dir.glob("e*.csv"))
    assert len(csv_files) >= 6, f"Expected at least 6 result CSVs, found {len(csv_files)}"

    for f in csv_files:
        df = pd.read_csv(f)
        assert list(df.columns) == EXPECTED_SCHEMA, f"File {f.name} does not match expected schema"
        assert not df["metric_value"].isna().any(), f"File {f.name} contains NaN metric values"
        assert not df["loss"].isna().any(), f"File {f.name} contains NaN loss values"
        assert (df["metric_value"] >= 0.0).all() and (df["metric_value"] <= 1.0).all(), (
            f"File {f.name} has metric values outside [0, 1]"
        )
