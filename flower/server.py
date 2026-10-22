import argparse
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model, PARAM_BYTES
from src.data import load_pooled_data, load_all_sites

def get_eval_fn(pooled_test, sites_data, output_csv, seed, rounds):
    x_pte, y_pte = pooled_test
    loss_fn = get_loss_fn()
    records = []

    def evaluate(server_round, parameters, config):
        model = Baseline()
        for p, w in zip(model.parameters(), parameters):
            p.data = torch.from_numpy(w)

        cumulative_bytes = server_round * 4 * 2 * PARAM_BYTES

        for sid, sdata in sites_data.items():
            x_te, y_te = sdata["test"]
            acc, test_loss = evaluate_model(model, x_te, y_te, loss_fn)
            records.append({
                "exp": "e5", "dataset": "fed_heart_disease", "method": "flower_fedavg",
                "scenario": "flower_docker", "site": sid, "heldout_site": "none",
                "seed": seed, "round": server_round, "sim_time_s": 0.0,
                "bytes_sent": cumulative_bytes, "n_clients_active": 4,
                "metric_name": "accuracy", "metric_value": acc, "loss": test_loss
            })

        all_acc, all_loss = evaluate_model(model, x_pte, y_pte, loss_fn)
        records.append({
            "exp": "e5", "dataset": "fed_heart_disease", "method": "flower_fedavg",
            "scenario": "flower_docker", "site": "all", "heldout_site": "none",
            "seed": seed, "round": server_round, "sim_time_s": 0.0,
            "bytes_sent": cumulative_bytes, "n_clients_active": 4,
            "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss
        })

        if server_round == rounds and output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            df = pd.DataFrame(records)
            # Append if exists, else write header
            if os.path.exists(output_csv):
                df.to_csv(output_csv, mode="a", header=False, index=False)
            else:
                df.to_csv(output_csv, index=False)

        return all_loss, {"accuracy": all_acc}

    return evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--output-csv", type=str, default="results/e5.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    initial_model = Baseline()
    initial_params = ndarrays_to_parameters([p.detach().cpu().numpy() for p in initial_model.parameters()])

    pooled_test = load_pooled_data(train=False)
    sites_data = load_all_sites()

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_fn=get_eval_fn(pooled_test, sites_data, args.output_csv, args.seed, args.rounds),
        initial_parameters=initial_params,
    )

    print(f"Flower server starting on {args.server_address} for {args.rounds} rounds (seed={args.seed})...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
