import copy
import torch
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model

PARAM_BYTES = 14 * 4

def aggregate_weights(client_weights, client_sizes):
    total_samples = sum(client_sizes)
    avg_sd = {}
    for key in client_weights[0].keys():
        avg_sd[key] = sum(
            client_weights[i][key].float() * (client_sizes[i] / total_samples)
            for i in range(len(client_weights))
        )
    return avg_sd

def train_client(global_model, x, y, epochs=1, lr=0.001, batch_size=4, mu=0.0):
    model = copy.deepcopy(global_model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_fn()
    indices = torch.randperm(len(x))

    for _ in range(epochs):
        for i in range(0, len(x), batch_size):
            b_idx = indices[i:i + batch_size]
            xb, yb = x[b_idx], y[b_idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if mu > 0.0:
                prox = sum((p - p_glob).pow(2).sum() for p, p_glob in zip(model.parameters(), global_model.parameters()))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
    return model.state_dict()

def run_federated(data, pooled_test=None, seeds=None, rounds=50, local_epochs=1, lr=0.001,
                  batch_size=4, mu=0.0, method="fedavg", eval_sites=None, exp="e1",
                  heldout_site="none", scenario="baseline"):
    if seeds is None:
        seeds = [42]
    loss_fn = get_loss_fn()
    train_ids = list(data.keys())
    eval_dict = eval_sites if eval_sites is not None else data
    records = []

    for seed in seeds:
        seed_everything(seed)
        global_model = Baseline()
        cumulative_bytes = 0

        for rnd in range(1, rounds + 1):
            client_weights = []
            client_sizes = []

            for cid in train_ids:
                x_tr, y_tr = data[cid]["train"]
                w_c = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size, mu=mu)
                client_weights.append(w_c)
                client_sizes.append(len(x_tr))

            cumulative_bytes += len(train_ids) * 2 * PARAM_BYTES
            new_weights = aggregate_weights(client_weights, client_sizes)
            global_model.load_state_dict(new_weights)

            for sid in eval_dict.keys():
                x_te, y_te = eval_dict[sid]["test"]
                acc, test_loss = evaluate_model(global_model, x_te, y_te, loss_fn)
                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": method,
                    "scenario": scenario, "site": sid, "heldout_site": heldout_site,
                    "seed": seed, "round": rnd, "sim_time_s": 0.0,
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": acc, "loss": test_loss
                })

            if pooled_test is not None:
                x_pte, y_pte = pooled_test
                all_acc, all_loss = evaluate_model(global_model, x_pte, y_pte, loss_fn)
                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": method,
                    "scenario": scenario, "site": "all", "heldout_site": heldout_site,
                    "seed": seed, "round": rnd, "sim_time_s": 0.0,
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss
                })

    return records
