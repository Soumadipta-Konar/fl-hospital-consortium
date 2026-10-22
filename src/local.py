import copy
import torch
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model

def train_epoch(model, x, y, optimizer, loss_fn, batch_size=4):
    model.train()
    indices = torch.randperm(len(x))
    total_loss = 0.0
    for i in range(0, len(x), batch_size):
        b_idx = indices[i:i + batch_size]
        xb, yb = x[b_idx], y[b_idx]
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(b_idx)
    return total_loss / len(x)

def run_local_only(data, seeds, epochs=50, lr=0.001, batch_size=4):
    loss_fn = get_loss_fn()
    records = []
    for seed in seeds:
        for site, site_data in data.items():
            seed_everything(seed)
            model = Baseline()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            x_tr, y_tr = site_data["train"]
            x_te, y_te = site_data["test"]

            for ep in range(1, epochs + 1):
                train_loss = train_epoch(model, x_tr, y_tr, optimizer, loss_fn, batch_size)
                acc, test_loss = evaluate_model(model, x_te, y_te, loss_fn)
                records.append({
                    "exp": "e1", "dataset": "fed_heart_disease", "method": "local-only",
                    "scenario": "baseline", "site": site, "heldout_site": "none",
                    "seed": seed, "round": ep, "sim_time_s": 0.0, "bytes_sent": 0,
                    "n_clients_active": 1, "metric_name": "accuracy",
                    "metric_value": acc, "loss": test_loss
                })
    return records

def run_pooled(data, pooled_train, pooled_test, seeds, epochs=50, lr=0.001, batch_size=4):
    loss_fn = get_loss_fn()
    x_ptr, y_ptr = pooled_train
    x_pte, y_pte = pooled_test
    records = []
    for seed in seeds:
        seed_everything(seed)
        model = Baseline()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            train_loss = train_epoch(model, x_ptr, y_ptr, optimizer, loss_fn, batch_size)
            for site, site_data in data.items():
                x_te, y_te = site_data["test"]
                acc, test_loss = evaluate_model(model, x_te, y_te, loss_fn)
                records.append({
                    "exp": "e1", "dataset": "fed_heart_disease", "method": "pooled",
                    "scenario": "baseline", "site": site, "heldout_site": "none",
                    "seed": seed, "round": ep, "sim_time_s": 0.0, "bytes_sent": 0,
                    "n_clients_active": 4, "metric_name": "accuracy",
                    "metric_value": acc, "loss": test_loss
                })
            all_acc, all_loss = evaluate_model(model, x_pte, y_pte, loss_fn)
            records.append({
                "exp": "e1", "dataset": "fed_heart_disease", "method": "pooled",
                "scenario": "baseline", "site": "all", "heldout_site": "none",
                "seed": seed, "round": ep, "sim_time_s": 0.0, "bytes_sent": 0,
                "n_clients_active": 4, "metric_name": "accuracy",
                "metric_value": all_acc, "loss": all_loss
            })
    return records
