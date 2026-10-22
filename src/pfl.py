import copy
import torch
import torch.nn as nn
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model, PARAM_BYTES
from src.fl import aggregate_weights, train_client


class PersonalizedMLP(nn.Module):
    """Personalized architecture with shared base representations and private local heads."""

    def __init__(self, input_dim=13, hidden_dim=8, output_dim=1):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.base(x)
        return torch.sigmoid(self.head(features))


def train_personalized_ditto(pers_model, glob_model, x, y, epochs=1, lr=0.001, batch_size=4, lam=0.1):
    """Updates client personalized model with proximal regularization towards global consensus model."""
    pers_model.train()
    optimizer = torch.optim.Adam(pers_model.parameters(), lr=lr)
    loss_fn = get_loss_fn()
    indices = torch.randperm(len(x))

    for _ in range(epochs):
        for i in range(0, len(x), batch_size):
            b_idx = indices[i:i + batch_size]
            xb, yb = x[b_idx], y[b_idx]
            optimizer.zero_grad()
            pred = pers_model(xb)
            loss = loss_fn(pred, yb)
            # Ditto proximal term: (lambda / 2) * ||v_k - w_glob||^2
            prox = sum((pv - pw.detach()).pow(2).sum() for pv, pw in zip(pers_model.parameters(), glob_model.parameters()))
            total_loss = loss + (lam / 2.0) * prox
            total_loss.backward()
            optimizer.step()

    return pers_model.state_dict()


def train_fedper_client(client_model, x, y, epochs=1, lr=0.001, batch_size=4):
    """Trains FedPer client: joint training of shared base and local head."""
    client_model.train()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    loss_fn = get_loss_fn()
    indices = torch.randperm(len(x))

    for _ in range(epochs):
        for i in range(0, len(x), batch_size):
            b_idx = indices[i:i + batch_size]
            xb, yb = x[b_idx], y[b_idx]
            optimizer.zero_grad()
            loss = loss_fn(client_model(xb), yb)
            loss.backward()
            optimizer.step()

    return client_model.state_dict()


def run_pfl(data, pooled_test, seeds, rounds=50, local_epochs=1, lr=0.001,
            batch_size=4, method="ditto", lam=0.1, ft_epochs=3,
            scenario="personalized", exp="e7"):
    """Orchestrates Personalized Federated Learning experiments across hospitals."""
    loss_fn = get_loss_fn()
    train_ids = list(data.keys())
    x_pte, y_pte = pooled_test
    records = []

    for seed in seeds:
        seed_everything(seed)
        cumulative_bytes = 0

        if method == "fedper":
            client_models = {cid: PersonalizedMLP() for cid in train_ids}
            global_base_keys = [k for k in client_models[train_ids[0]].state_dict().keys() if k.startswith("base.")]

            for rnd in range(1, rounds + 1):
                client_base_weights = []
                client_sizes = []

                for cid in train_ids:
                    x_tr, y_tr = data[cid]["train"]
                    sd = train_fedper_client(client_models[cid], x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)
                    client_models[cid].load_state_dict(sd)
                    # Extract only base layers for global federation
                    base_sd = {k: sd[k] for k in global_base_keys}
                    client_base_weights.append(base_sd)
                    client_sizes.append(len(x_tr))

                # Server aggregates only representation base layers
                cumulative_bytes += len(train_ids) * 2 * (13 * 8 + 8) * 4
                agg_base = aggregate_weights(client_base_weights, client_sizes)

                # Broadcast shared base back to clients, keeping heads local
                for cid in train_ids:
                    c_sd = client_models[cid].state_dict()
                    for k in global_base_keys:
                        c_sd[k] = agg_base[k]
                    client_models[cid].load_state_dict(c_sd)

                    x_te, y_te = data[cid]["test"]
                    acc, test_loss = evaluate_model(client_models[cid], x_te, y_te, loss_fn)
                    records.append({
                        "exp": exp, "dataset": "fed_heart_disease", "method": method,
                        "scenario": scenario, "site": cid, "heldout_site": "none",
                        "seed": seed, "round": rnd, "sim_time_s": 0.0,
                        "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                        "metric_name": "accuracy", "metric_value": acc, "loss": test_loss,
                    })

                # Evaluated on pooled test by averaging client predictions
                with torch.no_grad():
                    all_preds = torch.stack([client_models[cid](x_pte) for cid in train_ids], dim=0).mean(dim=0)
                    all_acc = float(((all_preds.numpy().ravel() > 0.5) == y_pte.numpy().ravel()).mean())
                    all_loss = float(loss_fn(all_preds, y_pte).item())

                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": method,
                    "scenario": scenario, "site": "all", "heldout_site": "none",
                    "seed": seed, "round": rnd, "sim_time_s": 0.0,
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss,
                })

        elif method == "ditto":
            global_model = Baseline()
            pers_models = {cid: Baseline() for cid in train_ids}

            for rnd in range(1, rounds + 1):
                client_weights = []
                client_sizes = []

                # 1. Standard global model training step
                for cid in train_ids:
                    x_tr, y_tr = data[cid]["train"]
                    w_c = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)
                    client_weights.append(w_c)
                    client_sizes.append(len(x_tr))

                    # 2. Personalized model step towards global consensus
                    p_sd = train_personalized_ditto(
                        pers_models[cid], global_model, x_tr, y_tr,
                        epochs=local_epochs, lr=lr, batch_size=batch_size, lam=lam,
                    )
                    pers_models[cid].load_state_dict(p_sd)

                cumulative_bytes += len(train_ids) * 2 * PARAM_BYTES
                new_weights = aggregate_weights(client_weights, client_sizes)
                global_model.load_state_dict(new_weights)

                # Evaluate personalized models locally
                for cid in train_ids:
                    x_te, y_te = data[cid]["test"]
                    acc, test_loss = evaluate_model(pers_models[cid], x_te, y_te, loss_fn)
                    records.append({
                        "exp": exp, "dataset": "fed_heart_disease", "method": method,
                        "scenario": scenario, "site": cid, "heldout_site": "none",
                        "seed": seed, "round": rnd, "sim_time_s": 0.0,
                        "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                        "metric_name": "accuracy", "metric_value": acc, "loss": test_loss,
                    })

                all_acc, all_loss = evaluate_model(global_model, x_pte, y_pte, loss_fn)
                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": method,
                    "scenario": scenario, "site": "all", "heldout_site": "none",
                    "seed": seed, "round": rnd, "sim_time_s": 0.0,
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss,
                })

        elif method == "finetune":
            global_model = Baseline()
            for rnd in range(1, rounds + 1):
                client_weights = []
                client_sizes = []
                for cid in train_ids:
                    x_tr, y_tr = data[cid]["train"]
                    w_c = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)
                    client_weights.append(w_c)
                    client_sizes.append(len(x_tr))

                cumulative_bytes += len(train_ids) * 2 * PARAM_BYTES
                new_weights = aggregate_weights(client_weights, client_sizes)
                global_model.load_state_dict(new_weights)

                # Local fine-tuning on copy of global model
                for cid in train_ids:
                    x_tr, y_tr = data[cid]["train"]
                    x_te, y_te = data[cid]["test"]
                    local_ft_model = copy.deepcopy(global_model)
                    ft_sd = train_client(local_ft_model, x_tr, y_tr, epochs=ft_epochs, lr=lr * 0.5, batch_size=batch_size)
                    local_ft_model.load_state_dict(ft_sd)
                    acc, test_loss = evaluate_model(local_ft_model, x_te, y_te, loss_fn)
                    records.append({
                        "exp": exp, "dataset": "fed_heart_disease", "method": method,
                        "scenario": scenario, "site": cid, "heldout_site": "none",
                        "seed": seed, "round": rnd, "sim_time_s": 0.0,
                        "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                        "metric_name": "accuracy", "metric_value": acc, "loss": test_loss,
                    })

                all_acc, all_loss = evaluate_model(global_model, x_pte, y_pte, loss_fn)
                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": method,
                    "scenario": scenario, "site": "all", "heldout_site": "none",
                    "seed": seed, "round": rnd, "sim_time_s": 0.0,
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss,
                })

    return records
