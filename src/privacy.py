import torch
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model, PARAM_BYTES
from src.fl import train_client

def clip_update(delta, clip_norm):
    total_norm = torch.sqrt(sum((p.float() ** 2).sum() for p in delta.values())).item()
    if clip_norm is not None and total_norm > clip_norm:
        factor = clip_norm / (total_norm + 1e-12)
        clipped = {k: v.float() * factor for k, v in delta.items()}
        return clipped, total_norm
    return {k: v.float() for k, v in delta.items()}, total_norm

def dp_aggregate(w_glob, client_deltas, clip_norm=None, noise_multiplier=0.0, client_sizes=None):
    k_clients = len(client_deltas)
    clipped_deltas = []
    for d in client_deltas:
        cd, _ = clip_update(d, clip_norm)
        clipped_deltas.append(cd)

    new_sd = {}
    total_samples = sum(client_sizes) if client_sizes is not None else None

    for k in w_glob.keys():
        if client_sizes is not None:
            delta_sum = sum(clipped_deltas[i][k] * (client_sizes[i] / total_samples) for i in range(k_clients))
        else:
            delta_sum = sum(clipped_deltas[i][k] for i in range(k_clients)) / float(k_clients)

        if noise_multiplier > 0.0 and clip_norm is not None:
            sigma = (noise_multiplier * clip_norm) / float(k_clients)
            noise = torch.randn_like(delta_sum) * sigma
            delta_sum = delta_sum + noise

        new_sd[k] = w_glob[k].float() + delta_sum
    return new_sd

def aggregate_median(client_weights):
    med_sd = {}
    for k in client_weights[0].keys():
        stacked = torch.stack([w[k].float() for w in client_weights], dim=0)
        med_sd[k] = torch.median(stacked, dim=0).values
    return med_sd

def aggregate_trimmed_mean(client_weights, beta=1):
    n = len(client_weights)
    if 2 * beta >= n:
        return aggregate_median(client_weights)
    trimmed_sd = {}
    for k in client_weights[0].keys():
        stacked = torch.stack([w[k].float() for w in client_weights], dim=0)
        sorted_stacked, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_stacked[beta:n - beta]
        trimmed_sd[k] = trimmed.mean(dim=0)
    return trimmed_sd

def aggregate_krum(client_weights, f=1):
    """Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).

    THEORETICAL PRECONDITION NOTICE:
    Theorem 1 of Blanchard et al. (2017) guarantees convergence to a benign vector
    if and only if: n >= 2f + 3.
    For f=1 adversary, Krum strictly requires at least n = 5 clients.
    In a 4-hospital consortium (n=4), this theoretical precondition is formally
    violated (4 < 2(1) + 3 = 5). For 4-client cross-silo consortia with f=1,
    Coordinate-wise Median and Trimmed Mean (requiring n >= 2f + 1, so 4 >= 3)
    are the formally valid Byzantine defenses.
    """
    n = len(client_weights)
    if n <= 2:
        return aggregate_median(client_weights)
    flat_weights = [torch.cat([p.flatten().float() for p in w.values()]) for w in client_weights]
    flat_mat = torch.stack(flat_weights, dim=0)

    diffs = flat_mat.unsqueeze(1) - flat_mat.unsqueeze(0)
    dists = torch.sum(diffs ** 2, dim=-1)

    num_neighbors = max(1, n - f - 2)
    scores = []
    for i in range(n):
        other_dists = torch.cat([dists[i, :i], dists[i, i + 1:]])
        sorted_dists, _ = torch.sort(other_dists)
        score = sorted_dists[:num_neighbors].sum().item()
        scores.append(score)

    best_idx = int(torch.argmin(torch.tensor(scores)).item())
    return {k: client_weights[best_idx][k].clone().float() for k in client_weights[best_idx]}

def aggregate_multi_krum(client_weights, f=1, m=2):
    n = len(client_weights)
    if n <= 2:
        return aggregate_median(client_weights)
    flat_weights = [torch.cat([p.flatten().float() for p in w.values()]) for w in client_weights]
    flat_mat = torch.stack(flat_weights, dim=0)

    diffs = flat_mat.unsqueeze(1) - flat_mat.unsqueeze(0)
    dists = torch.sum(diffs ** 2, dim=-1)

    num_neighbors = max(1, n - f - 2)
    scores = []
    for i in range(n):
        other_dists = torch.cat([dists[i, :i], dists[i, i + 1:]])
        sorted_dists, _ = torch.sort(other_dists)
        score = sorted_dists[:num_neighbors].sum().item()
        scores.append(score)

    m = min(m, n)
    top_indices = torch.topk(torch.tensor(scores), k=m, largest=False).indices.tolist()

    avg_sd = {}
    for k in client_weights[0].keys():
        avg_sd[k] = torch.stack([client_weights[idx][k].float() for idx in top_indices], dim=0).mean(dim=0)
    return avg_sd

def poison_labels(y, q=1.0):
    if q <= 0.0:
        return y.clone()
    mask = (torch.rand(len(y)) < q).float().unsqueeze(1)
    return (1.0 - y) * mask + y * (1.0 - mask)

def run_privacy_fl(data, pooled_test, seeds, rounds=50, local_epochs=1, lr=0.001,
                   batch_size=4, clip_norm=None, noise_multiplier=0.0,
                   attack_type="none", attack_site=0, flip_q=1.0,
                   garbage_scale=1.0, use_median=False, aggregation_rule=None,
                   beta=1, krum_f=1, krum_m=2, method="dp_fedavg",
                   scenario="clean", exp="e6"):
    loss_fn = get_loss_fn()
    train_ids = list(data.keys())
    x_pte, y_pte = pooled_test
    records = []

    for seed in seeds:
        seed_everything(seed)
        global_model = Baseline()
        cumulative_bytes = 0

        for rnd in range(1, rounds + 1):
            w_glob = global_model.state_dict()
            client_deltas = []
            client_weights = []
            client_sizes = []

            for cid in train_ids:
                x_tr, y_tr = data[cid]["train"]
                if attack_type == "labelflip" and cid == attack_site:
                    y_tr = poison_labels(y_tr, q=flip_q)
                
                if attack_type == "garbage" and cid == attack_site:
                    w_c = {k: w_glob[k].float() + torch.randn_like(w_glob[k].float()) * garbage_scale for k in w_glob}
                else:
                    w_c = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)

                client_weights.append(w_c)
                client_sizes.append(len(x_tr))
                client_deltas.append({k: w_c[k].float() - w_glob[k].float() for k in w_glob})

            cumulative_bytes += len(train_ids) * 2 * PARAM_BYTES

            if aggregation_rule == "median" or use_median:
                new_weights = aggregate_median(client_weights)
            elif aggregation_rule == "trimmed_mean":
                new_weights = aggregate_trimmed_mean(client_weights, beta=beta)
            elif aggregation_rule == "krum":
                new_weights = aggregate_krum(client_weights, f=krum_f)
            elif aggregation_rule == "multi_krum":
                new_weights = aggregate_multi_krum(client_weights, f=krum_f, m=krum_m)
            else:
                new_weights = dp_aggregate(
                    w_glob, client_deltas, clip_norm=clip_norm,
                    noise_multiplier=noise_multiplier, client_sizes=client_sizes
                )

            global_model.load_state_dict(new_weights)

            for sid in train_ids:
                x_te, y_te = data[sid]["test"]
                acc, test_loss = evaluate_model(global_model, x_te, y_te, loss_fn)
                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": method,
                    "scenario": scenario, "site": sid, "heldout_site": "none",
                    "seed": seed, "round": rnd, "sim_time_s": 0.0,
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": acc, "loss": test_loss
                })

            all_acc, all_loss = evaluate_model(global_model, x_pte, y_pte, loss_fn)
            records.append({
                "exp": exp, "dataset": "fed_heart_disease", "method": method,
                "scenario": scenario, "site": "all", "heldout_site": "none",
                "seed": seed, "round": rnd, "sim_time_s": 0.0,
                "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss
            })

    return records
