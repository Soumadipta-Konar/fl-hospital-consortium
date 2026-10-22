import numpy as np
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model
from src.fl import aggregate_weights, train_client, PARAM_BYTES

def sample_client_systems(client_ids, rng, heterogeneous_speed=False, heterogeneous_bw=False):
    systems = {}
    for cid in client_ids:
        speed = float(np.exp(rng.normal(0.0, 0.5))) if heterogeneous_speed else 1.0
        bw = float(rng.uniform(50.0, 200.0)) if heterogeneous_bw else 100.0
        systems[cid] = {"speed": speed, "bandwidth": bw}
    return systems

def simulate_round_timing(client_ids, data, systems, p_drop, deadline, epochs, base_time, rng):
    active_clients = []
    round_times = []
    round_bytes = 0

    for cid in client_ids:
        if p_drop > 0.0 and rng.random() < p_drop:
            continue
        n_samples = len(data[cid]["train"][0])
        t_comp = base_time * (n_samples / 100.0) * epochs / systems[cid]["speed"]
        down_bytes = PARAM_BYTES
        up_bytes = PARAM_BYTES
        t_comm = (down_bytes + up_bytes) / systems[cid]["bandwidth"]
        t_total = t_comp + t_comm
        round_bytes += down_bytes
        if t_total <= deadline:
            active_clients.append(cid)
            round_times.append(t_total)
            round_bytes += up_bytes

    sim_time = min(max(round_times), deadline) if round_times else deadline
    return active_clients, sim_time, round_bytes

def run_system_sim(data, pooled_test, seeds, rounds=50, local_epochs=1, lr=0.001,
                   batch_size=4, p_drop=0.0, deadline=1e6, base_time=1.0,
                   heterogeneous_speed=False, heterogeneous_bw=False, scenario="ideal"):
    loss_fn = get_loss_fn()
    x_pte, y_pte = pooled_test
    client_ids = list(data.keys())
    records = []

    for seed in seeds:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        systems = sample_client_systems(client_ids, rng, heterogeneous_speed, heterogeneous_bw)
        global_model = Baseline()
        cumulative_time = 0.0
        cumulative_bytes = 0

        for rnd in range(1, rounds + 1):
            active_ids, round_time, round_bytes = simulate_round_timing(
                client_ids, data, systems, p_drop, deadline, local_epochs, base_time, rng
            )
            cumulative_time += round_time
            cumulative_bytes += round_bytes

            if len(active_ids) > 0:
                client_weights = []
                client_sizes = []
                for cid in active_ids:
                    x_tr, y_tr = data[cid]["train"]
                    w_c = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)
                    client_weights.append(w_c)
                    client_sizes.append(len(x_tr))
                new_weights = aggregate_weights(client_weights, client_sizes)
                global_model.load_state_dict(new_weights)

            for cid in client_ids:
                x_te, y_te = data[cid]["test"]
                acc, test_loss = evaluate_model(global_model, x_te, y_te, loss_fn)
                records.append({
                    "exp": "e4", "dataset": "fed_heart_disease", "method": "fedavg",
                    "scenario": scenario, "site": cid, "heldout_site": "none",
                    "seed": seed, "round": rnd, "sim_time_s": round(cumulative_time, 2),
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(active_ids),
                    "metric_name": "accuracy", "metric_value": acc, "loss": test_loss
                })

            all_acc, all_loss = evaluate_model(global_model, x_pte, y_pte, loss_fn)
            records.append({
                "exp": "e4", "dataset": "fed_heart_disease", "method": "fedavg",
                "scenario": scenario, "site": "all", "heldout_site": "none",
                "seed": seed, "round": rnd, "sim_time_s": round(cumulative_time, 2),
                "bytes_sent": cumulative_bytes, "n_clients_active": len(active_ids),
                "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss
            })

    return records
