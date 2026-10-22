import heapq
import numpy as np
from src.seed import seed_everything
from src.models import Baseline, get_loss_fn, evaluate_model, PARAM_BYTES
from src.fl import train_client
from src.system import sample_client_systems


def staleness_polynomial(tau, alpha=0.5):
    """Polynomial staleness decay: s(tau) = (1 + tau)^(-alpha)."""
    return float((1.0 + max(0, tau)) ** (-alpha))


def staleness_hinge(tau, a=1.0, b=4):
    """Hinge staleness function with plateau for small delays."""
    if tau <= b:
        return 1.0
    return float(1.0 / (1.0 + a * (tau - b)))


def staleness_constant(tau):
    """Constant staleness function (no damping)."""
    return 1.0


def compute_staleness_weight(tau, func="poly", poly_alpha=0.5, hinge_a=1.0, hinge_b=4):
    if func == "poly":
        return staleness_polynomial(tau, alpha=poly_alpha)
    if func == "hinge":
        return staleness_hinge(tau, a=hinge_a, b=hinge_b)
    return staleness_constant(tau)


def async_server_update(w_server, w_client, staleness_weight, alpha=0.5):
    """Applies FedAsync server model update with staleness damping."""
    effective_alpha = max(0.001, min(1.0, alpha * staleness_weight))
    new_weights = {}
    for k in w_server.keys():
        new_weights[k] = (1.0 - effective_alpha) * w_server[k].float() + effective_alpha * w_client[k].float()
    return new_weights


def compute_client_cycle_time(data, systems, cid, epochs, base_time, rng):
    n_samples = len(data[cid]["train"][0])
    speed = systems[cid]["speed"]
    bw = systems[cid]["bandwidth"]
    t_comp = base_time * (n_samples / 100.0) * epochs / speed
    t_comm = (PARAM_BYTES * 2) / bw
    # Small jitter (5%) to model network/compute variance
    jitter = float(np.exp(rng.normal(0.0, 0.05)))
    return (t_comp + t_comm) * jitter


def run_async_fl(data, pooled_test, seeds, total_updates=100, local_epochs=1, lr=0.001,
                 batch_size=4, alpha=0.5, staleness_func="poly", poly_alpha=0.5,
                 base_time=1.0, heterogeneous_speed=True, heterogeneous_bw=True,
                 eval_interval=10, scenario="async_fedasync", exp="e8"):
    """Event-driven Asynchronous Federated Learning (FedAsync) simulation across hospitals."""
    loss_fn = get_loss_fn()
    train_ids = list(data.keys())
    x_pte, y_pte = pooled_test
    records = []

    for seed in seeds:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        systems = sample_client_systems(train_ids, rng, heterogeneous_speed, heterogeneous_bw)
        global_model = Baseline()

        server_clock = 0
        cumulative_bytes = 0
        event_queue = []  # Heap of (t_finish, event_id, client_id, start_server_clock, client_snapshot)

        # Initialize: all hospitals start training asynchronously from round 0
        event_counter = 0
        for cid in train_ids:
            x_tr, y_tr = data[cid]["train"]
            w_c = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)
            dt = compute_client_cycle_time(data, systems, cid, local_epochs, base_time, rng)
            heapq.heappush(event_queue, (dt, event_counter, cid, 0, w_c))
            event_counter += 1

        update_count = 0
        while update_count < total_updates and event_queue:
            t_finish, _, cid, start_clock, w_c = heapq.heappop(event_queue)
            update_count += 1
            cumulative_bytes += 2 * PARAM_BYTES

            # Calculate staleness relative to server model version
            tau = server_clock - start_clock
            s_weight = compute_staleness_weight(tau, func=staleness_func, poly_alpha=poly_alpha)

            # Update global server model
            w_server = global_model.state_dict()
            new_weights = async_server_update(w_server, w_c, s_weight, alpha=alpha)
            global_model.load_state_dict(new_weights)
            server_clock += 1

            # Client immediately begins its next local training round with updated global model
            x_tr, y_tr = data[cid]["train"]
            w_next = train_client(global_model, x_tr, y_tr, epochs=local_epochs, lr=lr, batch_size=batch_size)
            dt = compute_client_cycle_time(data, systems, cid, local_epochs, base_time, rng)
            heapq.heappush(event_queue, (t_finish + dt, event_counter, cid, server_clock, w_next))
            event_counter += 1

            # Periodic evaluation
            if update_count % eval_interval == 0 or update_count == total_updates:
                for sid in train_ids:
                    x_te, y_te = data[sid]["test"]
                    acc, test_loss = evaluate_model(global_model, x_te, y_te, loss_fn)
                    records.append({
                        "exp": exp, "dataset": "fed_heart_disease", "method": "fedasync",
                        "scenario": scenario, "site": sid, "heldout_site": "none",
                        "seed": seed, "round": update_count, "sim_time_s": round(t_finish, 2),
                        "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                        "metric_name": "accuracy", "metric_value": acc, "loss": test_loss,
                    })

                all_acc, all_loss = evaluate_model(global_model, x_pte, y_pte, loss_fn)
                records.append({
                    "exp": exp, "dataset": "fed_heart_disease", "method": "fedasync",
                    "scenario": scenario, "site": "all", "heldout_site": "none",
                    "seed": seed, "round": update_count, "sim_time_s": round(t_finish, 2),
                    "bytes_sent": cumulative_bytes, "n_clients_active": len(train_ids),
                    "metric_name": "accuracy", "metric_value": all_acc, "loss": all_loss,
                })

    return records
