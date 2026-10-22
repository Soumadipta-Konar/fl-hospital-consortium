import argparse
import os
import sys
from pathlib import Path
import pandas as pd

from src.data import load_all_sites, load_pooled_data, get_site_summary
from src.local import run_local_only, run_pooled
from src.fl import run_federated
from src.system import run_system_sim
from src.privacy import run_privacy_fl
from src.pfl import run_pfl
from src.async_fl import run_async_fl
from src.privacy_accountant import compute_privacy_curve, validate_clinical_privacy
from src.plots import generate_all_plots


def save_records(records, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Written {len(df)} records -> {out_csv}")


def run_stage_baselines(data, pooled_tr, pooled_te, seeds, rounds, out_dir):
    print(">>> Running Stage 2: Baselines (Local-only, Pooled, FedAvg, FedProx)...")
    recs = []
    recs.extend(run_local_only(data, seeds=seeds, epochs=rounds))
    recs.extend(run_pooled(data, pooled_tr, pooled_te, seeds=seeds, epochs=rounds))
    recs.extend(run_federated(data, pooled_test=pooled_te, seeds=seeds, rounds=rounds, method="fedavg", scenario="baseline"))
    recs.extend(run_federated(data, pooled_test=pooled_te, seeds=seeds, rounds=rounds, mu=0.01, method="fedprox", scenario="baseline"))
    save_records(recs, os.path.join(out_dir, "e1.csv"))


def run_stage_loso(data, pooled_te, seeds, rounds, out_dir):
    print(">>> Running Stage 3: Leave-One-Site-Out (LOSO)...")
    recs = []
    client_ids = list(data.keys())
    for heldout in client_ids:
        train_data = {cid: data[cid] for cid in client_ids if cid != heldout}
        r = run_federated(
            train_data, pooled_test=pooled_te, seeds=seeds, rounds=rounds,
            eval_sites=data, heldout_site=heldout, scenario="loso", exp="e2",
        )
        recs.extend(r)
    save_records(recs, os.path.join(out_dir, "e2.csv"))


def run_stage_contribution(data, pooled_te, seeds, rounds, out_dir):
    print(">>> Running Stage 3: Site Contribution Analysis...")
    recs = []
    client_ids = list(data.keys())
    # Full consortium
    r_full = run_federated(
        data, pooled_test=pooled_te, seeds=seeds, rounds=rounds,
        eval_sites=data, heldout_site="none", scenario="full", exp="e3",
    )
    recs.extend(r_full)
    # Leave each hospital out
    for heldout in client_ids:
        train_data = {cid: data[cid] for cid in client_ids if cid != heldout}
        r_ex = run_federated(
            train_data, pooled_test=pooled_te, seeds=seeds, rounds=rounds,
            eval_sites=data, heldout_site=heldout, scenario=f"without_{heldout}", exp="e3",
        )
        recs.extend(r_ex)
    save_records(recs, os.path.join(out_dir, "e3.csv"))


def run_stage_system(data, pooled_te, seeds, rounds, out_dir):
    print(">>> Running Stage 4: Systems Realism (Dropouts & Straggler Deadlines)...")
    recs = []
    recs.extend(run_system_sim(data, pooled_te, seeds=seeds, rounds=rounds, p_drop=0.0, deadline=1e6, scenario="ideal"))
    recs.extend(run_system_sim(data, pooled_te, seeds=seeds, rounds=rounds, p_drop=0.2, deadline=1e6, scenario="drop0.2"))
    recs.extend(run_system_sim(data, pooled_te, seeds=seeds, rounds=rounds, p_drop=0.5, deadline=1e6, scenario="drop0.5"))
    recs.extend(run_system_sim(data, pooled_te, seeds=seeds, rounds=rounds, deadline=2.0, heterogeneous_speed=True, scenario="deadline_2.0s"))
    recs.extend(run_system_sim(data, pooled_te, seeds=seeds, rounds=rounds, deadline=1.0, heterogeneous_speed=True, scenario="deadline_1.0s"))
    save_records(recs, os.path.join(out_dir, "e4.csv"))


def run_stage_privacy_and_defenses(data, pooled_te, seeds, rounds, out_dir):
    print(">>> Running Stage 6 & 8: Privacy & Byzantine Defenses...")
    recs = []
    # Clean baseline
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, clip_norm=10000.0, noise_multiplier=0.0, scenario="dp_z0.0"))
    # Label flip under FedAvg vs Median vs Trimmed Mean vs Krum
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="labelflip", attack_site=0, flip_q=0.5, scenario="labelflip_q0.5"))
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="labelflip", attack_site=0, flip_q=0.5, use_median=True, scenario="labelflip_median"))
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="labelflip", attack_site=0, flip_q=0.5, aggregation_rule="trimmed_mean", beta=1, scenario="labelflip_trimmed_mean"))
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="labelflip", attack_site=0, flip_q=0.5, aggregation_rule="krum", krum_f=1, scenario="labelflip_krum"))
    # Garbage updates
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="garbage", attack_site=0, garbage_scale=1.0, scenario="garbage_fedavg"))
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="garbage", attack_site=0, garbage_scale=1.0, use_median=True, scenario="garbage_median"))
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="garbage", attack_site=0, garbage_scale=1.0, aggregation_rule="trimmed_mean", beta=1, scenario="garbage_trimmed_mean"))
    recs.extend(run_privacy_fl(data, pooled_te, seeds=seeds, rounds=rounds, attack_type="garbage", attack_site=0, garbage_scale=1.0, aggregation_rule="krum", krum_f=1, scenario="garbage_krum"))
    save_records(recs, os.path.join(out_dir, "e6.csv"))


def run_stage_pfl(data, pooled_te, seeds, rounds, out_dir):
    print(">>> Running Personalized FL: Ditto, FedPer, and Fine-Tuning...")
    recs = []
    recs.extend(run_pfl(data, pooled_te, seeds=seeds, rounds=rounds, method="ditto", lam=0.1, scenario="ditto_lam0.1"))
    recs.extend(run_pfl(data, pooled_te, seeds=seeds, rounds=rounds, method="fedper", scenario="fedper_shared_base"))
    recs.extend(run_pfl(data, pooled_te, seeds=seeds, rounds=rounds, method="finetune", ft_epochs=2, scenario="fedavg_finetune"))
    save_records(recs, os.path.join(out_dir, "e7.csv"))


def run_stage_async(data, pooled_te, seeds, updates, out_dir):
    print(">>> Running Asynchronous FL: FedAsync with Staleness Damping...")
    recs = []
    recs.extend(run_async_fl(data, pooled_te, seeds=seeds, total_updates=updates, staleness_func="poly", poly_alpha=0.5, scenario="fedasync_poly"))
    recs.extend(run_async_fl(data, pooled_te, seeds=seeds, total_updates=updates, staleness_func="hinge", scenario="fedasync_hinge"))
    recs.extend(run_async_fl(data, pooled_te, seeds=seeds, total_updates=updates, staleness_func="const", scenario="fedasync_constant"))
    save_records(recs, os.path.join(out_dir, "e8.csv"))


def run_stage_privacy_budget(out_dir, max_rounds=50, noise_multiplier=1.0):
    print(">>> Running Privacy Accountant: Calculating (epsilon, delta) curves...")
    curve = compute_privacy_curve(noise_multiplier=noise_multiplier, max_rounds=max_rounds)
    out_csv = os.path.join(out_dir, "privacy_budget.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df = pd.DataFrame(curve)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Written {len(df)} privacy budget rows -> {out_csv}")
    final_eps = curve[-1]["epsilon"]
    eval_res = validate_clinical_privacy(final_eps)
    print(f"    Target HIPAA/GDPR Compliance: {eval_res['tier'].upper()} (eps={final_eps:.2f}, delta=1e-5)")


def main():
    parser = argparse.ArgumentParser(description="Unified FL Hospital Consortium Experiment Runner")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "baselines", "loso", "contribution", "system", "privacy", "pfl", "async", "budget", "plots", "summary"],
        help="Experiment stage to execute",
    )
    parser.add_argument("--rounds", type=int, default=50, help="Number of communication rounds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds list")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSV results")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to save figures")
    args = parser.parse_args()

    print("=" * 60)
    print("FL Hospital Consortium - Experiment Runner")
    print(f"Stage: {args.stage} | Rounds: {args.rounds} | Seeds: {args.seeds}")
    print("=" * 60)

    if args.stage == "plots":
        print(">>> Regenerating publication figures from results/...")
        generate_all_plots()
        print(f"[OK] Figures generated in {args.plots_dir}/")
        return

    if args.stage == "budget":
        run_stage_privacy_budget(args.output_dir, max_rounds=args.rounds)
        return

    # Check for FLamby dataset
    data_dir = Path("data/fed_heart_disease")
    has_data = data_dir.exists() and len(list(data_dir.glob("*.data"))) == 4

    if not has_data:
        print("[!] FLamby dataset files not found in data/fed_heart_disease.")
        print("    Run: python -m flamby.datasets.fed_heart_disease.dataset_creation_scripts.download --output-folder data/fed_heart_disease")
        sys.exit(1)

    if args.stage == "summary":
        summary_df = get_site_summary()
        print(summary_df.to_string(index=False))
        return

    data = load_all_sites()
    pooled_tr = load_pooled_data(train=True)
    pooled_te = load_pooled_data(train=False)

    if args.stage in ["baselines", "all"]:
        run_stage_baselines(data, pooled_tr, pooled_te, args.seeds, args.rounds, args.output_dir)

    if args.stage in ["loso", "all"]:
        run_stage_loso(data, pooled_te, args.seeds, args.rounds, args.output_dir)

    if args.stage in ["contribution", "all"]:
        run_stage_contribution(data, pooled_te, args.seeds, args.rounds, args.output_dir)

    if args.stage in ["system", "all"]:
        run_stage_system(data, pooled_te, args.seeds, args.rounds, args.output_dir)

    if args.stage in ["privacy", "all"]:
        run_stage_privacy_and_defenses(data, pooled_te, args.seeds, args.rounds, args.output_dir)

    if args.stage in ["pfl", "all"]:
        run_stage_pfl(data, pooled_te, args.seeds, args.rounds, args.output_dir)

    if args.stage in ["async", "all"]:
        run_stage_async(data, pooled_te, args.seeds, updates=args.rounds * 2, out_dir=args.output_dir)

    if args.stage in ["budget", "all"]:
        run_stage_privacy_budget(args.output_dir, max_rounds=args.rounds)

    if args.stage in ["plots", "all"]:
        print(">>> Regenerating all publication figures...")
        generate_all_plots()
        print("[OK] All figures successfully regenerated!")

    print("\n[COMPLETE] All requested experiment stages completed successfully.")


if __name__ == "__main__":
    main()
