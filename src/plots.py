import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOSPITALS = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]
SITE_LABELS = {0: "Cleveland", 1: "Hungarian", 2: "Switzerland", 3: "VA", "0": "Cleveland", "1": "Hungarian", "2": "Switzerland", "3": "VA", "all": "Consortium"}
COLORS = {
    "local-only": "#7f7f7f", "pooled": "#2ca02c", "fedavg": "#1f77b4", "fedprox": "#ff7f0e",
    "flower_fedavg": "#9467bd", "fedavg_poison": "#d62728", "median_defense": "#17becf",
    "ideal": "#1f77b4", "drop0.2": "#2ca02c", "drop0.5": "#ff7f0e",
    "deadline_inf": "#1f77b4", "deadline_2.0s": "#ff7f0e", "deadline_1.0s": "#d62728",
    "hetero_bw_deadline_2.0s": "#9467bd"
}

def setup_fig():
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig, ax

def plot_hospital_baselines(csv_path="results/e1.csv", out_path="plots/hospital_baselines.png"):
    df = pd.read_csv(csv_path)
    df = df[df["round"] == 50]
    methods = ["local-only", "fedavg", "pooled"]
    sites = ["0", "1", "2", "3", "all"]
    fig, ax = setup_fig()
    x = np.arange(len(sites))
    width = 0.25
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        means = [sub[sub["site"] == s]["metric_value"].mean() for s in sites]
        stds = [sub[sub["site"] == s]["metric_value"].std() for s in sites]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m, color=COLORS.get(m), alpha=0.9)
    ax.set_xticks(x + width)
    ax.set_xticklabels([SITE_LABELS[s] for s in sites])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 0.9)
    ax.set_title("Hospital baseline comparison (50 rounds, 10 seeds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_consortium_convergence(csv_path="results/e1.csv", out_path="plots/consortium_convergence.png"):
    df = pd.read_csv(csv_path)
    df = df[df["site"] == "all"]
    fig, ax = setup_fig()
    for m in ["fedavg", "fedprox", "pooled"]:
        sub = df[df["method"] == m].groupby("round")["metric_value"].agg(["mean", "std"])
        ax.plot(sub.index, sub["mean"], label=m, color=COLORS.get(m), lw=1.8)
        ax.fill_between(sub.index, sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=COLORS.get(m), alpha=0.15)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consortium test accuracy")
    ax.set_title("Federated convergence on pooled test set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_leave_one_out(e1_csv="results/e1.csv", e2_csv="results/e2.csv", out_path="plots/leave_one_out_generalization.png"):
    df1 = pd.read_csv(e1_csv)
    df2 = pd.read_csv(e2_csv)
    df1_loc = df1[(df1["round"] == 50) & (df1["method"] == "local-only")]
    df2_loso = df2[df2["round"] == 50]
    sites = [0, 1, 2, 3]
    loc_means, loc_stds, loso_means, loso_stds = [], [], [], []
    for s in sites:
        sub_loc = df1_loc[df1_loc["site"] == str(s)]["metric_value"]
        loc_means.append(sub_loc.mean())
        loc_stds.append(sub_loc.std())
        sub_loso = df2_loso[(df2_loso["heldout_site"] == s) & (df2_loso["site"] == s)]["metric_value"]
        loso_means.append(sub_loso.mean())
        loso_stds.append(sub_loso.std())
    fig, ax = setup_fig()
    x = np.arange(len(sites))
    w = 0.35
    ax.bar(x - w/2, loc_means, w, yerr=loc_stds, capsize=3, label="Local-only (trained on site)", color=COLORS["local-only"])
    ax.bar(x + w/2, loso_means, w, yerr=loso_stds, capsize=3, label="LOSO (unseen site, trained on others)", color=COLORS["fedavg"])
    ax.set_xticks(x)
    ax.set_xticklabels([SITE_LABELS[s] for s in sites])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 0.9)
    ax.set_title("Leave-one-site-out generalization vs local training")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_contribution_matrix(e1_csv="results/e1.csv", e3_csv="results/e3.csv", out_path="plots/hospital_contribution_matrix.png"):
    df1 = pd.read_csv(e1_csv)
    df3 = pd.read_csv(e3_csv)
    r1 = df1[(df1["round"] == 50) & (df1["method"] == "fedavg")]
    r3 = df3[df3["round"] == 50]
    full_acc = {s: r1[r1["site"] == str(s)]["metric_value"].mean() for s in range(4)}
    mat = np.zeros((4, 4))
    for j in range(4):
        for k in range(4):
            sub = r3[(r3["heldout_site"] == j) & (r3["site"] == k)]["metric_value"]
            mat[j, k] = (full_acc[k] - sub.mean()) * 100.0
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=150)
    vmax = max(abs(mat.min()), abs(mat.max()), 5.0)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    for j in range(4):
        for k in range(4):
            val = mat[j, k]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(k, j, f"{val:+.2f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(HOSPITALS, rotation=15, ha="right")
    ax.set_yticklabels(HOSPITALS)
    ax.set_xlabel("Target hospital evaluated (k)")
    ax.set_ylabel("Hospital removed from training (j)")
    ax.set_title("Pairwise site contribution matrix Δ(j, k)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy impact on site k (pp)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_dropout_resilience(csv_path="results/e4.csv", out_path="plots/client_dropout_resilience.png"):
    df = pd.read_csv(csv_path)
    df = df[df["site"] == "all"]
    fig, ax = setup_fig()
    for sc, lbl in [("ideal", "Ideal (p_drop=0.0)"), ("drop0.2", "Dropout p=0.2"), ("drop0.5", "Dropout p=0.5")]:
        sub = df[df["scenario"] == sc].groupby("round")["metric_value"].agg(["mean", "std"])
        ax.plot(sub.index, sub["mean"], label=lbl, color=COLORS[sc], lw=1.8)
        ax.fill_between(sub.index, sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=COLORS[sc], alpha=0.15)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consortium test accuracy")
    ax.set_title("Resilience to random client dropouts (50 rounds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_straggler_deadlines_convergence(csv_path="results/e4.csv", out_path="plots/straggler_deadlines_convergence.png"):
    df = pd.read_csv(csv_path)
    df = df[df["site"] == "all"]
    fig, ax = setup_fig()
    scenarios = [
        ("deadline_inf", "D = ∞ (no deadline, 4 active)", COLORS["deadline_inf"]),
        ("deadline_2.0s", "D = 2.0s (~1.8 active)", COLORS["deadline_2.0s"]),
        ("hetero_bw_deadline_2.0s", "D = 2.0s + Hetero BW (~1.7 active)", COLORS["hetero_bw_deadline_2.0s"]),
        ("deadline_1.0s", "D = 1.0s (starvation, 0.0 active)", COLORS["deadline_1.0s"])
    ]
    for sc, lbl, col in scenarios:
        sub = df[df["scenario"] == sc].groupby("round")["metric_value"].agg(["mean", "std"])
        ax.plot(sub.index, sub["mean"], label=lbl, color=col, lw=1.8)
        ax.fill_between(sub.index, sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=col, alpha=0.15)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consortium test accuracy")
    ax.set_title("Straggler deadline impact across 50 rounds")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_straggler_deadlines_wallclock(csv_path="results/e4.csv", out_path="plots/straggler_deadlines_wallclock.png"):
    df = pd.read_csv(csv_path)
    df = df[df["site"] == "all"]
    fig, ax = setup_fig()
    scenarios = [
        ("deadline_inf", "D = ∞ (completed 50 rnds in 172s)", COLORS["deadline_inf"]),
        ("deadline_2.0s", "D = 2.0s (completed 50 rnds in 85s)", COLORS["deadline_2.0s"]),
        ("hetero_bw_deadline_2.0s", "D = 2.0s + BW (completed 50 rnds in 82s)", COLORS["hetero_bw_deadline_2.0s"]),
        ("deadline_1.0s", "D = 1.0s (completed 50 rnds in 50s)", COLORS["deadline_1.0s"])
    ]
    for sc, lbl, col in scenarios:
        sub = df[df["scenario"] == sc].groupby("round")[["sim_time_s", "metric_value"]].agg(["mean", "std"])
        t = sub["sim_time_s"]["mean"]
        acc = sub["metric_value"]["mean"]
        std = sub["metric_value"]["std"]
        ax.plot(t, acc, label=lbl, color=col, lw=1.8)
        ax.fill_between(t, acc - std, acc + std, color=col, alpha=0.15)
        # Mark final round 50 completion point
        ax.scatter([t.iloc[-1]], [acc.iloc[-1]], color=col, s=40, zorder=5)
    ax.set_xlabel("Simulated wall-clock time (seconds)")
    ax.set_ylabel("Consortium test accuracy")
    ax.set_title("Test accuracy vs simulated wall-clock time (all 50 rounds)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_flower_parity(e1_csv="results/e1.csv", e5_csv="results/e5.csv", out_path="plots/flower_deployment_parity.png"):
    df1 = pd.read_csv(e1_csv)
    df5 = pd.read_csv(e5_csv)
    sub1 = df1[(df1["site"] == "all") & (df1["method"] == "fedavg")].groupby("round")["metric_value"].agg(["mean", "std"])
    sub5 = df5[df5["site"] == "all"].groupby("round")["metric_value"].agg(["mean", "std"])
    fig, ax = setup_fig()
    ax.plot(sub1.index, sub1["mean"], label="Single-process FedAvg (sim)", color=COLORS["fedavg"], lw=1.8)
    ax.fill_between(sub1.index, sub1["mean"] - sub1["std"], sub1["mean"] + sub1["std"], color=COLORS["fedavg"], alpha=0.15)
    ax.plot(sub5.index, sub5["mean"], label="Flower multi-process FedAvg", color=COLORS["flower_fedavg"], lw=1.8, linestyle="--")
    ax.fill_between(sub5.index, sub5["mean"] - sub5["std"], sub5["mean"] + sub5["std"], color=COLORS["flower_fedavg"], alpha=0.15)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consortium test accuracy")
    ax.set_title("Flower multi-process deployment parity verification")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_differential_privacy(csv_path="results/e6.csv", out_path="plots/differential_privacy_tradeoff.png"):
    df = pd.read_csv(csv_path)
    df = df[(df["round"] == 50) & (df["site"] == "all")]
    zs = [0.0, 0.01, 0.05, 0.1, 0.3]
    means, stds = [], []
    for z in zs:
        sub = df[df["scenario"] == f"dp_z{z}"]["metric_value"]
        means.append(sub.mean())
        stds.append(sub.std())
    fig, ax = setup_fig()
    ax.errorbar(zs, means, yerr=stds, fmt="o-", color=COLORS["fedavg"], capsize=4, lw=1.8, label="DP-FedAvg (C=0.1)")
    ax.set_xlabel("Noise multiplier z")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Accuracy vs differential privacy noise multiplier")
    ax.set_ylim(0.7, 0.82)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_byzantine_attacks(csv_path="results/e6.csv", out_path="plots/byzantine_attacks_and_defense.png"):
    df = pd.read_csv(csv_path)
    df = df[df["site"] == "all"]
    fig, ax = setup_fig()
    scenarios = [
        ("dp_z0.0", "Clean FedAvg", "#1f77b4", "-"),
        ("labelflip_q0.5", "Label-flip q=0.5 (FedAvg)", "#ff7f0e", "--"),
        ("labelflip_median", "Label-flip q=0.5 (Median)", "#d62728", "-."),
        ("garbage_fedavg", "Garbage update (FedAvg)", "#8c564b", "--"),
        ("garbage_median", "Garbage update (Median)", "#2ca02c", "-")
    ]
    for sc, lbl, col, ls in scenarios:
        sub = df[df["scenario"] == sc].groupby("round")["metric_value"].mean()
        ax.plot(sub.index, sub, label=lbl, color=col, lw=1.8, linestyle=ls)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Consortium test accuracy")
    ax.set_title("Byzantine attacks and coordinate-wise median defense")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def generate_all_plots():
    plot_hospital_baselines()
    plot_consortium_convergence()
    plot_leave_one_out()
    plot_contribution_matrix()
    plot_dropout_resilience()
    plot_straggler_deadlines_convergence()
    plot_straggler_deadlines_wallclock()
    plot_flower_parity()
    plot_differential_privacy()
    plot_byzantine_attacks()
