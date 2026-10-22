"""Cascaded Tanks benchmark: Table 1, Figure 2 and Figure 3 of the paper.

BTN-V (Volterra degree D = 3, memory M = 100, initial rank R = 20) is trained
10 times (seeds 0-9) with the lag precisions delta on and off. Reports RMSE,
NLL, training time and effective rank on the validation data, together with
the BMVALS baseline (predictions in data/BMVALS_all_runs.csv).

Figures (from seed / run 2):
  Figure 2  |W| per factor and rank component, delta off (top) and on (bottom)
  Figure 3  validation output with +-3 std bounds, BTN-V and BMVALS
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnv import BTNV

M, D, R_MAX = 100, 3, 20
SEEDS = range(10)
PLOT_SEED = 2


def gaussian_nll(y, mean, std):
    return np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((y - mean) ** 2) / (std**2))


# ---- Data: inputs scaled to [0, 1], outputs standardised (training statistics) ----
data = scipy.io.loadmat(ROOT / "data/dataBenchmark.mat")
u_train, y_train = data["uEst"].squeeze()[:, None], data["yEst"].squeeze()
u_test, y_test = data["uVal"].squeeze()[:, None], data["yVal"].squeeze()

u_min, u_max = u_train.min(axis=0), u_train.max(axis=0)
u_range = np.where(u_max - u_min == 0, 1, u_max - u_min)
u_train = (u_train - u_min) / u_range
u_test = (u_test - u_min) / u_range

y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / y_std

# The first M validation samples are skipped (their lag vectors contain the zero padding).
y_true = y_test[M:]

# ---- BTN-V, delta on and off ----
results = {}  # results[mode] = dict(rmse=[...], nll=[...], time=[...], rank=[...])
plot_runs = {}  # model and predictions of PLOT_SEED, for the figures
for mode in ["on", "off"]:
    res = dict(rmse=[], nll=[], time=[], rank=[])
    for seed in SEEDS:
        print(f"\n===== BTN-V, delta {mode}, seed {seed} =====")
        model = BTNV().train(
            u_train, y_train, M=M, D=D, R=R_MAX,
            a0=1e-3, b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6,
            max_iter=25, seed=seed, delta_update=(mode == "on"),
        )
        mean, std = model.predict(u_test)
        mean, std = mean[M:] * y_std + y_mean, std[M:] * y_std

        res["rmse"].append(np.sqrt(np.mean((mean - y_true) ** 2)))
        res["nll"].append(gaussian_nll(y_true, mean, std))
        res["time"].append(model.train_time)
        res["rank"].append(model.rank)
        if seed == PLOT_SEED:
            plot_runs[mode] = (model, mean, std)
    results[mode] = res

# ---- BMVALS baseline (10 runs, predictive mean and variance per validation sample) ----
bmvals = pd.read_csv(ROOT / "data/BMVALS_all_runs.csv")
bmvals.columns = ["run_index", "y_pred", "variance"]
results["BMVALS"] = dict(rmse=[], nll=[])
for run in np.unique(bmvals["run_index"]):
    g = bmvals[bmvals["run_index"] == run]
    mean, std = g["y_pred"].values, np.sqrt(g["variance"].values)
    results["BMVALS"]["rmse"].append(np.sqrt(np.mean((mean - y_true) ** 2)))
    results["BMVALS"]["nll"].append(gaussian_nll(y_true, mean, std))
    if run == PLOT_SEED:
        plot_runs["BMVALS"] = (None, mean, std)

# ---- Table 1 ----
def ms(values):
    return f"{np.mean(values):.5f} ± {np.std(values):.5f}" if values else "-"

rows = []
for name, key in [("BTN-V (delta: on)", "on"), ("BTN-V (delta: off)", "off"), ("BMVALS", "BMVALS")]:
    r = results[key]
    rows.append({"Method": name, "RMSE": ms(r["rmse"]), "NLL": ms(r["nll"]),
                 "Time (s)": ms(r.get("time", [])), "R_eff": ms(r.get("rank", []))})
pd.set_option("display.width", 200)
print("\n================ Table 1: performance over 10 runs ================\n")
print(pd.DataFrame(rows).to_string(index=False))
print("\nBMVALS training time is not computed here (it was measured with the original MATLAB code).")

# ---- Figure 2: |W^(d)| per rank component, delta off (top) and on (bottom) ----
fig, axes = plt.subplots(2, D, figsize=(2 * D, 4), sharey="row", sharex=True)
for row, mode in enumerate(["off", "on"]):
    model = plot_runs[mode][0]
    for d, ax in enumerate(axes[row]):
        W = model.W_D[d]
        for r in range(W.shape[1]):
            ax.plot(np.abs(W[:, r]), linewidth=3, linestyle="--" if r == 1 else "-", label=f"Comp {r + 1}", alpha=0.8)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", which="major", labelsize=11)
        if row == 0:
            ax.set_title(f"Mode {d + 1}", fontsize=13, fontweight="bold")
    axes[row, 0].set_ylabel(rf"Weights ($\mathbf{{\delta}}: \mathrm{{{mode}}}$)", fontsize=12)
fig.supxlabel("Memory (Lags)", fontsize=12)
handles, labels = axes[1, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()

# ---- Figure 3: validation output with +-3 std bounds ----
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
t = np.arange(len(y_true))
for ax, (key, title) in zip(axes, [("on", "BTN-V"), ("BMVALS", "BMVALS")]):
    _, mean, std = plot_runs[key]
    ax.plot(t, y_true, color="black", linestyle="-", linewidth=1.8, label="Actual Output")
    ax.plot(t, mean, color="black", linestyle="--", linewidth=1.8, label=f"{title} Prediction")
    ax.fill_between(t, mean - 3 * std, mean + 3 * std, color="gray", alpha=0.2, label="Prediction ±3 std")
    ax.set_title(title, fontsize=24)
    ax.set_ylabel("Validation Output", fontsize=22)
    ax.set_ylim([0, 11.5])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(direction="in", length=4, labelsize=20)
axes[1].set_xlabel("Time", fontsize=22)
plt.tight_layout()
plt.show()
