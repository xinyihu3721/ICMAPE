"""
Summary plot: overall accuracy vs. Dirichlet concentration (alpha/theta).

Loads all results_dirichlet_*.npz files from DATA_DIR, extracts the alpha
value from the filename suffix (a<value>), and plots mean accuracy +/- 95% CI
as a function of alpha on a log x-axis.  The training prior is Dir(1*1)
(alpha=1 = Uniform), highlighted with a vertical dashed line.
"""

from __future__ import annotations

import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "result_summary/bandit/dirichlet_robustness/A5P3" # change this to the desired output directory
OUT_PATH = os.path.join(DATA_DIR, "summary_dirichlet_accuracy.png")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
records = []
for path in glob.glob(os.path.join(DATA_DIR, "results_dirichlet_*.npz")):
    # Extract alpha from suffix  _a<value>.npz  (e.g. _a0p1 -> 0.1, _a10 -> 10)
    m = re.search(r"_a([\dp]+)\.npz$", os.path.basename(path))
    if m is None:
        continue
    alpha = float(m.group(1).replace("p", "."))
    d     = np.load(path, allow_pickle=True)
    x     = d["all_correct"].astype(float)
    n     = len(x)
    p     = x.mean()
    se    = np.sqrt(p * (1 - p) / n)
    base_acc = float(d["base_acc"])
    records.append({"alpha": alpha, "acc": p, "se": se, "base_acc": base_acc, "n": n})

if not records:
    raise FileNotFoundError(f"No results_dirichlet_*.npz found in {DATA_DIR}")

records.sort(key=lambda r: r["alpha"])
alphas   = np.array([r["alpha"]    for r in records])
accs     = np.array([r["acc"]      for r in records])
ses      = np.array([r["se"]       for r in records])
base_acc = records[0]["base_acc"]   # same across all files
n_per    = records[0]["n"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

Z95 = 1.96
fig, ax = plt.subplots(figsize=(7, 4))

x_pos = np.arange(len(alphas))
tick_labels = [str(a) for a in alphas]

bars = ax.bar(
    x_pos, accs, yerr=Z95 * ses,
    color="#3C9BC9", alpha=0.80, width=0.6,
    capsize=5, error_kw=dict(elinewidth=1.5, ecolor="black"),
    label=f"Accuracy +/- 95% CI  (n={n_per:,}/alpha)",
    zorder=3,
)

# Highlight the training prior bar (alpha=1)
# train_idx = list(alphas).index(1.0) if 1.0 in alphas else None
# if train_idx is not None:
#     bars[train_idx].set_edgecolor("#e03030")
#     bars[train_idx].set_linewidth(2.5)

# Baseline accuracy (i.i.d. Uniform)
# ax.axhline(base_acc, color="#888888", linestyle="--", linewidth=1.4,
#            label=f"Baseline acc (i.i.d. Uniform) = {base_acc:.3f}")

ax.set_xticks(x_pos)
ax.set_xticklabels(tick_labels)
ax.set_xlabel(r"Dirichlet concentration  $\alpha$", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
# Show raw decimal accuracy values on y-axis.
# ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=1))
# ax.set_ylim(max(0, accs.min() - 0.05), min(1.02, accs.max() + 0.05))
ax.set_ylim(0.5, 1)
ax.grid(axis="y", linewidth=0.5, alpha=0.4)
ax.legend(fontsize=9, framealpha=0.85)
ax.set_title(
    r"Overall Accuracy vs. Dirichlet Concentration $\alpha$",
    fontsize=12, fontweight="bold", pad=8,
)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {OUT_PATH}")
