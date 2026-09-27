"""
Regenerate Dirichlet robustness plots from saved results.

Produces per concentration value:
  tv_calib_<tag>_a<alpha>.png  -- abs(inf_prob - (1-delta)) vs TV scatter
  tv_acc_<tag>_a<alpha>.png    -- binned accuracy (fraction correct) vs TV with CI
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "figure.dpi":        150,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "result_summary/bandit/dirichlet_robustness/A5P3"
TAG      = "icmape_td3_H60_K5_P3_seed215" # change this to the desired tag
OUT_DIR  = DATA_DIR

CONCENTRATION_VALUES = [0.1, 0.5, 1, 5, 10]

DELTA    = 0.1
NUM_BINS = 10

# ---------------------------------------------------------------------------
# Nadaraya-Watson smoother helper
# ---------------------------------------------------------------------------
NW_BANDWIDTH = 0.05
NW_GRID_SIZE = 300

def _nw_smooth(x_vals, y_vals, bw=NW_BANDWIDTH):
    x_grid = np.linspace(x_vals.min(), x_vals.max(), NW_GRID_SIZE)
    w      = np.exp(-0.5 * ((x_vals[:, None] - x_grid[None, :]) / bw) ** 2)
    w_sum  = w.sum(0)
    valid  = w_sum > 1e-6
    means  = np.where(valid, (w * y_vals[:, None]).sum(0) / w_sum, np.nan)
    means2 = np.where(valid, (w * (y_vals ** 2)[:, None]).sum(0) / w_sum, np.nan)
    n_eff  = w_sum ** 2 / (w ** 2).sum(0)
    ci     = np.where(valid,
                      1.96 * np.sqrt(np.maximum(means2 - means ** 2, 0) / np.maximum(n_eff, 1)),
                      np.nan)
    return x_grid, means, ci, valid

# ---------------------------------------------------------------------------
# Per-concentration plots
# ---------------------------------------------------------------------------
for conc in CONCENTRATION_VALUES:
    a_str    = str(conc).replace(".", "p")
    npz_path = os.path.join(DATA_DIR, f"results_dirichlet_{TAG}_a{a_str}.npz")
    if not os.path.exists(npz_path):
        print(f"  [skip] {npz_path} not found")
        continue

    data        = np.load(npz_path, allow_pickle=True)
    inf_prob    = data["inf_prob"].astype(float)
    all_correct = data["all_correct"].astype(float)
    tv          = data["tv"].astype(float)
    base_acc    = float(data["base_acc"])

    finite = np.isfinite(tv) & np.isfinite(inf_prob)
    tv_f, inf_f, cor_f = tv[finite], inf_prob[finite], all_correct[finite]

    xlabel_tv = (
        rf"TV(Cat($\theta$), Uniform) -- "
        rf"$\theta \sim \mathrm{{Dir}}({conc}\cdot\mathbf{{1}}_P)$"
    )

    # ------------------------------------------------------------------
    # Plot 1: |inf_prob - (1-delta)| vs TV
    # ------------------------------------------------------------------
    calib  = np.abs(inf_f - (1.0 - DELTA))
    colors = np.where(cor_f == 1, '#4878d0', '#ee854a')

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(tv_f, calib, c=colors, alpha=0.06, s=3, linewidths=0, rasterized=True)

    # x_grid, means, ci, valid = _nw_smooth(tv_f, calib)
    # ax1.fill_between(x_grid[valid], (means - ci)[valid], (means + ci)[valid],
    #                  color='black', alpha=0.15)
    # ax1.plot(x_grid[valid], means[valid], color='black', linewidth=2, zorder=3)

    ax1.axhline(base_acc - (1.0 - DELTA), color='#e03030', linestyle='--',
                linewidth=1.8, zorder=4, label="Baseline (i.i.d. Uniform)")

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4878d0',
               markersize=7, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ee854a',
               markersize=7, label='Incorrect'),
        Line2D([0], [0], color='#e03030', linestyle='--', linewidth=1.8,
               label='Baseline (i.i.d. Uniform)'),
    ]
    ax1.set_xlabel(xlabel_tv)
    ax1.set_ylabel(rf"$|\hat{{p}}_\tau - (1-\delta)|$   $(\delta={DELTA})$")
    ax1.set_title(rf"Calibration gap vs TV", fontsize=12, fontweight='bold', pad=8)
    ax1.set_ylim(0, 0.5)
    ax1.legend(handles=legend_handles, loc='upper left',
               handletextpad=0.4, borderpad=0.6)
    fig1.tight_layout()
    out1 = os.path.join(OUT_DIR, f"tv_calib_{TAG}_a{a_str}.png")
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved -> {out1}")

    # ------------------------------------------------------------------
    # Plot 2: binned accuracy vs TV with Wilson 95% CI
    # ------------------------------------------------------------------
    tv_min, tv_max = tv_f.min(), tv_f.max()
    bins = np.linspace(tv_min, tv_max, NUM_BINS + 1)
    bin_centers, bin_acc, bin_ci_lo, bin_ci_hi = [], [], [], []

    for i in range(NUM_BINS):
        mask = (tv_f >= bins[i]) & (tv_f < bins[i + 1])
        if i == NUM_BINS - 1:
            mask = (tv_f >= bins[i]) & (tv_f <= bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        p     = cor_f[mask].mean()
        z     = 1.96
        denom  = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
        bin_acc.append(p)
        bin_ci_lo.append(max(0.0, center - half))
        bin_ci_hi.append(min(1.0, center + half))

    bin_centers = np.array(bin_centers)
    bin_acc     = np.array(bin_acc)
    bin_ci_lo   = np.array(bin_ci_lo)
    bin_ci_hi   = np.array(bin_ci_hi)
    bin_widths  = np.diff(np.linspace(tv_min, tv_max, NUM_BINS + 1))

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(bin_centers, bin_acc, width=bin_widths * 0.82,
            color='#4878d0', alpha=0.85, zorder=3, label="Accuracy")
    ax2.errorbar(bin_centers, bin_acc,
                 yerr=[bin_acc - bin_ci_lo, bin_ci_hi - bin_acc],
                 fmt='none', color='#333333', capsize=4, linewidth=1.4, zorder=4)
    ax2.axhline(base_acc, color='#e03030', linestyle='--', linewidth=1.8,
                zorder=5, label="Baseline (i.i.d. Uniform)")
    ax2.set_xlabel(xlabel_tv)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    # ax2.legend(loc='upper right', handletextpad=0.4, borderpad=0.6)
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, f"tv_acc_{TAG}_a{a_str}.png")
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved -> {out2}")

print("\nAll plots saved.")
