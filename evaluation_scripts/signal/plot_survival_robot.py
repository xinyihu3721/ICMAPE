"""
Plot survival functions S(t) = P(tau > t) for robot signal-detector exploration models.

Reads results_<tag>.npz files produced by result_robot_explore.py (which contain
all_stopping_time per episode) and computes the survival function on the fly.
Last timestep is dropped (always 0 by definition for a bounded horizon).

Saves: result_summary/robot/explore/survival_plot.png
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.frameon":    False,
    "legend.fontsize":   10,
})

IN_DIR = "result_summary/robot/explore"
OUT    = os.path.join(IN_DIR, "survival_plot.png")

ALG_STYLE = {
    "TD3":     dict(color="#3C9BC9", lw=2.2, ls="-",  label="TD3",     smooth=3),
    "VDN":     dict(color="#A6CA68", lw=2.2, ls="-",  label="VDN",     smooth=3),
    "INDEP":   dict(color="#FC757B", lw=2.2, ls="-",  label="INDEP",   smooth=3),
    "UNIFORM": dict(color="#FFB83D", lw=1.8, ls="--", label="Uniform", smooth=3),
}

# Group files by method
by_method = defaultdict(list)
for fp in sorted(glob.glob(os.path.join(IN_DIR, "results_*.npz"))):
    d = np.load(fp, allow_pickle=True)
    by_method[str(d["method"])].append(d)

if not by_method:
    raise FileNotFoundError(f"No results_*.npz files found in {IN_DIR}")

H = int(next(iter(by_method.values()))[0]["horizon"])
t_axis = np.arange(H - 1)  # drop last step (always 0 by definition)

fig, ax = plt.subplots(figsize=(7, 4.5))

print(f"\nPer-method survival std diagnostics (max over timesteps):")
for method, style in ALG_STYLE.items():
    runs = by_method.get(method)
    if not runs:
        continue

    surv_per_seed = []
    for d in runs:
        stop_times = d["all_stopping_time"].flatten()
        surv = (stop_times[:, None] > t_axis[None, :]).astype(float)
        surv_per_seed.append(surv.mean(axis=0))   # (H-1,)

    surv_stack = np.stack(surv_per_seed, axis=0)  # (n_seeds, H-1)
    mean = surv_stack.mean(0)
    std  = surv_stack.std(0)

    print(f"  {method:15s}: n_seeds={len(runs)}, max_std={std.max():.4f}, "
          f"mean_std={std.mean():.4f}"
          + (" <- near-zero: uniform policy has no learned weights, "
             "so its stopping distribution barely varies across seeds"
             if std.max() < 0.01 else ""))

    mean_s = uniform_filter1d(mean, size=style["smooth"], mode="nearest")
    std_s  = uniform_filter1d(std,  size=style["smooth"], mode="nearest")

    ax.plot(t_axis + 1, mean_s, color=style["color"], lw=style["lw"],
            ls=style["ls"], label=style["label"], zorder=3)
    # Always draw std band; for UNIFORM it is near-zero and visually absent
    # because a random policy has no learned weights -- its episode outcomes
    # are nearly identical across seeds once averaged over 10k episodes.
    ax.fill_between(t_axis + 1, mean_s - std_s, mean_s + std_s,
                    color=style["color"], alpha=0.15, zorder=2)

ax.set_xlabel("Timestep", fontsize=12)
ax.set_ylabel(r"$P(\tau > t)$", fontsize=12)
ax.set_title(f"Signal Localization (H={H})", fontsize=12, fontweight="bold", pad=8)
ax.set_xlim(1, H - 1)
ax.set_ylim(0, 1)
ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
ax.legend(loc="lower left", handlelength=2, handletextpad=0.5)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {OUT}")
