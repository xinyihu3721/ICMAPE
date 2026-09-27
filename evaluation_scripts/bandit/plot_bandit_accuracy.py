import numpy as np
import matplotlib.pyplot as plt
import os
import glob

RESULT_DIR = "result_summary/bandit"
CONDITIONS = {
    "A3P5": "A3P5",
    "A5P3": "A5P3",
}
METHOD_MAP = {
    "icmape_vdn":     "VDN",
    "icmape_indep":   "INDEP",
    "icmape_td3":     "TD3",
    "icmape_uniform": "UNIFORM",
}


DISPLAY_ORDER = ["VDN", "INDEP", "TD3", "UNIFORM"]
COLORS = ["#A6CA68", "#FC757B", "#3C9BC9", "#FFB83D"]  # VDN2, INDEP, TD3, UNIFORM


def load_condition(cond_dir):
    """Returns {method_label: {"acc": [...], "step": [...]}} per seed."""
    data = {label: {"acc": [], "step": []} for label in METHOD_MAP.values()}
    for npz_path in glob.glob(os.path.join(cond_dir, "results_*.npz")):
        fname = os.path.basename(npz_path)
        matched_key = next(
            (k for k in METHOD_MAP if f"results_{k}_H" in fname),
            None,
        )
        if matched_key is None:
            continue
        label = METHOD_MAP[matched_key]
        arr = np.load(npz_path)
        data[label]["acc"].append(float(arr["all_correct"].mean()))
        data[label]["step"].append(float(arr["all_stopping_time"].mean()))
    return data


def compute_stats(data, key):
    """Returns (avg_across_seeds, std_across_seeds) for the given key."""
    avgs, stds = [], []
    for label in DISPLAY_ORDER:
        vals = data[label][key]
        if len(vals) == 0:
            avgs.append(0.0)
            stds.append(0.0)
        else:
            avgs.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return np.array(avgs), np.array(stds)


METRICS = [
    ("acc",  "Accuracy (Correct Identification Rate)", "Accuracy", (0, 1.1), "{:.2f}"),
    ("step", "Avg Stopping Step", "Avg Stopping Step", None, "{:.1f}"),
]


def plot_cell(ax, data, key, ylabel, yaxis_label, ylim, fmt):
    avgs, stds = compute_stats(data, key)
    x = np.arange(len(DISPLAY_ORDER))
    bars = ax.bar(x, avgs, yerr=stds, capsize=5, color=COLORS, width=0.55,
                  error_kw=dict(elinewidth=1.5, ecolor="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_ORDER, fontsize=11)
    ax.set_ylabel(yaxis_label, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt.format(v)))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, avg, std in zip(bars, avgs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, avg + std + (ylim[1] * 0.02 if ylim else avg * 0.03),
                fmt.format(avg), ha="center", va="bottom", fontsize=9)


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(27, 5))
gs = GridSpec(1, 5, figure=fig, width_ratios=[2, 2, 0.15, 2, 2], wspace=0.35)

GROUP_LABELS = ["(a)", "(b)"]
axes_groups = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
               [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4])]]

for g_idx, (cond_label, cond_folder) in enumerate(CONDITIONS.items()):
    cond_dir = os.path.join(RESULT_DIR, cond_folder)
    data = load_condition(cond_dir)
    axs = axes_groups[g_idx]
    for m_idx, (key, ylabel, yaxis_label, ylim, fmt) in enumerate(METRICS):
        ax = axs[m_idx]
        plot_cell(ax, data, key, ylabel, yaxis_label, ylim, fmt)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")

    # group label below the pair
    x_left = axs[0].get_position().x0
    x_right = axs[1].get_position().x1
    x_mid = (x_left + x_right) / 2
    fig.text(x_mid, 0.03, f"{GROUP_LABELS[g_idx]} {cond_label}",
             ha="center", va="top", fontsize=13, fontweight="bold")

plt.tight_layout()

out_path = "result_summary/bandit_accuracy_grouped_bar_same row.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()

# --- Split into individual figures per condition ---
SPLIT_NAMES = {
    "A3P5": ("result_summary/bandit_accuracy_grouped_bar_a.png", "(a)"),
    "A5P3": ("result_summary/bandit_accuracy_grouped_bar_b.png", "(b)"),
}

for cond_label, cond_folder in CONDITIONS.items():
    cond_dir = os.path.join(RESULT_DIR, cond_folder)
    data = load_condition(cond_dir)
    split_path, group_label = SPLIT_NAMES[cond_label]

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    fig2.subplots_adjust(bottom=0.15)
    for ax, (key, ylabel, yaxis_label, ylim, fmt) in zip(axs2, METRICS):
        plot_cell(ax, data, key, ylabel, yaxis_label, ylim, fmt)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")

    fig2.text(0.5, 0.02, f"{group_label} {cond_label}",
              ha="center", va="top", fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(split_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {split_path}")
    plt.show()
