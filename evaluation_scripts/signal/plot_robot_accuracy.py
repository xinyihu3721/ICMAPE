import numpy as np
import matplotlib.pyplot as plt
import os
import glob

RESULT_DIR = "result_summary/robot/explore"

# Method names match the tags written by result_robot_explore.py
DISPLAY_ORDER = ["VDN", "INDEP", "TD3", "UNIFORM"]
COLORS        = ["#A6CA68", "#FC757B", "#3C9BC9", "#FFB83D"]


def load_results(result_dir):
    """Returns {method: {"acc": [per-seed means], "step": [per-seed means]}}."""
    data = {m: {"acc": [], "step": []} for m in DISPLAY_ORDER}
    for npz_path in glob.glob(os.path.join(result_dir, "results_*.npz")):
        fname = os.path.basename(npz_path)
        matched = next(
            (m for m in DISPLAY_ORDER if fname.startswith(f"results_{m}_H")),
            None,
        )
        if matched is None:
            continue
        arr = np.load(npz_path)
        data[matched]["acc"].append(float(arr["all_correct"].mean()))
        data[matched]["step"].append(float(arr["all_stopping_time"].mean()))
    return data


def compute_stats(data, key):
    avgs, stds = [], []
    for m in DISPLAY_ORDER:
        vals = data[m][key]
        if len(vals) == 0:
            avgs.append(0.0)
            stds.append(0.0)
        else:
            avgs.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return np.array(avgs), np.array(stds)


METRICS = [
    ("acc",  "Accuracy (Correct Identification Rate)", (0, 1.1), "{:.2f}"),
    ("step", "Avg Stopping Step",                               None,     "{:.1f}"),
]


def plot_cell(ax, data, key, title, ylim, fmt):
    avgs, stds = compute_stats(data, key)
    x = np.arange(len(DISPLAY_ORDER))
    bars = ax.bar(
        x, avgs, yerr=stds, capsize=5, color=COLORS, width=0.55,
        error_kw=dict(elinewidth=1.5, ecolor="black"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_ORDER, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt.format(v)))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, avg, std in zip(bars, avgs, stds):
        offset = (ylim[1] * 0.02 if ylim else avg * 0.03)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            avg + std + offset,
            fmt.format(avg),
            ha="center", va="bottom", fontsize=9,
        )


data = load_results(RESULT_DIR)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (key, title, ylim, fmt) in zip(axes, METRICS):
    plot_cell(ax, data, key, title, ylim, fmt)

# fig.suptitle("Robot Signal Detector Exploration (2 sources)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = "result_summary/robot_accuracy_bar.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
