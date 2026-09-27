"""
Robustness plots for the robot Beta distractor-intensity sweep.

The Beta distribution controls the distractor's intensity in
[intensity_min, intensity_max] (training prior = Uniform = scaled Beta(1,1)).

Input
-----
  result_summary/robot/beta_robustness/<tag>/summary_beta_<tag>.npz
  result_summary/robot/beta_robustness/<tag>/results_beta_<tag>_a*_b*.npz

Output (per model tag)
----------------------
  heatmap_accuracy_beta_<tag>_highlighted.png
  heatmap_stopstep_beta_<tag>_highlighted.png
  scatter_kl_vs_acc_<tag>.png
"""

from __future__ import annotations

import os
import glob
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.special import digamma, betaln

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR          = "result_summary/robot/beta_robustness"
ALPHA_BETA_VALUES = [0.5, 1, 3, 5, 7]

# Set to None to auto-discover all summary files under BASE_DIR.
SUMMARY_PATHS: list[str] | None = None
# Or list explicitly:
# SUMMARY_PATHS = [
#     "result_summary/robot/beta_robustness/INDEP_H35_seed215/summary_beta_INDEP_H35_seed215.npz",
# ]

# ---------------------------------------------------------------------------
# KL divergence: Beta(pa, pb) || Beta(qa, qb)
# Scaling to [intensity_min, intensity_max] is a bijection, so KL is unchanged.
# Training prior = Beta(1, 1) = Uniform on [0, 1].
# ---------------------------------------------------------------------------
def kl_beta(pa: float, pb: float, qa: float = 1.0, qb: float = 1.0) -> float:
    """KL( Beta(pa,pb) || Beta(qa,qb) )."""
    return float(
        betaln(qa, qb) - betaln(pa, pb)
        + (pa - qa) * digamma(pa)
        + (pb - qb) * digamma(pb)
        + (qa + qb - pa - pb) * digamma(pa + pb)
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
Z95         = 1.96
tick_labels = [str(v) for v in ALPHA_BETA_VALUES]
uniform_idx = ALPHA_BETA_VALUES.index(1)   # Beta(1,1) = Uniform training prior

COMBO_MARKERS    = {v: m for v, m in zip(ALPHA_BETA_VALUES, ["o", "s", "D", "^", "P"])}
ALPHA_COLORS     = plt.cm.viridis(np.linspace(0.1, 0.9, len(ALPHA_BETA_VALUES)))
alpha_color_map  = {v: ALPHA_COLORS[i] for i, v in enumerate(ALPHA_BETA_VALUES)}

combos   = list(itertools.product(ALPHA_BETA_VALUES, ALPHA_BETA_VALUES))
alphas_v = np.array([c[0] for c in combos])
betas_v  = np.array([c[1] for c in combos])
kl_fwd   = np.array([kl_beta(a, b, 1.0, 1.0) for a, b in combos])
kl_rev   = np.array([kl_beta(1.0, 1.0, a, b) for a, b in combos])


def _highlight_uniform(ax):
    ax.add_patch(mpatches.FancyBboxPatch(
        (uniform_idx, uniform_idx), 1, 1,
        boxstyle="square,pad=0",
        linewidth=3, edgecolor="red", facecolor="none",
        transform=ax.transData, clip_on=False,
    ))


def plot_heatmap(data, annot, title, cbar_label, out_path,
                 fmt="", vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        data, ax=ax,
        cmap=sns.color_palette("coolwarm", as_cmap=True).reversed(),
        annot=annot, fmt=fmt,
        vmin=vmin, vmax=vmax,
        xticklabels=tick_labels, yticklabels=tick_labels,
        cbar_kws={"label": cbar_label},
        linewidths=0.4, linecolor="lightgray",
        annot_kws={"size": 9},
    )
    _highlight_uniform(ax)
    ax.set_xlabel(r"$\beta$  (shape parameter)")
    ax.set_ylabel(r"$\alpha$  (shape parameter)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _add_scatter(ax, kl_vals, accs, xlabel: str) -> None:
    for a_val in ALPHA_BETA_VALUES:
        mask = alphas_v == a_val
        ax.scatter(
            kl_vals[mask], accs[mask],
            color=alpha_color_map[a_val],
            marker=COMBO_MARKERS[a_val],
            s=80, zorder=3, label=rf"$\alpha$={a_val}",
        )
        for x, y, b_val in zip(kl_vals[mask], accs[mask], betas_v[mask]):
            ax.annotate(
                rf"$\beta$={b_val}", (x, y),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7, color=alpha_color_map[a_val],
            )
    uni_mask = (alphas_v == 1.0) & (betas_v == 1.0)
    ax.scatter(
        kl_vals[uni_mask], accs[uni_mask],
        s=180, marker="*", color="red", zorder=5,
        label=r"Uniform ($\alpha$=$\beta$=1)",
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Accuracy  vs  " + xlabel, fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


# ---------------------------------------------------------------------------
# Per-summary processing
# ---------------------------------------------------------------------------

def process_summary(summary_path: str) -> None:
    summary = np.load(summary_path, allow_pickle=True)
    acc_grid      = summary["acc_grid"]           # [5, 5]
    stopstep_grid = summary["stopstep_grid"]      # [5, 5]
    H             = int(summary["horizon"])
    method        = str(summary["method"])
    seed          = int(summary["seed"])
    intensity_min = float(summary.get("intensity_min", 0.0))
    intensity_max = float(summary.get("intensity_max", 0.7))
    tag           = f"{method}_H{H}_seed{seed}"
    out_dir       = os.path.dirname(summary_path)

    print(f"\n{'='*60}")
    print(f"  Tag: {tag}")
    print(f"  Intensity range: [{intensity_min:.2f}, {intensity_max:.2f}]")

    # ---- build per-combo annotation strings (acc +/- 95% CI) ----
    annot_acc = np.empty((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)), dtype=object)
    for ai, alpha in enumerate(ALPHA_BETA_VALUES):
        for bi, beta_v in enumerate(ALPHA_BETA_VALUES):
            combo_tag = (
                f"a{str(alpha).replace('.','p')}_b{str(beta_v).replace('.','p')}"
            )
            fpath = os.path.join(out_dir, f"results_beta_{tag}_{combo_tag}.npz")
            if os.path.exists(fpath):
                d  = np.load(fpath)
                x  = d["all_correct"]
                n  = len(x)
                p  = x.mean()
                hw = Z95 * np.sqrt(p * (1 - p) / n)
                annot_acc[ai, bi] = f"{p:.2f}\n+/-{hw:.2f}"
            else:
                annot_acc[ai, bi] = f"{acc_grid[ai, bi]:.2f}"

    # ---- Heatmap 1: accuracy ----
    plot_heatmap(
        acc_grid, annot_acc,
        title=r"$P(\hat{c}_\tau = c)$ vs. Beta($\alpha$,$\beta$) distractor intensity prior ",
        cbar_label=r"Accuracy  $P(\hat{c}_\tau = c)$",
        out_path=os.path.join(out_dir, f"heatmap_accuracy_beta_{tag}_highlighted.png"),
        vmin=0.5, vmax=1.0,
    )

    # ---- Heatmap 2: stop step ----
    plot_heatmap(
        stopstep_grid + 1,
        annot=True, fmt=".1f",
        title=rf"Avg stop step vs. Beta($\alpha$,$\beta$) distractor intensity prior ",
        cbar_label="Avg stop step (1-based)",
        out_path=os.path.join(out_dir, f"heatmap_stopstep_beta_{tag}_highlighted.png"),
        vmin=0, vmax=10,
    )

    # ---- KL scatter ----
    accs = np.array([
        acc_grid[ALPHA_BETA_VALUES.index(a), ALPHA_BETA_VALUES.index(b)]
        for a, b in combos
    ])

    x_min = min(kl_fwd.min(), kl_rev.min())
    x_max = max(kl_fwd.max(), kl_rev.max())
    x_pad = (x_max - x_min) * 0.05

    fig_kl, (ax_fwd, ax_rev) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _add_scatter(ax_fwd, kl_fwd, accs,
                 r"KL( Beta($\alpha$,$\beta$) $\|$ Uniform )")
    _add_scatter(ax_rev, kl_rev, accs,
                 r"KL( Uniform $\|$ Beta($\alpha$,$\beta$) )")
    for ax in (ax_fwd, ax_rev):
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    fig_kl.suptitle(
        f"Accuracy vs. KL divergence from training prior  [{tag}]\n"
        rf"Distractor intensity ~ Beta($\alpha$,$\beta$) scaled to "
        f"[{intensity_min:.2f}, {intensity_max:.2f}]",
        fontsize=10,
    )
    plt.tight_layout()
    out_kl = os.path.join(out_dir, f"scatter_kl_vs_acc_{tag}.png")
    plt.savefig(out_kl, dpi=150)
    plt.close(fig_kl)
    print(f"Saved -> {out_kl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths = SUMMARY_PATHS
    if paths is None:
        paths = sorted(glob.glob(
            os.path.join(BASE_DIR, "**", "summary_beta_*.npz"), recursive=True
        ))
    if not paths:
        print(f"No summary files found under {BASE_DIR}.")
        return

    print(f"Found {len(paths)} summary file(s).")
    for p in paths:
        process_summary(p)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
