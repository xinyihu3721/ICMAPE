"""
Regenerate robustness plots from saved Beta-sweep results.

Produces:
  1. heatmap_accuracy_beta_<tag>_highlighted.png  -- accuracy heatmap with the
       training prior (Uniform = Beta(1,1)) cell outlined in red
  2. scatter_kl_fwd_vs_acc_<tag>.png              -- KL(Beta||Uniform) vs accuracy
  3. scatter_kl_rev_vs_acc_<tag>.png              -- KL(Uniform||Beta) vs accuracy
"""

import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.special import digamma, betaln

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "result_summary/bandit/beta_robustness/A5P3" # change this to the desired output directory
TAG      = "icmape_td3_H60_K5_P3_seed215"
OUT_DIR  = DATA_DIR

ALPHA_BETA_VALUES = [0.5, 1, 3, 5, 7]

# ---------------------------------------------------------------------------
# Load summary
# ---------------------------------------------------------------------------
summary = np.load(os.path.join(DATA_DIR, f"summary_beta_{TAG}.npz"), allow_pickle=True)
acc_grid      = summary["acc_grid"]          # (5, 5)  rows=alpha, cols=beta
stopstep_grid = summary["stopstep_grid"]     # (5, 5)
ab_vals       = summary["alpha_beta_values"] # [0.5, 1, 3, 5, 7]
TOTAL_SAMPLES = int(summary["total_samples"])

tick_labels = [str(v) for v in ALPHA_BETA_VALUES]
uniform_idx = ALPHA_BETA_VALUES.index(1)     # Beta(1,1) == Uniform

# ---------------------------------------------------------------------------
# KL divergence helpers (Beta vs Beta(1,1) = Uniform on [0,1])
#   KL(P || Q) for Beta distributions:
#     = log B(q_a,q_b)/B(p_a,p_b)
#       + (p_a - q_a)*psi(p_a)
#       + (p_b - q_b)*psi(p_b)
#       + (q_a+q_b - p_a-p_b)*psi(p_a+p_b)
# ---------------------------------------------------------------------------
def kl_beta(pa, pb, qa=1.0, qb=1.0):
    """KL(Beta(pa,pb) || Beta(qa,qb))"""
    return (
        betaln(qa, qb) - betaln(pa, pb)
        + (pa - qa) * digamma(pa)
        + (pb - qb) * digamma(pb)
        + (qa + qb - pa - pb) * digamma(pa + pb)
    )

# ---------------------------------------------------------------------------
# Load per-combo raw data to compute 95% CIs
# ---------------------------------------------------------------------------
Z95 = 1.96

ci_grid  = np.zeros((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)))   # half-width
annot_acc = np.empty((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)), dtype=object)
std_stop_grid = np.zeros((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)))
annot_stop = np.empty((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)), dtype=object)

for ai, alpha in enumerate(ALPHA_BETA_VALUES):
    for bi, beta_v in enumerate(ALPHA_BETA_VALUES):
        c_str  = str(alpha).replace(".", "p") + "_b" + str(beta_v).replace(".", "p")
        fpath  = os.path.join(DATA_DIR, f"results_beta_{TAG}_a{c_str}.npz")
        data   = np.load(fpath)
        x      = data["all_correct"]
        stop_t = data["all_stopping_time"].astype(float) + 1.0  # display as 1-based
        n      = len(x)
        p      = x.mean()
        se     = np.sqrt(p * (1 - p) / n)
        hw     = Z95 * se
        ci_grid[ai, bi]   = hw
        annot_acc[ai, bi] = f"{p:.2f}\n+/-{hw:.2f}"
        std_stop_grid[ai, bi] = stop_t.std()
        annot_stop[ai, bi] = f"{stop_t.mean():.2f}\n+/-{stop_t.std():.2f}"

# ---------------------------------------------------------------------------
# Build per-combo arrays for scatter plots
# ---------------------------------------------------------------------------
combos   = list(itertools.product(ALPHA_BETA_VALUES, ALPHA_BETA_VALUES))
alphas   = np.array([c[0] for c in combos])
betas    = np.array([c[1] for c in combos])
accs     = np.array([acc_grid[ALPHA_BETA_VALUES.index(a), ALPHA_BETA_VALUES.index(b)]
                     for a, b in combos])
kl_fwd   = np.array([kl_beta(a, b, 1.0, 1.0) for a, b in combos])  # KL(Beta||Uniform)
kl_rev   = np.array([kl_beta(1.0, 1.0, a, b) for a, b in combos])  # KL(Uniform||Beta)

# ============================================================
# Plot 1 -- Accuracy heatmap with training-prior cell outlined
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    acc_grid,
    ax=ax,
    cmap=sns.color_palette("coolwarm", as_cmap=True).reversed(),
    annot=annot_acc,
    fmt="",
    vmin=0.5, vmax=1.0,
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    cbar_kws={"label": r"$P(\hat{c}_\tau = c)$"},
    linewidths=0.4,
    linecolor="lightgray",
    annot_kws={"size": 10},
)
# Highlight the Uniform cell (alpha=1, beta=1)
ax.add_patch(mpatches.FancyBboxPatch(
    (uniform_idx, uniform_idx), 1, 1,
    boxstyle="square,pad=0",
    linewidth=3, edgecolor="red", facecolor="none",
    transform=ax.transData, clip_on=False,
))
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title(r"$P(\hat{c}_\tau = c)$ vs. Beta($\alpha$,$\beta$) arm-mean prior")
plt.tight_layout()
out1 = os.path.join(OUT_DIR, f"heatmap_accuracy_beta_{TAG}_highlighted.png")
plt.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved -> {out1}")

# ============================================================
# Plot 2 -- Stop step heatmap with training-prior cell outlined
# ============================================================
fig_ss, ax_ss = plt.subplots(figsize=(7, 5))
H = int(summary["horizon"])
sns.heatmap(
    stopstep_grid + 1,
    ax=ax_ss,
    cmap=sns.color_palette("coolwarm", as_cmap=True).reversed(),
    annot=annot_stop,
    fmt="",
    vmin=15, vmax=25,
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    cbar_kws={"label": "Avg stop step, range [0, 20]"},
    linewidths=0.4,
    linecolor="lightgray",
)
ax_ss.add_patch(mpatches.FancyBboxPatch(
    (uniform_idx, uniform_idx), 1, 1,
    boxstyle="square,pad=0",
    linewidth=3, edgecolor="red", facecolor="none",
    transform=ax_ss.transData, clip_on=False,
))
ax_ss.set_xlabel(r"$\beta$")
ax_ss.set_ylabel(r"$\alpha$")
ax_ss.set_title(r"Avg stop step vs. Beta($\alpha$,$\beta$) arm-mean prior")
plt.tight_layout()
out_ss = os.path.join(OUT_DIR, f"heatmap_stopstep_beta_{TAG}_highlighted.png")
plt.savefig(out_ss, dpi=150)
plt.close(fig_ss)
print(f"Saved -> {out_ss}")

# ============================================================
# Shared scatter helpers
# ============================================================
COMBO_MARKERS = {v: m for v, m in zip(ALPHA_BETA_VALUES, ["o", "s", "D", "^", "P"])}
ALPHA_COLORS  = plt.cm.viridis(np.linspace(0.1, 0.9, len(ALPHA_BETA_VALUES)))
alpha_color_map = {v: ALPHA_COLORS[i] for i, v in enumerate(ALPHA_BETA_VALUES)}

def _add_scatter(ax, kl_vals, xlabel):
    for a_val in ALPHA_BETA_VALUES:
        mask = alphas == a_val
        ax.scatter(
            kl_vals[mask], accs[mask],
            color=alpha_color_map[a_val],
            marker=COMBO_MARKERS[a_val],
            s=80, zorder=3,
            label=rf"$\alpha$={a_val}",
        )
        for x, y, b_val in zip(kl_vals[mask], accs[mask], betas[mask]):
            ax.annotate(
                rf"$\beta$={b_val}",
                (x, y),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7, color=alpha_color_map[a_val],
            )
    uni_mask = (alphas == 1.0) & (betas == 1.0)
    ax.scatter(
        kl_vals[uni_mask], accs[uni_mask],
        s=180, marker="*", color="red", zorder=5,
        label=r"Uniform ($\alpha$=$\beta$=1)",
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r"$P(\hat{c}_\tau = c)$", fontsize=10)
    ax.set_title(r"$P(\hat{c}_\tau = c)$" + f"  vs  {xlabel}", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

# ============================================================
# Plots 2 & 3 -- Both KL scatters side-by-side, shared x scale
# ============================================================
x_min = min(kl_fwd.min(), kl_rev.min())
x_max = max(kl_fwd.max(), kl_rev.max())
x_pad = (x_max - x_min) * 0.05

fig_kl, (ax_fwd, ax_rev) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

_add_scatter(ax_fwd, kl_fwd, r"KL( Beta($\alpha$,$\beta$) $\|$ Uniform )")
_add_scatter(ax_rev, kl_rev, r"KL( Uniform $\|$ Beta($\alpha$,$\beta$) )")

for ax in (ax_fwd, ax_rev):
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, 1)

plt.tight_layout()
out_kl = os.path.join(OUT_DIR, f"scatter_kl_vs_acc_{TAG}.png")
plt.savefig(out_kl, dpi=150)
plt.close(fig_kl)
print(f"Saved -> {out_kl}")

print("\nAll plots saved.")
