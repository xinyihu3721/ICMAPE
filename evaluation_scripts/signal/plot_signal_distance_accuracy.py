"""
Evaluate and plot accuracy vs. exact Manhattan distance between the true signal
and distractor locations.

Instead of sampling uniformly and grouping post-hoc, this script **fixes the
distance** for each batch: the env's valid (true-signal, distractor) pair pool
is temporarily restricted to only pairs whose Manhattan distance equals the
target value, giving exactly N_PER_DIST clean samples per distance.

Output (per model)
------------------
  <OUT_DIR>/distance_accuracy_<tag>.npz   -- raw data per distance
  <OUT_DIR>/distance_accuracy_<tag>.png   -- bar chart accuracy vs distance
"""

from __future__ import annotations

import os
import math
import types

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from icmape.agents.robot.sequential.multicpe_sac_robot import MULTICPE_SAC_ROBOT
from icmape.agents.robot.sequential.multicpe_td3_robot import MULTICPE_TD3_ROBOT
from icmape.agents.robot.sequential.multicpe_qlearning import MULTICPE_QLEARNING_ROBOT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATHS = [
    "path/to/model_1.pkl",
    # "path/to/model_2.pkl",
]

# Manhattan distances to evaluate.  None -> auto-discover from the env
# (every integer from min pairwise distance to max in steps of DIST_STEP).
TARGET_DISTANCES: list[int] | None = None
DIST_STEP   = 3      # step size when auto-discovering distances
N_PER_DIST  = 10000   # episodes per distance
BATCH_SIZE  = 1000    # episodes per inference batch (tune for VRAM)
OUT_DIR     = "result_summary/robot/explore" # change this to the desired output directory
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Agent class / method detection
# ---------------------------------------------------------------------------

def detect_agent_class(model_path: str):
    basename = os.path.basename(model_path)
    if basename.startswith("ROBOT_MULTICPE_SAC"):
        return MULTICPE_SAC_ROBOT
    elif basename.startswith("ROBOT_MULTICPE_TD3"):
        return MULTICPE_TD3_ROBOT
    elif basename.startswith("ROBOT_MULTICPE_QLEARNING"):
        return MULTICPE_QLEARNING_ROBOT
    raise ValueError(
        f"Cannot detect agent class from '{basename}'. "
        "Expected prefix ROBOT_MULTICPE_SAC / TD3 / QLEARNING."
    )


def detect_method(agent) -> str:
    if isinstance(agent, MULTICPE_SAC_ROBOT):
        return "SAC"
    elif isinstance(agent, MULTICPE_TD3_ROBOT):
        return "TD3"
    elif isinstance(agent, MULTICPE_QLEARNING_ROBOT):
        if getattr(agent, "uniform_action_only", False):
            return "UNIFORM"
        if getattr(agent, "VDN_mode", False):
            return "VDN"
        return "INDEP"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Fixed-distance patch
# ---------------------------------------------------------------------------

def _make_sample_envs_fixed_dist(original_sample_envs, target_dist: int):
    """
    Return a patched sample_envs that restricts the (true signal, distractor)
    pair pool to pairs whose Manhattan distance is exactly target_dist.

    The patch temporarily replaces _valid_signal_locs / _valid_distractor_locs /
    _num_valid_pairs on the env instance around each call to the original
    _refresh_signals_and_smoke, so all other logic (intensity sampling, smoke maps,
    observations) is unchanged.
    """
    def patched_sample_envs(self, B: int, N: int, seed_offset: int = 0):
        envs = original_sample_envs(B, N, seed_offset=seed_offset)

        if not hasattr(envs, "_signal_rcs"):
            raise RuntimeError(
                "Fixed-distance sampling requires BatchedRobotEnv2 (true signal "
                "vs distractor). Current env has no '_signal_rcs' attribute."
            )

        # Precompute filtered pair arrays (once per env instance).
        p1s   = envs._valid_signal_locs    # [num_pairs, 2]
        p2s   = envs._valid_distractor_locs
        dists = np.abs(p1s[:, 0] - p2s[:, 0]) + np.abs(p1s[:, 1] - p2s[:, 1])
        mask  = dists == target_dist
        fp1s  = p1s[mask]
        fp2s  = p2s[mask]
        n_filtered = len(fp1s)

        if n_filtered == 0:
            raise ValueError(
                f"No valid true-signal/distractor pairs with Manhattan distance "
                f"exactly {target_dist}. Available range: [{dists.min()}, {dists.max()}]."
            )

        orig_refresh = envs._refresh_signals_and_smoke

        def fixed_dist_refresh():
            # Temporarily restrict the pair pool, call original, then restore.
            saved = (envs._valid_signal_locs, envs._valid_distractor_locs, envs._num_valid_pairs)
            envs._valid_signal_locs       = fp1s
            envs._valid_distractor_locs   = fp2s
            envs._num_valid_pairs         = n_filtered
            orig_refresh()
            envs._valid_signal_locs, envs._valid_distractor_locs, envs._num_valid_pairs = saved

        envs._refresh_signals_and_smoke = fixed_dist_refresh
        return envs

    return patched_sample_envs


# ---------------------------------------------------------------------------
# Single-batch eval
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_batch(agent, B: int, seed_offset: int):
    """Return (correct [B], stop_t [B]) for one batch."""
    traj = agent.sample_trajectories(B, N=1, seed_offset=seed_offset)
    H    = agent.cfg.environment.horizon

    Bn, Nn = traj.stopping_time.shape
    end_t  = traj.stopping_time.to(dtype=torch.long).clamp(min=0, max=H - 1)
    b_idx  = torch.arange(Bn, device=agent.device).unsqueeze(1).expand(Bn, Nn)
    n_idx  = torch.arange(Nn, device=agent.device).unsqueeze(0).expand(Bn, Nn)

    pred_idx = traj.exp_signal_location_idx[b_idx, n_idx, end_t, 0].to(dtype=torch.long)
    true_idx = traj.true_signal_idx.to(dtype=torch.long)

    correct = (pred_idx == true_idx).to(dtype=torch.float32).flatten().cpu().numpy()
    stop_t  = traj.stopping_time.flatten().cpu().numpy()
    return correct, stop_t


# ---------------------------------------------------------------------------
# Distance range discovery
# ---------------------------------------------------------------------------

def _discover_distances(agent, step: int) -> list[int]:
    """Return all integer distances achievable by the env, sampled every `step`."""
    envs = agent.sample_envs(1, 1, seed_offset=0)
    if not hasattr(envs, "_valid_signal_locs"):
        raise RuntimeError("Distance discovery requires BatchedRobotEnv2.")
    p1s   = envs._valid_signal_locs
    p2s   = envs._valid_distractor_locs
    dists = np.abs(p1s[:, 0] - p2s[:, 0]) + np.abs(p1s[:, 1] - p2s[:, 1])
    d_min, d_max = int(dists.min()), int(dists.max())
    return list(range(d_min, d_max + 1, step))


# ---------------------------------------------------------------------------
# Per-model evaluate + plot
# ---------------------------------------------------------------------------
COLORS = ["#3C9BC9", "#FC757B", "#A6CA68", "#FFB83D", "#9B59B6", "#E67E22"]


def evaluate_and_plot(model_path: str, device: torch.device) -> None:
    if not os.path.exists(model_path):
        print(f"[SKIP] Model not found: {model_path}")
        return

    AgentClass = detect_agent_class(model_path)
    print(f"\nLoading {AgentClass.__name__} from:\n  {model_path}")
    agent = AgentClass.load(model_path, device)
    agent.i_net.eval()

    method = detect_method(agent)
    cfg    = agent.cfg.environment
    H      = cfg.horizon
    seed   = agent.cfg.experiment.seed
    tag    = f"{method}_H{H}_seed{seed}"
    print(f"  method={method}  H={H}  seed={seed}")

    target_distances = TARGET_DISTANCES
    if target_distances is None:
        target_distances = _discover_distances(agent, DIST_STEP)
        print(f"  Auto-discovered {len(target_distances)} distances: "
              f"{target_distances[0]}..{target_distances[-1]} (step={DIST_STEP})")

    n_batches = max(1, math.ceil(N_PER_DIST / BATCH_SIZE))
    actual_n  = n_batches * BATCH_SIZE

    original_sample_envs = agent.sample_envs

    result_dists  = []
    result_means  = []
    result_ses    = []
    result_steps  = []

    for di, dist in enumerate(target_distances):
        print(f"  [{di + 1}/{len(target_distances)}]  distance={dist} ...")
        try:
            agent.sample_envs = types.MethodType(
                _make_sample_envs_fixed_dist(original_sample_envs, dist),
                agent,
            )
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue

        all_correct = []
        all_stop_t  = []
        for i in range(n_batches):
            correct, stop_t = eval_batch(agent, BATCH_SIZE, seed_offset=di * n_batches + i)
            all_correct.append(correct)
            all_stop_t.append(stop_t)

        all_correct = np.concatenate(all_correct)
        all_stop_t  = np.concatenate(all_stop_t)

        p   = float(all_correct.mean())
        se  = float(np.sqrt(p * (1 - p) / len(all_correct)))
        avg_step = float(all_stop_t.mean())

        result_dists.append(dist)
        result_means.append(p)
        result_ses.append(se)
        result_steps.append(avg_step)
        print(f"    acc={p:.3f} +/- {se:.3f}  avg_step={avg_step:.1f}")
        print(f"    n={len(all_correct)}")

    agent.sample_envs = original_sample_envs  # restore

    if not result_dists:
        print("  No valid distances -- nothing to save or plot.")
        return

    dists_arr = np.array(result_dists)
    means_arr = np.array(result_means)
    ses_arr   = np.array(result_ses)
    steps_arr = np.array(result_steps)

    # ---- Save ---------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, f"distance_accuracy_{tag}.npz")
    np.savez(
        npz_path,
        distances=dists_arr,
        mean_accuracy=means_arr,
        se_accuracy=ses_arr,
        mean_stop_step=steps_arr,
        n_per_dist=np.array(actual_n),
        method=np.array(method),
        horizon=np.array(H),
        seed=np.array(seed),
    )
    print(f"  Data saved -> {npz_path}")

    # ---- Plot ---------------------------------------------------------------
    fig, (ax_acc, ax_step) = plt.subplots(
        2, 1, figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.10},
        sharex=True,
    )

    color = COLORS[0]
    Z95   = 1.96

    # Top: bar chart accuracy +/- 95% CI
    ax_acc.bar(
        dists_arr, means_arr,
        yerr=Z95 * ses_arr, capsize=3,
        color=color, alpha=0.70, width=max(1, DIST_STEP) * 0.75,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
        label=f"{method} (n={actual_n:,} per dist)",
    )
    # ax_acc.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
    #                label="chance (0.5)")
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_ylim(-0.02, 1.08)
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_acc.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax_acc.legend(fontsize=9, framealpha=0.85)
    ax_acc.set_title(
        "Accuracy vs. Manhattan Distance (True Signal vs Distractor)",
        fontsize=11, fontweight="bold", pad=7,
    )

    # Bottom: avg stop step
    ax_step.plot(dists_arr, steps_arr + 1, marker="o", markersize=4,
                 linewidth=1.6, color="#888888")
    ax_step.set_ylabel("Avg stop", fontsize=9, color="#444444")
    ax_step.tick_params(axis="y", labelsize=8)
    ax_step.set_ylim(bottom=0)
    ax_step.grid(axis="y", linewidth=0.4, alpha=0.35)

    ax_step.set_xlabel(
        "Manhattan distance (true signal <-> distractor)", fontsize=15
    )

    plot_path = os.path.join(OUT_DIR, f"distance_accuracy_{tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    for path in MODEL_PATHS:
        evaluate_and_plot(path, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
