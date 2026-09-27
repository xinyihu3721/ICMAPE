"""
Evaluate a list of saved models.

Agent class is auto-detected from the model filename prefix:
  icmape_td3_*     -> MULTICPE_TD3
  icmape_vdn_*     -> MULTICPE_QLEARNING (VDN)
  icmape_indep_*   -> MULTICPE_QLEARNING (independent)
  icmape_uniform_* -> MULTICPE_QLEARNING (uniform baseline)

Metrics
-------
- Average accuracy   : fraction of episodes where all arm capacities are
                       correctly identified at the effective stopping step.
- Average stop step  : mean stopping step (0-based).

Output files
------------
  action_heatmap_<tag>.png   -- heatmap of simultaneous pull distribution
  results_<tag>.npz          -- all_correct, all_stopping_time, co_pull_avg + metadata
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from icmape.agents.bandit.sequential.multicpe import MULTICPE_QLEARNING
from icmape.agents.bandit.sequential.multicpe_td3 import MULTICPE_TD3

# ---------------------------------------------------------------------------
# Configuration -- set MODEL_PATHS, everything else is auto-detected
# ---------------------------------------------------------------------------
MODEL_PATHS = [
    "path/to/model_1.pkl",
    # "path/to/model_2.pkl",
]

TOTAL_SAMPLES = 10_000
BATCH_SIZE    = 1_000    # tune for VRAM
OUT_DIR       = "results_summary/bandit/A5P3" # change this to the desired output directory
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Process each model in order
# ---------------------------------------------------------------------------
for model_idx, MODEL_PATH in enumerate(MODEL_PATHS):
    print(f"\n[{model_idx + 1}/{len(MODEL_PATHS)}] Processing: {MODEL_PATH}")

    # -----------------------------------------------------------------------
    # Auto-detect agent class from filename
    # -----------------------------------------------------------------------
    basename = os.path.basename(MODEL_PATH)
    if basename.startswith("icmape_td3"):
        AgentClass = MULTICPE_TD3
        method = "icmape_td3"
    elif basename.startswith("icmape_vdn"):
        AgentClass = MULTICPE_QLEARNING
        method = "icmape_vdn"
    elif basename.startswith("icmape_indep"):
        AgentClass = MULTICPE_QLEARNING
        method = "icmape_indep"
    elif basename.startswith("icmape_uniform"):
        AgentClass = MULTICPE_QLEARNING
        method = "icmape_uniform"
    else:
        print(f"  WARNING: Cannot detect agent class from '{basename}'. Skipping.")
        continue

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"  Loading {AgentClass.__name__} ...")
    agent = AgentClass.load(MODEL_PATH, device)
    cfg   = agent.cfg.environment
    K     = cfg.num_actions
    H     = cfg.horizon
    P     = cfg.num_players

    seed = agent.cfg.experiment.seed
    tag  = f"{method}_H{H}_K{K}_P{P}_seed{seed}"

    print(f"  method={method}  H={H}  K={K}  P={P}  seed={seed}")

    agent.i_net.eval()

    # -----------------------------------------------------------------------
    # Batched evaluation -- single pass
    # -----------------------------------------------------------------------
    n_batches         = TOTAL_SAMPLES // BATCH_SIZE
    all_correct       = []
    all_stopping_time = []
    co_pull_counts    = np.zeros((K, P + 1), dtype=np.float64)

    for i in range(n_batches):
        print(f"  Batch {i + 1}/{n_batches} ...")
        traj = agent.sample_trajectories(BATCH_SIZE, N=1).to_numpy()

        all_correct.append(traj.correct_capacity.flatten())       # (B,)
        all_stopping_time.append(traj.stopping_time.flatten())    # (B,)

        # co-pull histogram -- only steps up to stopping time
        acts   = traj.exp_actions[:, 0, :, :]                    # (B, H, P)
        stop_h = traj.stopping_time[:, 0].astype(int)            # (B,)
        step_mask = np.arange(H)[None, :] <= stop_h[:, None]     # (B, H) bool
        for k in range(K):
            pulls = (acts == k).sum(axis=-1)                      # (B, H)
            for m in range(P + 1):
                co_pull_counts[k, m] += ((pulls == m) & step_mask).sum()

    all_correct       = np.concatenate(all_correct)
    all_stopping_time = np.concatenate(all_stopping_time)

    # -----------------------------------------------------------------------
    # Scalar metrics
    # -----------------------------------------------------------------------
    avg_acc  = all_correct.mean()
    std_acc  = all_correct.std()
    avg_step = all_stopping_time.mean()
    std_step = all_stopping_time.std()

    print()
    print(f"=== Evaluation results [{tag}]  ({TOTAL_SAMPLES:,} samples) ===")
    print(f"  Avg accuracy  (all arms correct) : {avg_acc:.4f}  ({avg_acc * 100:.2f}%)  +/- {std_acc:.4f}")
    print(f"  Avg stop step (0-based)          : {avg_step:.2f}  +/- {std_step:.2f}")
    print(f"  Avg stop step (1-based)          : {avg_step + 1:.2f}")

    # -----------------------------------------------------------------------
    # Heatmap
    # -----------------------------------------------------------------------
    co_pull_avg      = co_pull_counts / TOTAL_SAMPLES   # (K, P+1): avg steps/episode
    co_pull_avg_plot = co_pull_avg[:, 1:]               # exclude m=0 from visualization

    heatmap_path = os.path.join(OUT_DIR, f"action_heatmap_{tag}.png")
    fig, ax = plt.subplots(figsize=(P * 0.8 + 2, K * 1.5 + 1))
    sns.heatmap(
        co_pull_avg_plot,                           # (K, P) -> y = arm (k=0..K-1), x = m (1..P)
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        xticklabels=list(range(1, P + 1)),
        yticklabels=[f"Arm {k}" for k in range(K)],
        cbar_kws={"label": "Avg steps per episode"},
    )
    ax.set_xlabel("# players pulling arm simultaneously (m)")
    ax.set_ylabel("Arm")
    ax.set_title(
        f"Simultaneous pull distribution  [{tag}]\n"
        f"({TOTAL_SAMPLES:,} samples)  acc={avg_acc * 100:.1f}%  avg_stop={avg_step + 1:.1f}"
    )
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved -> {heatmap_path}")

    # -----------------------------------------------------------------------
    # Save data
    # -----------------------------------------------------------------------
    data_path = os.path.join(OUT_DIR, f"results_{tag}.npz")
    np.savez(
        data_path,
        all_correct=all_correct,               # (TOTAL_SAMPLES,) float, 1=all arms correct
        all_stopping_time=all_stopping_time,   # (TOTAL_SAMPLES,) float, 0-based stop step
        co_pull_avg=co_pull_avg,               # (K, P+1) avg steps/episode per (arm, m)
        method=np.array(method),
        horizon=np.array(H),
        num_actions=np.array(K),
        num_players=np.array(P),
        seed=np.array(seed),
        total_samples=np.array(TOTAL_SAMPLES),
    )
    print(f"  Data  saved -> {data_path}")

print("\nAll models processed.")
