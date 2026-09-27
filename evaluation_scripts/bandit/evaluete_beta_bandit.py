"""
Robustness evaluation under Beta-distributed arm means.

For each (alpha, beta) pair from {0.5, 1, 3, 5, 7}^2 (25 combinations) arm means
are drawn from Beta(alpha, beta) scaled to [min_value, max_value] instead of the
uniform distribution used during training.  Everything else (capacity sampling,
agent inference, metrics) is identical to evaluate_bandit_env.py.

Agent class is auto-detected from the model filename prefix:
  icmape_td3_*     -> MULTICPE_TD3
  icmape_vdn_*     -> MULTICPE_QLEARNING (VDN)
  icmape_indep_*   -> MULTICPE_QLEARNING (independent)
  icmape_uniform_* -> MULTICPE_QLEARNING (uniform baseline)

Output files (written to OUT_DIR)
----------------------------------
  results_beta_<tag>_a<alpha>_b<beta>.npz   -- per-combination raw data
  summary_beta_<tag>.npz                    -- 5x5 accuracy / stop-step grids
"""

import os
import itertools
import types

import numpy as np
import torch
from icmape.agents.bandit.sequential.multicpe    import MULTICPE_QLEARNING
from icmape.agents.bandit.sequential.multicpe_td3 import MULTICPE_TD3
from icmape.envs.bandit.builder import create_bandit_env, sample_env_beta
from icmape.envs.bandit.base   import VecBanditEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH    = "path/to/model.pkl"
TOTAL_SAMPLES = 10_000
BATCH_SIZE    = 1_000
OUT_DIR       = "/result_summary/bandit/beta_robustness"
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

ALPHA_BETA_VALUES = [0.5, 1, 3, 5, 7]

# ---------------------------------------------------------------------------
# Load model (once)
# ---------------------------------------------------------------------------
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
    raise ValueError(
        f"Cannot detect agent class from filename '{basename}'. "
        "Expected prefix 'icmape_td3', 'icmape_vdn', 'icmape_indep', or 'icmape_uniform'."
    )

print(f"Loading {AgentClass.__name__} from: {MODEL_PATH}")
agent = AgentClass.load(MODEL_PATH, device)
cfg   = agent.cfg.environment
K     = cfg.num_actions
H     = cfg.horizon
P     = cfg.num_players

seed = agent.cfg.experiment.seed
tag  = f"{method}_H{H}_K{K}_P{P}_seed{seed}"
print(f"  method={method}  H={H}  K={K}  P={P}  seed={seed}")
agent.i_net.eval()
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: build a custom sample_envs bound method for a given (alpha, beta)
# ---------------------------------------------------------------------------
def _make_sample_envs(env_cfg, alpha: float, beta_param: float):
    """Return a replacement for agent.sample_envs that uses Beta-sampled means."""
    def sample_envs(self, B: int, N: int) -> VecBanditEnv:
        envs = [sample_env_beta(env_cfg, alpha, beta_param) for _ in range(B)]
        repeated = [create_bandit_env(e.env_type, **e.parameters()) for e in envs for _ in range(N)]
        return VecBanditEnv(repeated)
    return sample_envs

# ---------------------------------------------------------------------------
# Sweep over all (alpha, beta) combinations
# ---------------------------------------------------------------------------
n_batches   = TOTAL_SAMPLES // BATCH_SIZE
combos      = list(itertools.product(ALPHA_BETA_VALUES, ALPHA_BETA_VALUES))
n_combos    = len(combos)

acc_grid      = np.zeros((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)))
stopstep_grid = np.zeros((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)))

for combo_idx, (alpha, beta_param) in enumerate(combos):
    print(f"\n[{combo_idx + 1}/{n_combos}]  alpha={alpha}  beta={beta_param}")

    # Monkey-patch sample_envs on the agent instance for this (alpha, beta)
    agent.sample_envs = types.MethodType(
        _make_sample_envs(agent.cfg.environment, alpha, beta_param),
        agent
    )

    all_correct       = []
    all_stopping_time = []

    for i in range(n_batches):
        print(f"  Batch {i + 1}/{n_batches} ...")
        traj = agent.sample_trajectories(BATCH_SIZE, N=1).to_numpy()
        all_correct.append(traj.correct_capacity.flatten())
        all_stopping_time.append(traj.stopping_time.flatten())

    all_correct       = np.concatenate(all_correct)
    all_stopping_time = np.concatenate(all_stopping_time)

    avg_acc  = float(all_correct.mean())
    std_acc  = float(all_correct.std())
    avg_step = float(all_stopping_time.mean())
    std_step = float(all_stopping_time.std())

    print(f"  Avg accuracy  : {avg_acc:.4f}  +/- {std_acc:.4f}")
    print(f"  Avg stop step : {avg_step:.2f}  +/- {std_step:.2f}")

    ai = ALPHA_BETA_VALUES.index(alpha)
    bi = ALPHA_BETA_VALUES.index(beta_param)
    acc_grid[ai, bi]      = avg_acc
    stopstep_grid[ai, bi] = avg_step

    # Per-combination npz
    combo_tag = f"a{alpha}_b{beta_param}".replace(".", "p")
    np.savez(
        os.path.join(OUT_DIR, f"results_beta_{tag}_{combo_tag}.npz"),
        all_correct=all_correct,
        all_stopping_time=all_stopping_time,
    )

# ---------------------------------------------------------------------------
# Summary npz
# ---------------------------------------------------------------------------
np.savez(
    os.path.join(OUT_DIR, f"summary_beta_{tag}.npz"),
    acc_grid=acc_grid,
    stopstep_grid=stopstep_grid,
    alpha_beta_values=np.array(ALPHA_BETA_VALUES),
    horizon=np.array(H),
    total_samples=np.array(TOTAL_SAMPLES),
)
print(f"\nSummary saved -> {os.path.join(OUT_DIR, f'summary_beta_{tag}.npz')}")

print("\nDone.")