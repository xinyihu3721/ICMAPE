"""
Robustness evaluation under Dirichlet-Categorical arm capacities.

For each concentration  in CONCENTRATION_VALUES, TOTAL_SAMPLES episodes are
run.  Each episode draws theta ~ Dirichlet(alpha*1_P) then samples arm capacities from
Categorical(theta).  Per-episode total variation distance and inference probability
are recorded alongside accuracy.

Agent class is auto-detected from the model filename prefix:
  icmape_td3_*     -> MULTICPE_TD3
  icmape_vdn_*     -> MULTICPE_QLEARNING (VDN)
  icmape_indep_*   -> MULTICPE_QLEARNING (independent)
  icmape_uniform_* -> MULTICPE_QLEARNING (uniform baseline)

Output files (written to OUT_DIR)
----------------------------------
  results_dirichlet_<tag>_a<alpha>.npz  -- per-concentration raw data
"""

import os
import types

import numpy as np
import torch

from icmape.agents.bandit.sequential.multicpe    import MULTICPE_QLEARNING
from icmape.agents.bandit.sequential.multicpe_td3 import MULTICPE_TD3
from icmape.envs.bandit.builder import create_bandit_env, sample_env, sample_env_with_cap_theta
from icmape.envs.bandit.base   import VecBanditEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH    = "path/to/model.pkl"
TOTAL_SAMPLES = 10_000
BATCH_SIZE    = 1_000
OUT_DIR       = "/result_summary/bandit/dirichlet_robustness"
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

CONCENTRATION_VALUES = [0.1, 0.5, 1, 5, 10]

# ---------------------------------------------------------------------------
# Load model
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
# DirichletEnvSampler -- tracks per-episode theta and TV distance
# ---------------------------------------------------------------------------
class DirichletEnvSampler:
    def __init__(self, env_cfg, P: int, concentration: float):
        self.env_cfg       = env_cfg
        self.P             = P
        self.concentration = concentration
        self.last_tv       = None  # shape (B,) after each call

    def sample_envs(self, agent_self, B: int, N: int) -> VecBanditEnv:
        thetas = np.random.dirichlet(self.concentration * np.ones(self.P), size=B)
        self.last_tv = 0.5 * np.sum(np.abs(thetas - 1.0 / self.P), axis=1)
        envs     = [sample_env_with_cap_theta(self.env_cfg, thetas[i]) for i in range(B)]
        repeated = [create_bandit_env(e.env_type, **e.parameters()) for e in envs for _ in range(N)]
        return VecBanditEnv(repeated)

# ---------------------------------------------------------------------------
# Baseline: i.i.d. Uniform capacity (training prior)
# ---------------------------------------------------------------------------
def _uniform_sample_envs(self, B: int, N: int) -> VecBanditEnv:
    envs     = [sample_env(cfg) for _ in range(B)]
    repeated = [create_bandit_env(e.env_type, **e.parameters()) for e in envs for _ in range(N)]
    return VecBanditEnv(repeated)

print("\n[Baseline]  i.i.d. Uniform capacity (training prior)")
agent.sample_envs = types.MethodType(_uniform_sample_envs, agent)
n_batches = TOTAL_SAMPLES // BATCH_SIZE

base_correct = []
for i in range(n_batches):
    print(f"  Batch {i+1}/{n_batches} ...")
    traj = agent.sample_trajectories(BATCH_SIZE, N=1).to_numpy()
    base_correct.append(traj.correct_capacity.flatten())
base_correct = np.concatenate(base_correct)
base_acc     = float(base_correct.mean())
print(f"  Avg accuracy : {base_acc:.4f}")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for conc in CONCENTRATION_VALUES:
    print(f"\n[alpha={conc}]")
    sampler = DirichletEnvSampler(agent.cfg.environment, P, conc)
    agent.sample_envs = types.MethodType(sampler.sample_envs, agent)

    all_correct, all_inf_prob, all_tv = [], [], []

    for i in range(n_batches):
        print(f"  Batch {i+1}/{n_batches} ...")
        traj = agent.sample_trajectories(BATCH_SIZE, N=1).to_numpy()
        correct = traj.correct_capacity.flatten()
        stop    = traj.stopping_time.flatten().astype(int)
        inf_prob_batch = traj.rew_infer_capacity[np.arange(BATCH_SIZE), 0, stop.clip(0, H - 1)]
        all_correct.append(correct)
        all_inf_prob.append(inf_prob_batch)
        all_tv.append(sampler.last_tv.copy())

    all_correct = np.concatenate(all_correct)
    inf_prob    = np.concatenate(all_inf_prob)
    tv          = np.concatenate(all_tv)

    avg_acc = float(all_correct.mean())
    print(f"  Avg accuracy  : {avg_acc:.4f}")
    print(f"  E[inf_prob]   : {inf_prob.mean():.4f}")
    print(f"  E[TV]         : {tv.mean():.4f}")

    a_str = str(conc).replace(".", "p")
    np.savez(
        os.path.join(OUT_DIR, f"results_dirichlet_{tag}_a{a_str}.npz"),
        inf_prob=inf_prob,
        all_correct=all_correct,
        tv=tv,
        base_acc=np.array(base_acc),
    )
    print(f"  Saved -> results_dirichlet_{tag}_a{a_str}.npz")

print("\nDone.")
