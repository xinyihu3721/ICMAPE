"""
Robustness evaluation under Beta-distributed distractor intensity.

The two-source robot env trains with the distractor's intensity drawn from
Uniform[intensity_min, intensity_max] (= scaled Beta(1,1)).  This script
sweeps over (alpha, beta) pairs and replaces that uniform draw with
Beta(alpha, beta) scaled to the same [intensity_min, intensity_max] interval,
evaluating whether the agent's accuracy degrades under different intensity priors.

Agent class is auto-detected from model filename prefix (same as result_robot_explore.py).

Output files (per model, written to OUT_DIR/<tag>/)
---------------------------------------------------
  results_beta_<tag>_a<alpha>_b<beta>.npz   -- per-combination raw data
  summary_beta_<tag>.npz                    -- acc / stop-step grids + metadata
"""

from __future__ import annotations

import os
import itertools
import types

import numpy as np
import torch

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

TOTAL_SAMPLES     = 10_000
BATCH_SIZE        = 1_000
ALPHA_BETA_VALUES = [0.5, 1, 3, 5, 7]
OUT_DIR           = "result_summary/robot/beta_robustness" # change this to the desired output directory
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Agent class detection (same logic as result_robot_explore.py)
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
        f"Cannot detect agent class from filename '{basename}'. "
        "Expected prefix 'ROBOT_MULTICPE_SAC', 'ROBOT_MULTICPE_TD3', "
        "or 'ROBOT_MULTICPE_QLEARNING'."
    )


def detect_method(agent) -> str:
    if isinstance(agent, MULTICPE_SAC_ROBOT):
        return "SAC"
    elif isinstance(agent, MULTICPE_TD3_ROBOT):
        return "TD3"
    elif isinstance(agent, MULTICPE_QLEARNING_ROBOT):
        if getattr(agent, "uniform_action_only", False):
            return "UNIFORM"
        elif getattr(agent, "VDN_mode", False):
            return "VDN"
        return "INDEP"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Beta-intensity patch helpers
# ---------------------------------------------------------------------------

def _make_sample_envs_beta(
    original_sample_envs,
    alpha: float,
    beta_param: float,
    intensity_min: float,
    intensity_max: float,
):
    """Return a replacement sample_envs that patches the env to use Beta intensity."""
    scale = float(intensity_max - intensity_min)
    offset = float(intensity_min)

    def patched_sample_envs(self, B: int, N: int, seed_offset: int = 0):
        envs = original_sample_envs(B, N, seed_offset=seed_offset)

        # Only BatchedRobotEnv2 (two-source) has _signal_rcs / _distractor_intensities.
        if not hasattr(envs, "_signal_rcs"):
            raise RuntimeError(
                "Beta-intensity robustness test requires a two-source environment "
                "(BatchedRobotEnv2).  Current env has no '_signal_rcs' attribute."
            )

        _orig_refresh = envs._refresh_signals_and_smoke  # bound method

        def beta_refresh():
            # Run original: samples signal/distractor positions + uniform intensities + smoke maps.
            _orig_refresh()
            # Override intensities: Beta(alpha, beta) scaled to [intensity_min, intensity_max].
            envs._distractor_intensities = np.array(
                [envs._rngs[b].beta(alpha, beta_param) * scale + offset
                 for b in range(envs.batch_size)],
                dtype=np.float32,
            )
            # Recompute smoke maps with the new intensities.
            r1s, c1s = envs._signal_rcs[:, 0, 0], envs._signal_rcs[:, 0, 1]
            r2s, c2s = envs._signal_rcs[:, 1, 0], envs._signal_rcs[:, 1, 1]
            smoke1 = envs._smoke_lookup[r1s, c1s]
            smoke2 = envs._distractor_intensities[:, None, None] * envs._smoke_lookup[r2s, c2s]
            dist1  = envs._dist_lookup[r1s, c1s]
            dist2  = envs._dist_lookup[r2s, c2s]
            envs._smoke_maps = np.where(dist1 <= dist2, smoke1, smoke2)

        # Assign as instance attribute so reset() calls beta_refresh() instead.
        envs._refresh_signals_and_smoke = beta_refresh
        return envs

    return patched_sample_envs


# ---------------------------------------------------------------------------
# Single-batch eval (mirrors result_robot_explore.eval_batch)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_batch(agent, B: int, seed_offset: int):
    """Return (correct [B], stop_t [B], intensities [B]) for one batch."""
    traj = agent.sample_trajectories(B, N=1, seed_offset=seed_offset)
    H = agent.cfg.environment.horizon

    Bn, Nn = traj.stopping_time.shape
    end_t  = traj.stopping_time.to(dtype=torch.long).clamp(min=0, max=H - 1)
    b_idx  = torch.arange(Bn, device=agent.device).unsqueeze(1).expand(Bn, Nn)
    n_idx  = torch.arange(Nn, device=agent.device).unsqueeze(0).expand(Bn, Nn)

    pred_idx = traj.exp_signal_location_idx[b_idx, n_idx, end_t, 0].to(dtype=torch.long)
    true_idx = traj.true_signal_idx.to(dtype=torch.long)

    correct = (pred_idx == true_idx).to(dtype=torch.float32).flatten().cpu().numpy()
    stop_t  = traj.stopping_time.flatten().cpu().numpy()

    # Recover per-episode distractor intensities from signal_maps.
    # signal_maps shape: [B, G, G]; true signal is always 1.0, distractor is the smaller value.
    G = traj.signal_maps.shape[-1]
    flat = traj.signal_maps.reshape(Bn, -1).cpu().numpy()       # [B, G*G]
    top2 = np.partition(flat, -2, axis=-1)[:, -2]             # second-largest per row
    intensities = top2.astype(np.float32)                     # [B]

    return correct, stop_t, intensities


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, device: torch.device) -> None:
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

    intensity_min = float(getattr(cfg, "distractor_intensity_min", 0.0))
    intensity_max = float(getattr(cfg, "distractor_intensity_max", 0.7))
    print(f"  intensity range: [{intensity_min}, {intensity_max}]  (training: Uniform)")

    out_dir = os.path.join(OUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    n_batches = max(1, TOTAL_SAMPLES // BATCH_SIZE)
    combos    = list(itertools.product(ALPHA_BETA_VALUES, ALPHA_BETA_VALUES))
    n_combos  = len(combos)

    acc_grid      = np.zeros((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)))
    stopstep_grid = np.zeros((len(ALPHA_BETA_VALUES), len(ALPHA_BETA_VALUES)))

    # Capture original sample_envs once before any patching.
    original_sample_envs = agent.sample_envs

    for combo_idx, (alpha, beta_param) in enumerate(combos):
        print(f"\n[{combo_idx + 1}/{n_combos}]  alpha={alpha}  beta={beta_param}")

        agent.sample_envs = types.MethodType(
            _make_sample_envs_beta(
                original_sample_envs, alpha, beta_param, intensity_min, intensity_max
            ),
            agent,
        )

        all_correct    = []
        all_stop_t     = []
        all_intensities = []

        for i in range(n_batches):
            print(f"  Batch {i + 1}/{n_batches} ...")
            correct, stop_t, intensities = eval_batch(agent, BATCH_SIZE, seed_offset=i)
            all_correct.append(correct)
            all_stop_t.append(stop_t)
            all_intensities.append(intensities)

        all_correct     = np.concatenate(all_correct)
        all_stop_t      = np.concatenate(all_stop_t)
        all_intensities = np.concatenate(all_intensities)

        avg_acc  = float(all_correct.mean())
        avg_step = float(all_stop_t.mean())
        print(f"  Avg accuracy  : {avg_acc:.4f}")
        print(f"  Avg stop step : {avg_step:.2f}")
        print(f"  Avg intensity : {all_intensities.mean():.3f}  "
              f"(expected Beta({alpha},{beta_param}) mean "
              f"= {alpha/(alpha+beta_param) * (intensity_max - intensity_min) + intensity_min:.3f})")

        ai = ALPHA_BETA_VALUES.index(alpha)
        bi = ALPHA_BETA_VALUES.index(beta_param)
        acc_grid[ai, bi]      = avg_acc
        stopstep_grid[ai, bi] = avg_step

        combo_tag = f"a{str(alpha).replace('.','p')}_b{str(beta_param).replace('.','p')}"
        np.savez(
            os.path.join(out_dir, f"results_beta_{tag}_{combo_tag}.npz"),
            all_correct=all_correct,
            all_stopping_time=all_stop_t,
            all_intensities=all_intensities,
        )

    # Restore original sample_envs
    agent.sample_envs = original_sample_envs

    summary_path = os.path.join(out_dir, f"summary_beta_{tag}.npz")
    np.savez(
        summary_path,
        acc_grid=acc_grid,
        stopstep_grid=stopstep_grid,
        method=np.array(method),
        horizon=np.array(H),
        seed=np.array(seed),
        intensity_min=np.array(intensity_min),
        intensity_max=np.array(intensity_max),
    )
    print(f"\n  Summary saved -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    for path in MODEL_PATHS:
        evaluate_model(path, device)
    print("\nAll models evaluated.")


if __name__ == "__main__":
    main()
