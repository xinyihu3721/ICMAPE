"""
Evaluate saved robot signal-detector exploration models.

Supports MULTICPE_SAC_ROBOT, MULTICPE_TD3_ROBOT, and MULTICPE_QLEARNING_ROBOT.
Agent class is auto-detected from the model filename prefix:
  ROBOT_MULTICPE_SAC_*       -> MULTICPE_SAC_ROBOT         (SAC)
  ROBOT_MULTICPE_TD3_*       -> MULTICPE_TD3_ROBOT          (TD3)
  ROBOT_MULTICPE_QLEARNING_* -> MULTICPE_QLEARNING_ROBOT
                                  VDN_mode=True  -> VDN
                                  uniform_action_only=True -> UNIFORM
                                  else           -> INDEP

Metrics
-------
- Average accuracy  : fraction of episodes where argmax of the inference
                      distribution at the stopping step equals the true
                      signal location.
- Average stop step : mean stopping step (0-based).

Output files (per model)
------------------------
  results_<tag>.npz -- all_correct, all_stopping_time + metadata
"""

from __future__ import annotations

import os
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

TOTAL_SAMPLES = 10_000      # total evaluation episodes per model
BATCH_SIZE    = 1000        # episodes per batch (tune for VRAM)
OUT_DIR       = "result_summary/robot/explore" # change this to the desired output directory
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Agent class detection
# ---------------------------------------------------------------------------

def detect_agent_class(model_path: str):
    """Return (AgentClass, cfg_data) detected from filename prefix."""
    basename = os.path.basename(model_path)
    if basename.startswith("ROBOT_MULTICPE_SAC"):
        return MULTICPE_SAC_ROBOT
    elif basename.startswith("ROBOT_MULTICPE_TD3"):
        return MULTICPE_TD3_ROBOT
    elif basename.startswith("ROBOT_MULTICPE_QLEARNING"):
        return MULTICPE_QLEARNING_ROBOT
    else:
        raise ValueError(
            f"Cannot detect agent class from filename '{basename}'. "
            "Expected prefix 'ROBOT_MULTICPE_SAC', 'ROBOT_MULTICPE_TD3', "
            "or 'ROBOT_MULTICPE_QLEARNING'."
        )


def detect_method(agent) -> str:
    """Return a human-readable method name from the loaded agent."""
    if isinstance(agent, MULTICPE_SAC_ROBOT):
        return "SAC"
    elif isinstance(agent, MULTICPE_TD3_ROBOT):
        return "TD3"
    elif isinstance(agent, MULTICPE_QLEARNING_ROBOT):
        if getattr(agent, "uniform_action_only", False):
            return "UNIFORM"
        elif getattr(agent, "VDN_mode", False):
            return "VDN"
        else:
            return "INDEP"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Single-batch evaluation helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_batch(agent, B: int, seed_offset: int):
    """Return (all_correct [B], stopping_time [B], signal_positions [B, num_sources, 2] or None)."""
    traj = agent.sample_trajectories(B, N=1, seed_offset=seed_offset)
    H = agent.cfg.environment.horizon

    Bn, Nn = traj.stopping_time.shape
    end_t = traj.stopping_time.to(dtype=torch.long).clamp(min=0, max=H - 1)  # [B, 1]
    b_idx = torch.arange(Bn, device=agent.device).unsqueeze(1).expand(Bn, Nn)
    n_idx = torch.arange(Nn, device=agent.device).unsqueeze(0).expand(Bn, Nn)

    pred_signal_idx = traj.exp_signal_location_idx[b_idx, n_idx, end_t, 0].to(dtype=torch.long)  # [B, 1]
    true_signal_idx = traj.true_signal_idx.to(dtype=torch.long)                                    # [B, 1]

    correct = (pred_signal_idx == true_signal_idx).to(dtype=torch.float32).flatten()  # [B]
    stop_t  = traj.stopping_time.flatten()                                              # [B]

    signal_pos = (
        traj.signal_positions.cpu().numpy()
        if traj.signal_positions is not None else None
    )

    return correct.cpu().numpy(), stop_t.cpu().numpy(), signal_pos


# ---------------------------------------------------------------------------
# Main evaluation loop
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

    n_batches  = max(1, TOTAL_SAMPLES // BATCH_SIZE)
    actual_n   = n_batches * BATCH_SIZE
    all_correct          = []
    all_stopping_time    = []
    all_signal_positions = []

    for i in range(n_batches):
        print(f"  Batch {i + 1}/{n_batches} ...")
        correct, stop_t, signal_pos = eval_batch(agent, BATCH_SIZE, seed_offset=i)
        all_correct.append(correct)
        all_stopping_time.append(stop_t)
        if signal_pos is not None:
            all_signal_positions.append(signal_pos)

    all_correct          = np.concatenate(all_correct)
    all_stopping_time    = np.concatenate(all_stopping_time)
    all_signal_positions = np.concatenate(all_signal_positions, axis=0) if all_signal_positions else None

    avg_acc   = all_correct.mean()
    std_acc   = all_correct.std()
    avg_step  = all_stopping_time.mean()
    std_step  = all_stopping_time.std()

    print()
    print(f"=== Evaluation results [{tag}]  ({actual_n:,} samples) ===")
    print(f"  Avg accuracy  (signal location correct) : {avg_acc:.4f}  ({avg_acc * 100:.2f}%)  +/- {std_acc:.4f}")
    print(f"  Avg stop step (0-based)                : {avg_step:.2f}  +/- {std_step:.2f}")
    print(f"  Avg stop step (1-based)                : {avg_step + 1:.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    data_path = os.path.join(OUT_DIR, f"results_{tag}.npz")
    save_kwargs = dict(
        all_correct=all_correct,
        all_stopping_time=all_stopping_time,
        method=np.array(method),
        horizon=np.array(H),
        seed=np.array(seed),
        total_samples=np.array(actual_n),
    )
    if all_signal_positions is not None:
        save_kwargs["signal_positions"] = all_signal_positions
    np.savez(data_path, **save_kwargs)
    print(f"  Data saved -> {data_path}")
    if all_signal_positions is not None:
        print(f"  signal_positions shape: {all_signal_positions.shape}  (episodes x sources x row/col)")


def main() -> None:
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Models to evaluate: {len(MODEL_PATHS)}")

    for path in MODEL_PATHS:
        evaluate_model(path, device)

    print("\nAll models evaluated.")


if __name__ == "__main__":
    main()
