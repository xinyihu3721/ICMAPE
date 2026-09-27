"""Robot signal-detector training entry point.

All robot-specific agent dispatch, wandb naming, and save-path logic lives here.
train_sequential.py handles bandit agents; robot agents are dispatched below.

Usage
-----
  # Two-source signal detection (RobotExploration2 env):
  python train_robot.py +experiments/robot/SignalDetector2=robot_multicpe_sac
  python train_robot.py +experiments/robot/SignalDetector2=robot_multicpe_td3
  python train_robot.py +experiments/robot/SignalDetector2=robot_multicpe_qlearning

  # Override any field inline:
  python train_robot.py +experiments/robot/SignalDetector2=robot_multicpe_sac experiment.seed=42
  python train_robot.py +experiments/robot/SignalDetector2=robot_multicpe_sac training.max_training_steps=400000
"""

import os
import sys
import torch
import numpy as np
import wandb
from datetime import datetime

import dataclasses
import hydra
from omegaconf import DictConfig

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icmape.dataclasses import Config
from icmape.agents.bandit.config import AlgorithmType
from icmape.agents.robot.sequential.multicpe_sac_robot import MULTICPE_SAC_ROBOT
from icmape.agents.robot.sequential.multicpe_td3_robot import MULTICPE_TD3_ROBOT
from icmape.agents.robot.sequential.multicpe_qlearning import MULTICPE_QLEARNING_ROBOT


def run_train_robot(cfg: DictConfig) -> None:
    _valid = {f.name for f in dataclasses.fields(Config)}
    cfg: Config = Config(**{k: v for k, v in cfg.items() if k in _valid})
    date = datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
    print(f"Configuration: {cfg}", flush=True)

    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    agent_type = cfg.agent.type
    env_type_tag = getattr(getattr(cfg.environment, "robot_env", None), "type", "RobotEnv")

    base_run_name = (
        f"{agent_type} - "
        f"({cfg.environment.horizon}"
        f"/G{cfg.environment.grid_size}"
        f"/V{cfg.environment.view_side}"
        f"/{cfg.experiment.seed})"
    )
    wandb_tags = [
        cfg.agent.type,
        env_type_tag,
        f"H={cfg.environment.horizon}",
        f"G={cfg.environment.grid_size}",
        f"V={cfg.environment.view_side}",
        f"P={cfg.environment.num_players}",
        f"Smoke Decay={cfg.environment.smoke_decay}",
        f"Smoke Sigma={cfg.environment.smoke_obs_sigma}",
    ]

    gumbel_tau_start, gumbel_tau_end = None, None
    if agent_type == AlgorithmType.ROBOT_MULTICPE_TD3:
        gumbel_tau_start = float(getattr(cfg.agent.actor_network, "gumbel_temperature", 1.0))
        gumbel_tau_end = float(getattr(cfg.agent.actor_network, "gumbel_tau_min", 0.1))
        wandb_tags += [f"tau_start={gumbel_tau_start}", f"tau_end={gumbel_tau_end}"]

    if agent_type == AlgorithmType.ROBOT_MULTICPE_QLEARNING:
        if getattr(cfg.agent, "uniform_action_only", False):
            base_run_name = f"UNIFORM_{base_run_name}"
            wandb_tags.append("uniform_action_only")
        elif not getattr(cfg.agent.critic_network, "VDN_mode", True):
            base_run_name = f"VDN0_{base_run_name}"
            wandb_tags.append("VDN0_mode")
        else:
            wandb_tags.append("VDN_mode")

    wandb.init(
        project=cfg.experiment.name,
        notes=f"[{date}] - {cfg.experiment.description}",
        name=base_run_name,
        tags=wandb_tags,
        group=getattr(cfg.experiment, "wandb_group", None),
        config=cfg,
    )

    if agent_type == AlgorithmType.ROBOT_MULTICPE_SAC:
        algo = "SAC"
    elif agent_type == AlgorithmType.ROBOT_MULTICPE_TD3:
        algo = "TD3"
    elif agent_type == AlgorithmType.ROBOT_MULTICPE_QLEARNING:
        if getattr(cfg.agent, "uniform_action_only", False):
            algo = "Uniform"
        elif getattr(cfg.agent.critic_network, "VDN_mode", False):
            algo = "VDN"
        else:
            algo = "INDEP"
    else:
        algo = str(agent_type)
    wandb.config.update({"algo": algo})
    if gumbel_tau_start is not None:
        wandb.config.update({"gumbel_tau_start": gumbel_tau_start, "gumbel_tau_end": gumbel_tau_end})

    device = "cpu" if os.environ.get("FORCE_CPU", "false").lower() == "true" \
        else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if agent_type == AlgorithmType.ROBOT_MULTICPE_SAC:
        agent = MULTICPE_SAC_ROBOT(cfg, device)
    elif agent_type == AlgorithmType.ROBOT_MULTICPE_TD3:
        agent = MULTICPE_TD3_ROBOT(cfg, device)
    elif agent_type == AlgorithmType.ROBOT_MULTICPE_QLEARNING:
        agent = MULTICPE_QLEARNING_ROBOT(cfg, device)
    else:
        raise ValueError(
            f"Robot agent type '{agent_type}' not recognised. "
            "Expected one of: ROBOT_MULTICPE_SAC, ROBOT_MULTICPE_TD3, ROBOT_MULTICPE_QLEARNING"
        )

    env_name = (
        f"{cfg.experiment.name}"
        f"_H{cfg.environment.horizon}"
        f"_G{cfg.environment.grid_size}"
        f"_V{cfg.environment.view_side}"
        f"_D{cfg.environment.smoke_decay}"
        f"_S{cfg.environment.smoke_obs_sigma}"
    )
    PATH = f"models/robots/{env_name}_{algo}"
    os.makedirs(PATH, exist_ok=True)

    agent.train(PATH)
    agent.save(PATH)

    print("Training complete.", flush=True)
    wandb.finish()


@hydra.main(config_path="configs", config_name="config_robot")
def train(cfg: DictConfig) -> None:
    run_train_robot(cfg)


if __name__ == "__main__":
    train()
