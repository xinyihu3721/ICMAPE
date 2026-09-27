from __future__ import annotations
from dataclasses import dataclass

from icmape.agents.bandit.config import BaseAgentConfig
from icmape.envs.bandit.config import BanditEnvConfig
from icmape.envs.robot.config import RobotSequentialEnvConfig

@dataclass
class Config:
    experiment: ExperimentConfig
    training: TrainingConfig
    agent: BaseAgentConfig
    environment: BanditEnvConfig | RobotSequentialEnvConfig  
    eval: EvalConfig

@dataclass
class ExperimentConfig:
    name: str
    description: str
    seed: int
    num_processes: int
    wandb_group: str | None = None

@dataclass
class TrainingConfig:
    @dataclass
    class EvalTrainingConfig:
        eval_period: int  # Frequency of evaluations
        eval_num_env_instances: int
        eval_num_trajectories: int
    
    save_frequency: int
    max_training_steps: int
    cost_per_timestep: float
    eval: EvalTrainingConfig
    checkpoint_path: str | None = None  # Path to checkpoint file to load and continue training from
    continue_training: bool = False  # If True, continue training from checkpoint
    start_step: int | None = None  # Starting step for training (only used if continue_training=True)

    
@dataclass
class EvalConfig:
  num_env_instances: int
  num_trajectories: int