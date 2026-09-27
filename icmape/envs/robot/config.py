from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class RobotEnvType(str, Enum):
    EXPLORATION2 = "RobotExploration2"


@dataclass
class RobotExplorationConfig:
    """Nested env kind (values come from ``configs/environment/robot_env/*.yaml``)."""

    type: RobotEnvType


@dataclass
class RobotSequentialEnvConfig:
    """Top-level Hydra ``environment`` group -- all tunables are set in YAML, not here."""

    robot_env: RobotExplorationConfig
    num_actions: int
    horizon: int
    num_agents: int
    num_players: int
    enable_stopping_action: bool
    decentralized: bool
    reward_struct: str
    num_rewards: int
    grid_size: int
    view_side: int
    num_signal_cells_min: int
    num_signal_cells_max: int
    signal_hit_prob: float
    distractor_intensity_min: float
    distractor_intensity_max: float
    noise_prob: float
    smoke_decay: float
    smoke_obs_sigma: float
    use_l1_view_mask: bool = False
    l1_view_radius: Optional[int] = None
    seed: Optional[int] = None
    start_positions: Optional[Tuple[Tuple[int, int], ...]] = None
    max_history_len: Optional[int] = None  # None -> agent uses ``horizon`` for net history tensors
    min_signal_dist: Optional[int] = None  # EXPLORATION2 only: min Manhattan dist between signal and distractor


@dataclass
class RobotEnvConfig:
    """Built in code from ``RobotSequentialEnvConfig``; no defaults beyond optional placement."""

    robot_env: RobotExplorationConfig
    num_agents: int
    grid_size: int
    view_side: int
    horizon: int
    num_signal_cells_min: int
    num_signal_cells_max: int
    signal_hit_prob: float
    distractor_intensity_min: float
    distractor_intensity_max: float
    noise_prob: float
    smoke_decay: float
    smoke_obs_sigma: float
    use_l1_view_mask: bool = False
    l1_view_radius: Optional[int] = None
    enable_stopping_action: bool = True
    validate_actions: bool = True
    record_experience: bool = True
    seed: Optional[int] = None
    start_positions: Optional[Tuple[Tuple[int, int], ...]] = None
    min_signal_dist: Optional[int] = None  # EXPLORATION2 only: min Manhattan dist between signal and distractor
