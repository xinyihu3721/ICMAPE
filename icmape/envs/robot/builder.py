from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from icmape.envs.robot.base import VecRobotEnv, RobotEnv, BatchedRobotEnv2
from icmape.envs.robot.config import (
    RobotEnvConfig,
    RobotEnvType,
    RobotExplorationConfig,
    RobotSequentialEnvConfig,
)
from icmape.envs.robot.explore2_env import RobotExploration2Env


def create_robot_env(type: RobotEnvType, **kwargs) -> RobotEnv:
    if type == RobotEnvType.EXPLORATION2:
        return RobotExploration2Env(**kwargs)
    raise Exception(f"Environment type {type} not implemented!")


def _start_positions_array(
    cfg: RobotEnvConfig,
) -> np.ndarray | None:
    if cfg.start_positions is None:
        return None
    arr = np.asarray(cfg.start_positions, dtype=np.int32)
    if arr.size == 0:
        return None
    return arr.reshape(cfg.num_agents, 2)


def _seq_env_get(
    env: RobotSequentialEnvConfig | Mapping[str, Any], key: str, default: Any = None
) -> Any:
    if isinstance(env, Mapping):
        return env.get(key, default)
    return getattr(env, key, default)


def robot_env_config_from_sequential(
    env: RobotSequentialEnvConfig | Mapping[str, Any], seed: int | None = None
) -> RobotEnvConfig:
    """Build RobotEnvConfig from Hydra env (dataclass or plain dict from YAML)."""
    re = _seq_env_get(env, "robot_env")
    if isinstance(re, Mapping):
        rt = re["type"] if isinstance(re, dict) else re.get("type")
        robot_env = RobotExplorationConfig(
            type=rt if isinstance(rt, RobotEnvType) else RobotEnvType(str(rt))
        )
    elif isinstance(re, RobotExplorationConfig):
        robot_env = re
    else:
        raise TypeError(
            f"robot_env must be mapping or RobotExplorationConfig, got {type(re)!r}"
        )

    sp = _seq_env_get(env, "start_positions")
    if sp is not None and isinstance(sp, (list, tuple)) and len(sp) > 0:
        if isinstance(sp[0], (list, tuple)):
            sp = tuple(tuple(int(x) for x in row) for row in sp)

    seed_val = _seq_env_get(env, "seed") if seed is None else seed

    lr = _seq_env_get(env, "l1_view_radius", None)
    l1_radius = int(lr) if lr is not None else None

    msd = _seq_env_get(env, "min_signal_dist", None)
    min_signal_dist = int(msd) if msd is not None else None

    return RobotEnvConfig(
        robot_env=robot_env,
        num_agents=int(_seq_env_get(env, "num_players")),
        grid_size=int(_seq_env_get(env, "grid_size")),
        view_side=int(_seq_env_get(env, "view_side")),
        horizon=int(_seq_env_get(env, "horizon")),
        seed=seed_val,
        start_positions=sp,
        num_signal_cells_min=int(_seq_env_get(env, "num_signal_cells_min")),
        num_signal_cells_max=int(_seq_env_get(env, "num_signal_cells_max")),
        signal_hit_prob=float(_seq_env_get(env, "signal_hit_prob")),
        distractor_intensity_min=float(_seq_env_get(env, "distractor_intensity_min", 0.0)),
        distractor_intensity_max=float(_seq_env_get(env, "distractor_intensity_max", 0.7)),
        noise_prob=float(_seq_env_get(env, "noise_prob")),
        smoke_decay=float(_seq_env_get(env, "smoke_decay", 0.3)),
        smoke_obs_sigma=float(_seq_env_get(env, "smoke_obs_sigma", 0.05)),
        use_l1_view_mask=bool(_seq_env_get(env, "use_l1_view_mask", False)),
        l1_view_radius=l1_radius,
        enable_stopping_action=bool(_seq_env_get(env, "enable_stopping_action", True)),
        validate_actions=bool(_seq_env_get(env, "validate_actions", False)),
        record_experience=bool(_seq_env_get(env, "record_experience", False)),
        min_signal_dist=min_signal_dist,
    )


def sample_env(cfg: RobotEnvConfig) -> RobotEnv:
    return RobotExploration2Env(
        num_agents=cfg.num_agents,
        grid_size=cfg.grid_size,
        view_side=cfg.view_side,
        max_steps=cfg.horizon,
        seed=cfg.seed,
        start_positions=_start_positions_array(cfg),
        min_signal_dist=cfg.min_signal_dist if cfg.min_signal_dist is not None else 5,
        signal_hit_prob=cfg.signal_hit_prob,
        distractor_intensity_min=cfg.distractor_intensity_min,
        distractor_intensity_max=cfg.distractor_intensity_max,
        noise_prob=cfg.noise_prob,
        smoke_decay=cfg.smoke_decay,
        smoke_obs_sigma=cfg.smoke_obs_sigma,
        use_l1_view_mask=cfg.use_l1_view_mask,
        l1_view_radius=cfg.l1_view_radius,
        enable_stopping_action=cfg.enable_stopping_action,
        validate_actions=cfg.validate_actions,
        record_experience=cfg.record_experience,
    )


def sample_envs(cfg: RobotEnvConfig, B: int) -> VecRobotEnv:
    """Sample B independent environments."""
    return VecRobotEnv([sample_env(cfg) for _ in range(B)])


def sample_envs_batched(
    cfg: RobotEnvConfig,
    B: int,
    N: int,
    base_seed: int,
) -> BatchedRobotEnv2:
    """Sample B * N batched environments."""
    return BatchedRobotEnv2(
        batch_size=B * N,
        num_agents=cfg.num_agents,
        grid_size=cfg.grid_size,
        view_side=cfg.view_side,
        max_steps=cfg.horizon,
        base_seed=base_seed,
        start_positions=_start_positions_array(cfg),
        min_signal_dist=cfg.min_signal_dist if cfg.min_signal_dist is not None else 5,
        signal_hit_prob=cfg.signal_hit_prob,
        distractor_intensity_min=cfg.distractor_intensity_min,
        distractor_intensity_max=cfg.distractor_intensity_max,
        noise_prob=cfg.noise_prob,
        smoke_decay=cfg.smoke_decay,
        smoke_obs_sigma=cfg.smoke_obs_sigma,
        enable_stopping_action=cfg.enable_stopping_action,
    )
