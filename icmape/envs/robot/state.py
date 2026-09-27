from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class RobotExperience(NamedTuple):
    step: int
    actions: np.ndarray
    positions: np.ndarray
    belief_maps: np.ndarray
    reward: float
    done: bool


class RobotState:
    """Per-step trajectory: each agent holds one full-grid map.

    belief_maps shape [num_agents, grid_size, grid_size].
    """

    def __init__(
        self,
        positions: np.ndarray,
        belief_maps: np.ndarray,
        max_steps: int,
    ) -> None:
        self.positions = positions
        self.belief_maps = belief_maps
        self.max_steps = max_steps
        self.history: list[RobotExperience] = []

    @staticmethod
    def zeros(
        num_agents: int,
        grid_size: int,
        max_steps: int,
    ) -> "RobotState":
        positions = np.zeros((num_agents, 2), dtype=np.int32)
        belief_maps = np.zeros((num_agents, grid_size, grid_size), dtype=np.float32)
        return RobotState(
            positions=positions,
            belief_maps=belief_maps,
            max_steps=max_steps,
        )

    def add(self, exp: RobotExperience) -> "RobotState":
        self.history.append(exp)
        self.positions = exp.positions.copy()
        self.belief_maps = exp.belief_maps.copy()
        return self

    def to_observation(self) -> dict[str, np.ndarray]:
        g = self.belief_maps.sum(axis=0).astype(np.float32, copy=False)
        return {
            "belief_maps": self.belief_maps.astype(np.float32, copy=True),
            "global_observation": np.array(g, dtype=np.float32, copy=True),
        }

    def to_tensor(
        self,
        device: torch.device | str = torch.device("cpu"),
    ) -> dict[str, torch.Tensor]:
        dev = torch.device(device)
        return {
            "belief_maps": torch.as_tensor(
                self.belief_maps, dtype=torch.float32, device=dev
            ),
            "global_observation": torch.as_tensor(
                self.belief_maps.sum(axis=0), dtype=torch.float32, device=dev
            ),
        }
