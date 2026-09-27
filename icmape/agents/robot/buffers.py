"""Replay buffer for robot signal-detector inference (mirrors ``icpe.agents.bandit.buffers`` layout)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RobotReplayBufferSamples:
    states: torch.Tensor  # [batch_size, num_players, grid_size, grid_size]
    actions: torch.Tensor  # Shape: [batch_size, num_players]
    location_history: torch.Tensor  # Shape: [batch_size, num_players, horizon, 2]
    next_states: torch.Tensor  # [batch_size, num_players, grid_size, grid_size]
    dones: torch.Tensor  # Shape: [batch_size]
    traj_current_timestep: torch.Tensor  # Shape: [batch_size]
    true_signal_maps: torch.Tensor  # [batch_size, grid_size, grid_size] ground-truth true-signal map (0/1)


# Backwards-compatible names used in ``multicpe_sac_robot`` / ``__init__.py``
RobotSACReplayBufferSamples = RobotReplayBufferSamples


class RobotReplayBuffer:
    """Ring buffer for robot sequential MULTICPE SAC (multi-agent transitions)."""

    def __init__(
        self,
        buffer_size: int,
        grid_size: int,
        view_side: int,
        num_actions: int,
        num_players: int,
        num_traj: int,
        device: torch.device,
        horizon: int | None = None,
        enable_last_transitions_sampling: bool = False,
    ) -> None:
        """
        Initializes the buffer for storing robot sequential SAC transitions.

        Args:
            buffer_size: Maximum number of transitions the buffer holds.
            grid_size: Size of the environment grid (assumes square grid).
            num_actions: Number of possible actions for each player/robot.
            num_players: Number of robots/players.
            num_traj: Number of trajectories.
            device: Torch device to put tensors on when sampling.
            horizon: Length of history for each stored sequence.
            enable_last_transitions_sampling: Whether to track and prioritize recent transitions in sampling.
        """
        self.buffer_size = buffer_size + (buffer_size % num_traj)
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.view_side = view_side
        self.num_players = num_players
        self.num_traj = num_traj
        self.device = device
        self.horizon = horizon
        self.enable_last_transitions_sampling = enable_last_transitions_sampling
        self.reset()

    def reset(self) -> None:
        """Resets/allocates all internal buffers to initial (zeroed) state."""
        if self.horizon is None:
            raise ValueError("RobotSACReplayBuffer requires horizon")
        self.states = np.zeros(
            (self.buffer_size, self.num_players, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        self.next_states = np.zeros(
            (self.buffer_size, self.num_players, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        self.actions = np.zeros((self.buffer_size, self.num_players), dtype=np.int64)
        self.location_history = np.zeros((self.buffer_size, self.num_players, self.horizon, 2), dtype=np.int64)
        self.dones = np.zeros(self.buffer_size, dtype=bool)
        self.traj_current_timestep = np.zeros(self.buffer_size, dtype=np.float32)
        # Ground-truth true-signal map per transition (episode-constant per env, stored per step)
        self.true_signal_maps = np.zeros(
            (self.buffer_size, self.grid_size, self.grid_size), dtype=np.float32
        )
        self.new_transitions_idxs: list[int] = []
        self.pos = 0
        self.full = False

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        traj_current_timestep: torch.Tensor | None = None,
        location_history: torch.Tensor | None = None,
        true_signal_maps: torch.Tensor | None = None,
    ) -> None:
        """
        Adds transitions to the buffer.

        Args:
            true_signal_maps: Ground-truth true-signal map per row,
                shape ``[B, grid_size, grid_size]``. Required.
        """
        if true_signal_maps is None:
            raise ValueError("true_signal_maps is required (ground-truth signal map per transition row).")
        num_transitions = states.shape[0]
        insert_start = self.pos
        insert_end = self.pos + num_transitions

        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy()
        next_states_np = next_states.detach().cpu().numpy()
        location_history_np = (
            location_history.detach().cpu().numpy()
            if location_history is not None
            else None
        )
        traj_timestep_np = (
            traj_current_timestep.detach().cpu().numpy() if traj_current_timestep is not None else None
        )

        signal_np = true_signal_maps.detach().cpu().numpy().astype(np.float32, copy=False)

        if insert_end <= self.buffer_size:
            sl = slice(insert_start, insert_end)
            self.states[sl] = states_np
            self.actions[sl] = actions_np
            self.dones[sl] = dones_np
            self.next_states[sl] = next_states_np
            self.location_history[sl] = location_history_np
            self.true_signal_maps[sl] = signal_np
            if traj_timestep_np is not None:
                self.traj_current_timestep[sl] = traj_timestep_np
            self.pos = insert_end
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
        else:
            first_part = self.buffer_size - insert_start
            second_part = num_transitions - first_part
            sl1 = slice(insert_start, self.buffer_size)
            sl2 = slice(0, second_part)
            self.states[sl1] = states_np[:first_part]
            self.actions[sl1] = actions_np[:first_part]
            self.dones[sl1] = dones_np[:first_part]
            self.next_states[sl1] = next_states_np[:first_part]
            self.location_history[sl1] = location_history_np[:first_part]
            self.true_signal_maps[sl1] = signal_np[:first_part]
            if traj_timestep_np is not None:
                self.traj_current_timestep[sl1] = traj_timestep_np[:first_part]

            self.states[sl2] = states_np[first_part:]
            self.actions[sl2] = actions_np[first_part:]
            self.dones[sl2] = dones_np[first_part:]
            self.next_states[sl2] = next_states_np[first_part:]
            self.location_history[sl2] = location_history_np[first_part:]
            self.true_signal_maps[sl2] = signal_np[first_part:]
            if traj_timestep_np is not None:
                self.traj_current_timestep[sl2] = traj_timestep_np[first_part:]

            self.full = True
            self.pos = second_part

    def sample(self, batch_size: int, add_last_transtions: bool = True) -> RobotReplayBufferSamples:
        """Samples a batch of transitions including ``true_signal_maps``."""
        _ = add_last_transtions
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound == 0:
            raise ValueError("Cannot sample from an empty buffer")
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return RobotReplayBufferSamples(
            states=torch.tensor(self.states[batch_inds], device=self.device, dtype=torch.float32),
            actions=torch.tensor(self.actions[batch_inds], device=self.device, dtype=torch.long),
            location_history=torch.tensor(
                self.location_history[batch_inds], device=self.device, dtype=torch.long
            ),
            next_states=torch.tensor(self.next_states[batch_inds], device=self.device, dtype=torch.float32),
            dones=torch.tensor(self.dones[batch_inds], device=self.device, dtype=torch.float32),
            traj_current_timestep=torch.tensor(
                self.traj_current_timestep[batch_inds], device=self.device, dtype=torch.float32
            ),
            true_signal_maps=torch.tensor(
                self.true_signal_maps[batch_inds], device=self.device, dtype=torch.float32
            ),
        )

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos


RobotSACReplayBuffer = RobotReplayBuffer
