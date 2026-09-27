from __future__ import annotations

import math
from typing import Any, Tuple

import gymnasium
import numpy as np

from icmape.envs.robot.config import RobotEnvType
from icmape.envs.robot.state import RobotState


class RobotEnv(gymnasium.Env):
    """Base class for robot (grid exploration) environments."""

    num_agents: int
    grid_size: int
    max_steps: int
    current_state: RobotState
    robot_env_type: RobotEnvType | None = None

    def parameters(self) -> dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "grid_size": self.grid_size,
            "max_steps": self.max_steps,
        }

    def get_true_map(self) -> np.ndarray:
        """Ground-truth signal map; shape [grid_size, grid_size]."""
        raise NotImplementedError


class VecRobotEnv:
    """Batched wrapper mirroring VecBanditEnv: reset / step return state, dones, stopped.

    The state is the stacked belief_maps from each sub-env's RobotState (shape
    [batch, num_agents, grid_size, grid_size]), not a stacked RobotState object.
    """

    def __init__(self, envs: list[RobotEnv]) -> None:
        if len(envs) < 1:
            raise ValueError("envs must contain at least one environment")
        self.envs = envs
        self.batch_size = len(envs)
        self.num_agents = envs[0].num_agents
        self.grid_size = envs[0].grid_size
        self.max_steps = envs[0].max_steps

    @property
    def horizon(self) -> int:
        return self.max_steps

    @staticmethod
    def _belief_maps_batch(envs: list[RobotEnv]) -> np.ndarray:
        return np.stack([e.current_state.belief_maps for e in envs], axis=0)

    def get_signal_maps(self) -> np.ndarray:
        """Stacked ground-truth signal maps, shape [batch_size, grid_size, grid_size]."""
        return np.stack([e.get_true_map() for e in self.envs], axis=0)

    def get_positions_batch(self) -> np.ndarray:
        """Stacked current area positions, shape [batch_size, num_agents, 2]."""
        return np.stack([e.current_state.positions for e in self.envs], axis=0)

    def reset(
        self, seeds: list[int | None] | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset all sub-environments; return batched belief_maps, dones, stopped."""
        if seeds is None:
            seeds = [None] * self.batch_size
        if len(seeds) != self.batch_size:
            raise ValueError("seeds length must match number of environments")

        for env, seed in zip(self.envs, seeds):
            if hasattr(env, "reset_fast"):
                env.reset_fast(seed=seed)  # type: ignore[attr-defined]
            else:
                env.reset(seed=seed)

        batched = self._belief_maps_batch(self.envs)
        z = np.zeros(self.batch_size, dtype=np.bool_)
        return batched, z, z

    def step(
        self, actions: list[np.ndarray | list[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One step per sub-env; return batched belief_maps, dones, stopped."""
        if len(actions) != self.batch_size:
            raise ValueError("actions length must match number of environments")

        dones: list[bool] = []
        stopped: list[bool] = []
        for env, act in zip(self.envs, actions):
            if hasattr(env, "step_fast"):
                terminated, truncated = env.step_fast(act)  # type: ignore[attr-defined]
            else:
                _obs, _reward, terminated, truncated, _info = env.step(act)
            dones.append(bool(terminated or truncated))
            stopped.append(bool(terminated and not truncated))

        batched = self._belief_maps_batch(self.envs)
        return (
            batched,
            np.asarray(dones, dtype=np.bool_),
            np.asarray(stopped, dtype=np.bool_),
        )

    def parameters(self) -> list[dict[str, Any]]:
        return [env.parameters() for env in self.envs]


def _generate_fixed_start_positions(M: int, num_areas_per_side: int) -> np.ndarray:
    if M == 1:
        c = (num_areas_per_side - 1) // 2
        return np.asarray([[c, c]], dtype=np.int32)
    if M == 2:
        hi = num_areas_per_side - 1
        return np.asarray([[0, 0], [hi, hi]], dtype=np.int32)
    low = 0
    high = num_areas_per_side - 1
    k_grid = math.ceil(math.sqrt(M))
    xs = np.linspace(low, high, num=k_grid, dtype=int)
    ys = np.linspace(low, high, num=k_grid, dtype=int)
    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x in xs:
        for y in ys:
            p = (int(x), int(y))
            if p in seen:
                continue
            seen.add(p)
            pts.append(p)
            if len(pts) == M:
                return np.asarray(pts, dtype=np.int32)
    return np.asarray(pts[:M], dtype=np.int32)


class BatchedRobotEnv2:
    """Batched env for RobotExploration2Env (true signal + distractor per episode).

    Valid signal/distractor pairs are precomputed once at init (all ordered
    pairs with Manhattan distance >= min_signal_dist).

    Smoke and distance lookup tables of shape [G, G, G, G] are precomputed so
    the two-source Voronoi smoke map per batch item reduces to two vectorized
    index lookups and a single np.where, avoiding Python loops over source positions.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_agents: int,
        grid_size: int,
        view_side: int,
        max_steps: int,
        base_seed: int,
        start_positions: np.ndarray | None,
        min_signal_dist: int,
        signal_hit_prob: float,
        distractor_intensity_min: float,
        distractor_intensity_max: float,
        noise_prob: float,
        smoke_decay: float,
        smoke_obs_sigma: float,
        enable_stopping_action: bool,
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_agents = int(num_agents)
        self.grid_size = int(grid_size)
        self.view_side = int(view_side)
        self.max_steps = int(max_steps)
        self.horizon = self.max_steps
        self.num_areas_per_side = self.grid_size // self.view_side
        self.min_signal_dist = int(min_signal_dist)
        self.enable_stopping_action = bool(enable_stopping_action)
        self.stay_action_index = 4
        self.stop_action_index = 5 if self.enable_stopping_action else -1
        self.detection_prob = float(signal_hit_prob)
        self.distractor_intensity_min = float(distractor_intensity_min)
        self.distractor_intensity_max = float(distractor_intensity_max)
        self._distractor_intensities = np.full(self.batch_size, self.distractor_intensity_max, dtype=np.float32)
        self.noise_prob = float(noise_prob)
        self.smoke_obs_sigma = float(smoke_obs_sigma)
        self._smoke_decay = float(smoke_decay)
        self._steps = np.zeros(self.batch_size, dtype=np.int32)
        self._action_to_delta = np.asarray(
            [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)], dtype=np.int32
        )

        _modes = np.arange(self.num_agents, dtype=np.int64) % 3
        _modes[0] = 1
        if self.num_agents > 1:
            _modes[1] = 1
        self._agent_modes = _modes.astype(np.int64, copy=False)

        self._rngs = [
            np.random.default_rng(int(base_seed) + i * 97) for i in range(self.batch_size)
        ]

        if start_positions is None:
            sp = _generate_fixed_start_positions(self.num_agents, self.num_areas_per_side)
        else:
            sp = np.asarray(start_positions, dtype=np.int32).reshape(self.num_agents, 2)
        self._start_positions = sp

        B, P, G = self.batch_size, self.num_agents, self.grid_size
        self._belief = np.zeros((B, P, G, G), dtype=np.float32)
        self._precomputed_observation_maps = np.zeros((B, P, G, G), dtype=np.float32)
        self._positions = np.zeros((B, P, 2), dtype=np.int32)
        self._signal_rcs = np.zeros((B, 2, 2), dtype=np.int32)   # [B, 2_sources, (r,c)]; index 0=true signal, 1=distractor
        self._signal_maps = np.zeros((B, G, G), dtype=np.float32)
        self._smoke_maps = np.zeros((B, G, G), dtype=np.float32)
        self._position_history = np.zeros((B, self.max_steps + 1, P, 2), dtype=np.int32)
        self._position_history_len = 0

        K = self.num_areas_per_side
        self._area_view_masks = np.zeros((K, K, G, G), dtype=bool)
        for ar in range(K):
            for ac in range(K):
                r0, c0 = ar * self.view_side, ac * self.view_side
                self._area_view_masks[ar, ac, r0 : r0 + self.view_side, c0 : c0 + self.view_side] = True
        self._area_flat_indices = np.empty((K, K), dtype=object)
        self._area_mask_sizes = np.zeros((K, K), dtype=np.int32)
        for ar in range(K):
            for ac in range(K):
                rows, cols = np.where(self._area_view_masks[ar, ac])
                flat = rows.astype(np.int64) * G + cols.astype(np.int64)
                self._area_flat_indices[ar, ac] = flat
                self._area_mask_sizes[ar, ac] = int(flat.size)

        rr = np.arange(G, dtype=np.float32)[:, None]
        cc = np.arange(G, dtype=np.float32)[None, :]
        self._smoke_lookup = np.zeros((G, G, G, G), dtype=np.float32)
        self._dist_lookup = np.zeros((G, G, G, G), dtype=np.float32)
        for fr in range(G):
            for fc in range(G):
                d = np.abs(rr - fr) + np.abs(cc - fc)
                self._smoke_lookup[fr, fc] = np.exp(-self._smoke_decay * d).astype(np.float32)
                self._dist_lookup[fr, fc] = d.astype(np.float32)

        # Precompute all valid (true signal, distractor) pairs once
        coords = np.array([(r, c) for r in range(G) for c in range(G)], dtype=np.int32)
        dists = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
        mask = np.triu(dists >= self.min_signal_dist, k=1)
        ri, ci = np.where(mask)
        p1_half, p2_half = coords[ri], coords[ci]
        self._valid_signal_locs = np.concatenate([p1_half, p2_half], axis=0)
        self._valid_distractor_locs = np.concatenate([p2_half, p1_half], axis=0)
        self._num_valid_pairs = len(self._valid_signal_locs)

    def _refresh_signals_and_smoke(self) -> None:
        """Sample true signal + distractor per env and compute Voronoi smoke maps -- vectorized."""
        G = self.grid_size
        self._signal_maps.fill(0.0)
        pair_idxs = np.array(
            [int(self._rngs[b].integers(0, self._num_valid_pairs)) for b in range(self.batch_size)],
            dtype=np.int32,
        )
        r1s = self._valid_signal_locs[pair_idxs, 0]
        c1s = self._valid_signal_locs[pair_idxs, 1]
        r2s = self._valid_distractor_locs[pair_idxs, 0]
        c2s = self._valid_distractor_locs[pair_idxs, 1]
        self._signal_rcs[:, 0, 0] = r1s
        self._signal_rcs[:, 0, 1] = c1s
        self._signal_rcs[:, 1, 0] = r2s
        self._signal_rcs[:, 1, 1] = c2s
        for b in range(self.batch_size):
            self._signal_maps[b, r1s[b], c1s[b]] = 1.0
        self._distractor_intensities = np.array(
            [self._rngs[b].uniform(self.distractor_intensity_min, self.distractor_intensity_max)
             for b in range(self.batch_size)],
            dtype=np.float32,
        )
        smoke1 = self._smoke_lookup[r1s, c1s]
        smoke2 = self._distractor_intensities[:, None, None] * self._smoke_lookup[r2s, c2s]
        dist1 = self._dist_lookup[r1s, c1s]
        dist2 = self._dist_lookup[r2s, c2s]
        self._smoke_maps = np.where(dist1 <= dist2, smoke1, smoke2)

    def _precompute_observation_maps(self) -> None:
        B, P, G = self.batch_size, self.num_agents, self.grid_size
        for b in range(B):
            r1, c1 = int(self._signal_rcs[b, 0, 0]), int(self._signal_rcs[b, 0, 1])
            r2, c2 = int(self._signal_rcs[b, 1, 0]), int(self._signal_rcs[b, 1, 1])
            for i in range(P):
                mode = int(self._agent_modes[i])
                if mode == 0:
                    obs = (self._rngs[b].random((G, G)) < self.noise_prob).astype(np.float32)
                    obs[r1, c1] = 1.0
                    obs[r2, c2] = 1.0
                elif mode == 1:
                    obs = np.zeros((G, G), dtype=np.float32)
                    obs[r1, c1] = 1.0 if self._rngs[b].random() < self.detection_prob else 0.0
                    obs[r2, c2] = (
                        self._distractor_intensities[b]
                        if self._rngs[b].random() < self.detection_prob
                        else 0.0
                    )
                else:
                    obs = np.clip(
                        self._smoke_maps[b]
                        + self._rngs[b].normal(0.0, self.smoke_obs_sigma, size=(G, G)),
                        0.0, 1.0,
                    ).astype(np.float32)
                self._precomputed_observation_maps[b, i] = obs

    def _update_beliefs(self, acts: np.ndarray | None = None) -> None:
        belief_flat = self._belief.reshape(self.batch_size, self.num_agents, -1)
        precomputed_flat = self._precomputed_observation_maps.reshape(
            self.batch_size, self.num_agents, -1
        )
        for b in range(self.batch_size):
            for i in range(self.num_agents):
                if acts is not None:
                    a = int(acts[b, i])
                    if a == self.stay_action_index or (
                        self.enable_stopping_action and a == self.stop_action_index
                    ):
                        continue
                ar, ac = int(self._positions[b, i, 0]), int(self._positions[b, i, 1])
                flat = self._area_flat_indices[ar, ac]
                if int(self._area_mask_sizes[ar, ac]) == 0:
                    continue
                belief_flat[b, i, flat] = precomputed_flat[b, i, flat]

    def reset(
        self, seeds: list[int | None] | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seeds is not None:
            for b, s in enumerate(seeds):
                if s is not None:
                    self._rngs[b] = np.random.default_rng(int(s))
        self._steps.fill(0)
        self._belief.fill(-1.0)
        self._positions[:, :, :] = self._start_positions[None, :, :]
        self._refresh_signals_and_smoke()
        self._precompute_observation_maps()
        self._position_history_len = 1
        self._position_history[:, 0, :, :] = self._positions
        self._update_beliefs(acts=None)
        z = np.zeros(self.batch_size, dtype=np.bool_)
        return self._belief, z, z

    def step(
        self, actions: list[np.ndarray | list[int]] | np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        acts = np.asarray(actions, dtype=np.int64).reshape(self.batch_size, self.num_agents)
        self._steps += 1
        stopped = self.enable_stopping_action & np.any(acts == self.stop_action_index, axis=1)
        deltas = self._action_to_delta[acts]
        stationary = acts == self.stay_action_index
        if self.enable_stopping_action:
            stationary = stationary | (acts == self.stop_action_index)
        deltas[stationary] = 0
        self._positions = self._positions + deltas
        hi = self.num_areas_per_side - 1
        self._positions[:, :, 0] = np.clip(self._positions[:, :, 0], 0, hi)
        self._positions[:, :, 1] = np.clip(self._positions[:, :, 1], 0, hi)
        if self._position_history_len < self._position_history.shape[1]:
            self._position_history[:, self._position_history_len, :, :] = self._positions
            self._position_history_len += 1
        self._update_beliefs(acts=acts)
        dones = stopped | (self._steps >= self.max_steps)
        return self._belief, dones.astype(np.bool_), stopped.astype(np.bool_)

    def get_signal_maps(self) -> np.ndarray:
        return self._signal_maps

    def get_signal_positions(self) -> np.ndarray:
        """Signal/distractor coordinates, shape [batch_size, 2, 2]; index 0=true signal, 1=distractor."""
        return self._signal_rcs.copy()

    def get_positions_batch(self) -> np.ndarray:
        return self._positions

    def parameters(self) -> list[dict[str, Any]]:
        return []
