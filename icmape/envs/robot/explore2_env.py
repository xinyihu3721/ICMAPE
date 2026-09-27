
from __future__ import annotations

from typing import Any, Sequence

import gymnasium
import numpy as np

from icmape.envs.robot.base import RobotEnv
from icmape.envs.robot.config import RobotEnvType
from icmape.envs.robot.env_utils import (
    create_signal_map,
    compute_smoke_map_two,
    _view_mask_for_area,
    generate_fixed_start_positions,
    precompute_valid_signal_pairs,
)
from icmape.envs.robot.state import RobotExperience, RobotState

NUM_OBSERVATION_MODES = 3


class RobotExploration2Env(RobotEnv):
    """Multi-agent signal-detector environment with exactly two sources per episode.

    Identical area-grid / action / belief-update mechanics as RobotExploration2Env,
    but each episode samples a true signal cell and a distractor cell whose Manhattan
    distance is at least min_signal_dist apart. Valid pairs are precomputed once
    at init so each reset is O(1) sampling from a pre-built index.
    """

    metadata = {"render_modes": ["human"]}

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY_ACTION = 4
    STOP_ACTION = 5

    def __init__(
        self,
        num_agents: int,
        grid_size: int = 15,
        view_side: int = 5,
        max_steps: int = 500,
        seed: int | None = None,
        start_positions: np.ndarray | Sequence[Sequence[int]] | None = None,
        min_signal_dist: int = 5,
        signal_hit_prob: float = 0.9,
        distractor_intensity_min: float = 0.0,
        distractor_intensity_max: float = 0.7,
        noise_prob: float = 0.1,
        smoke_decay: float = 0.3,
        smoke_obs_sigma: float = 0.05,
        use_l1_view_mask: bool = False,
        l1_view_radius: int | None = None,
        enable_stopping_action: bool = True,
        validate_actions: bool = True,
        record_experience: bool = True,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.view_side = int(view_side)
        self.num_areas_per_side = self.grid_size // self.view_side
        self.max_steps = max_steps
        self.min_signal_dist = int(min_signal_dist)
        self.detection_prob = float(signal_hit_prob)
        self.distractor_intensity_min = float(distractor_intensity_min)
        self.distractor_intensity_max = float(distractor_intensity_max)
        self._distractor_intensity = self.distractor_intensity_max  # overwritten each reset
        self.noise_prob = float(noise_prob)
        self.smoke_decay = float(smoke_decay)
        self.smoke_obs_sigma = float(smoke_obs_sigma)
        self.use_l1_view_mask = bool(use_l1_view_mask)
        self.l1_view_radius = (
            int(l1_view_radius) if l1_view_radius is not None else max(0, self.view_side // 2)
        )
        self.enable_stopping_action = enable_stopping_action
        self.validate_actions = bool(validate_actions)
        self.record_experience = bool(record_experience)

        self.stay_action_index = self.STAY_ACTION
        self.stop_action_index = self.STOP_ACTION if self.enable_stopping_action else -1
        self._num_action_space = 5 + int(self.enable_stopping_action)
        self._action_to_delta = np.asarray(
            [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)], dtype=np.int32
        )

        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self.robot_env_type = RobotEnvType.EXPLORATION2

        # Signal state: row 0 = true signal, row 1 = distractor
        self._signal_cells_rc = np.zeros((2, 2), dtype=np.int32)  # [[r1,c1],[r2,c2]]
        self._signal_map_f32 = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.true_signal_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)

        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.agent_belief = np.zeros(
            (self.num_agents, self.grid_size, self.grid_size), dtype=np.float32
        )
        self._precomputed_observation_maps = np.zeros(
            (self.num_agents, self.grid_size, self.grid_size), dtype=np.float32
        )

        self.start_positions = self._normalize_start_positions(start_positions)
        self._position_history = np.zeros(
            (self.max_steps + 1, self.num_agents, 2), dtype=np.int32
        )
        self._position_history_len = 0

        self.current_state = RobotState.zeros(
            num_agents=self.num_agents,
            grid_size=self.grid_size,
            max_steps=self.max_steps,
        )

        # Precompute area geometry (identical to RobotExploration2Env)
        self._area_view_masks = self._build_area_view_masks()
        self._area_view_mask_indices = self._build_area_view_mask_indices()
        self._area_flat_indices = self._build_area_flat_indices()
        self._area_mask_sizes = self._build_area_mask_sizes()

        # Sensor mode assignment:  agents 0,1 -> mode 1; agents 2+ -> i % 3
        _modes = np.arange(self.num_agents, dtype=np.int64) % NUM_OBSERVATION_MODES
        _modes[0] = 1
        if self.num_agents > 1:
            _modes[1] = 1
        self._agent_modes = _modes

        # Precompute all valid (true-signal, distractor) cell pairs ONCE
        self._valid_signal_locs, self._valid_distractor_locs = precompute_valid_signal_pairs(
            self.grid_size, self.min_signal_dist
        )
        self._num_valid_pairs = len(self._valid_signal_locs)

        self.action_space = gymnasium.spaces.MultiDiscrete(
            np.full(self.num_agents, self._num_action_space, dtype=np.int64)
        )
        self.observation_space = gymnasium.spaces.Dict(
            {
                "belief_maps": gymnasium.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_agents, self.grid_size, self.grid_size),
                    dtype=np.float32,
                ),
                "global_observation": gymnasium.spaces.Box(
                    low=-3,
                    high=float(self.num_agents),
                    shape=(self.grid_size, self.grid_size),
                    dtype=np.float32,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Geometry helpers (identical pattern to RobotExploration2Env)
    # ------------------------------------------------------------------

    def _normalize_start_positions(
        self, start_positions: np.ndarray | Sequence[Sequence[int]] | None
    ) -> np.ndarray:
        if start_positions is None:
            arr = np.asarray(
                generate_fixed_start_positions(self.num_agents, self.num_areas_per_side),
                dtype=np.int32,
            )
        else:
            arr = np.asarray(start_positions, dtype=np.int32).reshape(self.num_agents, 2)
        return arr

    def _build_area_view_masks(self) -> np.ndarray:
        K, G = self.num_areas_per_side, self.grid_size
        table = np.zeros((K, K, G, G), dtype=bool)
        for ar in range(K):
            for ac in range(K):
                table[ar, ac] = _view_mask_for_area(
                    ar, ac,
                    view_side=self.view_side,
                    grid_size=self.grid_size,
                    use_l1_view_mask=self.use_l1_view_mask,
                    l1_view_radius=self.l1_view_radius,
                )
        return table

    def _build_area_view_mask_indices(self) -> np.ndarray:
        K = self.num_areas_per_side
        idx_table = np.empty((K, K), dtype=object)
        for ar in range(K):
            for ac in range(K):
                idx_table[ar, ac] = np.where(self._area_view_masks[ar, ac])
        return idx_table

    def _build_area_flat_indices(self) -> np.ndarray:
        K = self.num_areas_per_side
        flat_table = np.empty((K, K), dtype=object)
        for ar in range(K):
            for ac in range(K):
                rows, cols = self._area_view_mask_indices[ar, ac]
                flat_table[ar, ac] = rows.astype(np.int64) * self.grid_size + cols.astype(np.int64)
        return flat_table

    def _build_area_mask_sizes(self) -> np.ndarray:
        K = self.num_areas_per_side
        out = np.zeros((K, K), dtype=np.int32)
        for ar in range(K):
            for ac in range(K):
                out[ar, ac] = int(self._area_flat_indices[ar, ac].size)
        return out

    # ------------------------------------------------------------------
    # Signal sampling
    # ------------------------------------------------------------------

    def _sample_signal_and_distractor(self) -> None:
        """O(1) -- pick a random index into the precomputed valid-pair pool."""
        idx = int(self._rng.integers(0, self._num_valid_pairs))
        r1, c1 = int(self._valid_signal_locs[idx, 0]), int(self._valid_signal_locs[idx, 1])
        r2, c2 = int(self._valid_distractor_locs[idx, 0]), int(self._valid_distractor_locs[idx, 1])
        self._signal_cells_rc[0] = [r1, c1]   # true signal
        self._signal_cells_rc[1] = [r2, c2]   # distractor
        self.true_signal_mask.fill(False)
        self.true_signal_mask[r1, c1] = True
        self._signal_map_f32 = create_signal_map(self.grid_size, [(r1, c1), (r2, c2)])

    # ------------------------------------------------------------------
    # Observation precomputation (once per episode)
    # ------------------------------------------------------------------

    def _precompute_observation_maps(self) -> None:
        """Full-grid sensor outputs for each agent, fixed for the episode."""
        r1, c1 = int(self._signal_cells_rc[0, 0]), int(self._signal_cells_rc[0, 1])
        r2, c2 = int(self._signal_cells_rc[1, 0]), int(self._signal_cells_rc[1, 1])
        G = self.grid_size
        for i in range(self.num_agents):
            mode = int(self._agent_modes[i])
            if mode == 0:
                # Noisy global field -- both source cells forced to 1
                obs = (self._rng.random((G, G)) < self.noise_prob).astype(np.float32)
                obs[r1, c1] = 1.0
                obs[r2, c2] = 1.0
            elif mode == 1:
                # Sparse point detector with asymmetric intensities.
                obs = np.zeros((G, G), dtype=np.float32)
                obs[r1, c1] = 1.0 if self._rng.random() < self.detection_prob else 0.0
                obs[r2, c2] = (
                    self._distractor_intensity if self._rng.random() < self.detection_prob else 0.0
                )
            else:
                # Voronoi smoke + Gaussian noise
                obs = np.clip(
                    self._smoke_map
                    + self._rng.normal(0.0, self.smoke_obs_sigma, size=(G, G)),
                    0.0, 1.0,
                ).astype(np.float32)
            self._precomputed_observation_maps[i] = obs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_true_map(self) -> np.ndarray:
        """Ground truth map: only the true signal cell is marked."""
        out = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        r1, c1 = int(self._signal_cells_rc[0, 0]), int(self._signal_cells_rc[0, 1])
        out[r1, c1] = 1.0
        return out

    def parameters(self) -> dict[str, Any]:
        """Kwargs to recreate this env via ``RobotExploration2Env(**parameters())``."""
        return {
            "num_agents": self.num_agents,
            "grid_size": self.grid_size,
            "view_side": self.view_side,
            "max_steps": self.max_steps,
            "seed": None,
            "start_positions": self.start_positions.copy(),
            "min_signal_dist": self.min_signal_dist,
            "signal_hit_prob": self.detection_prob,
            "distractor_intensity_min": self.distractor_intensity_min,
            "distractor_intensity_max": self.distractor_intensity_max,
            "noise_prob": self.noise_prob,
            "smoke_decay": self.smoke_decay,
            "smoke_obs_sigma": self.smoke_obs_sigma,
            "use_l1_view_mask": self.use_l1_view_mask,
            "l1_view_radius": self.l1_view_radius,
            "enable_stopping_action": self.enable_stopping_action,
            "validate_actions": self.validate_actions,
            "record_experience": self.record_experience,
        }

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def _reset_core(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None, copy_state: bool = True
    ) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        del options
        self._steps = 0
        self.agent_belief.fill(-1.0)
        self.agent_positions = self.start_positions.copy()
        seen: set[tuple[int, int]] = set()
        for i in range(self.num_agents):
            tup = (int(self.agent_positions[i, 0]), int(self.agent_positions[i, 1]))
            if tup in seen:
                raise ValueError("start_positions must be unique area indices.")
            seen.add(tup)
        self._distractor_intensity = self._rng.uniform(
            self.distractor_intensity_min, self.distractor_intensity_max
        )
        self._sample_signal_and_distractor()
        self._smoke_map = compute_smoke_map_two(
            self._signal_cells_rc[0], self._signal_cells_rc[1],
            self.grid_size, intensity2=self._distractor_intensity, decay=self.smoke_decay,
        )
        self._precompute_observation_maps()
        self._reset_position_histories()
        self.current_state = RobotState.zeros(
            num_agents=self.num_agents, grid_size=self.grid_size, max_steps=self.max_steps,
        )
        self._update_beliefs_from_local_views()
        self._sync_state_from_world(copy_state=copy_state)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._reset_core(seed=seed, options=options, copy_state=True)
        return self.current_state.to_observation(), self._info_dict()

    def reset_fast(self, *, seed: int | None = None) -> None:
        """Fast reset for VecRobotEnv: updates state without building obs/info dicts."""
        self._reset_core(seed=seed, options=None, copy_state=False)

    def _step_core(self, actions, *, copy_state: bool = True) -> tuple[bool, bool]:
        acts = np.asarray(actions, dtype=np.int64).reshape(-1)
        self._steps += 1
        stopped_now = bool(
            self.enable_stopping_action and np.any(acts == self.stop_action_index)
        )
        self._apply_actions(acts)
        self._append_position_histories()
        self._update_beliefs_from_local_views(acts)
        terminated = stopped_now
        truncated = (not terminated) and bool(self._steps >= self.max_steps)
        self._sync_state_from_world(copy_state=copy_state)
        if self.record_experience:
            self.current_state.add(
                RobotExperience(
                    step=self._steps - 1,
                    actions=acts.copy(),
                    positions=self.agent_positions.copy(),
                    belief_maps=self.agent_belief.copy(),
                    reward=0.0,
                    done=terminated or truncated,
                )
            )
        return terminated, truncated

    def step(self, actions):
        terminated, truncated = self._step_core(actions, copy_state=True)
        obs = self.current_state.to_observation()
        return obs, 0.0, terminated, truncated, self._info_dict()

    def step_fast(self, actions) -> tuple[bool, bool]:
        """Fast step for VecRobotEnv: skips observation/info construction."""
        return self._step_core(actions, copy_state=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_stationary_action(self, a: int) -> bool:
        if a == self.stay_action_index:
            return True
        return self.enable_stopping_action and a == self.stop_action_index

    def _reset_position_histories(self) -> None:
        self._position_history_len = 1
        self._position_history[0] = self.agent_positions

    def _append_position_histories(self) -> None:
        if self._position_history_len < self._position_history.shape[0]:
            self._position_history[self._position_history_len] = self.agent_positions
            self._position_history_len += 1

    def _stack_position_history(self) -> np.ndarray:
        if self._position_history_len <= 0:
            return np.zeros((0, self.num_agents, 2), dtype=np.int32)
        return self._position_history[: self._position_history_len].copy()

    def _info_dict(self) -> dict[str, Any]:
        return {
            "signal_map": self.get_true_map(),
            "signal_locations": self._signal_cells_rc.copy(),  # shape [2, 2]; row 0 = true signal, row 1 = distractor
            "position_history": self._stack_position_history(),
        }

    def _update_beliefs_from_local_views(self, acts: np.ndarray | None = None) -> None:
        belief_flat = self.agent_belief.reshape(self.num_agents, -1)
        precomputed_flat = self._precomputed_observation_maps.reshape(self.num_agents, -1)
        for i in range(self.num_agents):
            if acts is not None and self._is_stationary_action(int(acts[i])):
                continue
            ar, ac = int(self.agent_positions[i, 0]), int(self.agent_positions[i, 1])
            flat = self._area_flat_indices[ar, ac]
            nmask = int(self._area_mask_sizes[ar, ac])
            if nmask == 0:
                continue
            belief_flat[i, flat] = precomputed_flat[i, flat]

    def _sync_state_from_world(self, *, copy_state: bool = True) -> None:
        if copy_state:
            self.current_state.positions = self.agent_positions.copy()
            self.current_state.belief_maps = self.agent_belief.copy()
        else:
            self.current_state.positions = self.agent_positions
            self.current_state.belief_maps = self.agent_belief

    def _apply_actions(self, actions: np.ndarray) -> None:
        acts = np.asarray(actions, dtype=np.int64).reshape(-1)
        deltas = self._action_to_delta[acts].copy()
        if self.enable_stopping_action:
            stationary = (acts == self.stay_action_index) | (acts == self.stop_action_index)
        else:
            stationary = acts == self.stay_action_index
        deltas[stationary] = 0
        hi = max(0, self.num_areas_per_side - 1)
        new_pos = self.agent_positions + deltas
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, hi)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, hi)
        self.agent_positions = new_pos

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=object)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.true_signal_mask[r, c]:
                    grid[r, c] = "S"
        n = self.view_side
        for i, (ar, ac) in enumerate(self.agent_positions):
            tr, tc = int(ar) * n, int(ac) * n
            label = str(i % 10)
            cur = str(grid[tr, tc])
            grid[tr, tc] = label if cur == "." else f"{label}S"
        return "\n".join(" ".join(row) for row in grid)
