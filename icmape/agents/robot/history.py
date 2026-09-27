from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class HistoryData:
    """Robot-specific trajectory history used by eval/diagnostics."""

    policy_mean: torch.Tensor           # shape (B, K)
    signal_maps: torch.Tensor           # shape (B, G, G)
    exp_actions: torch.Tensor           # shape (B, N, H, P)
    position_history: torch.Tensor      # shape (B, N, H, P, 2)
    exp_distribution: torch.Tensor      # shape (B, N, H, K)
    rew_exp_actions: torch.Tensor       # shape (B, N, H, 1), max(probs_next)
    exp_signal_location_idx: torch.Tensor  # shape (B, N, H, 1), argmax(probs_next)
    exp_optimal_reward: torch.Tensor    # shape (B, N, H, 1), probs at true signal location
    log_probs: torch.Tensor             # shape (B, N, H, K, P)
    true_signal_idx: torch.Tensor       # shape (B, N), flattened true-signal-cell index
    stopping_time: torch.Tensor         # shape (B, N)
    signal_positions: Optional[torch.Tensor] = None  # shape (B, num_sources, 2), (row, col) per source

    def to_numpy(self) -> "HistoryData":
        def t2np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        return HistoryData(
            policy_mean=t2np(self.policy_mean),
            signal_maps=t2np(self.signal_maps),
            exp_actions=t2np(self.exp_actions),
            position_history=t2np(self.position_history),
            exp_distribution=t2np(self.exp_distribution),
            rew_exp_actions=t2np(self.rew_exp_actions),
            exp_signal_location_idx=t2np(self.exp_signal_location_idx),
            exp_optimal_reward=t2np(self.exp_optimal_reward),
            log_probs=t2np(self.log_probs),
            true_signal_idx=t2np(self.true_signal_idx),
            stopping_time=t2np(self.stopping_time),
            signal_positions=t2np(self.signal_positions) if self.signal_positions is not None else None,
        )

    def statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return coarse eval stats and per-stage exploration distribution."""
        traj = self.to_numpy()
        B, N, H, K = traj.exp_distribution.shape

        # Estimation quality from inference-net reward proxy at final stage.
        final_rew = traj.exp_optimal_reward[:, :, -1, 0]            # [B, N]
        avgs = [float(final_rew[b].mean()) for b in range(B)] + [float(final_rew.mean())]
        stds = [float(final_rew[b].std()) for b in range(B)] + [float(final_rew.std())]
        indexes = [f"Env_{b}" for b in range(B)] + ["Overall"]
        estimation_df = pd.DataFrame({"avg": avgs, "std": stds}, index=indexes)

        idxs = pd.MultiIndex.from_tuples(
            [(b, h) for b in range(B) for h in range(H)],
            names=["Env", "Stage"],
        )
        avg_exp_dist = traj.exp_distribution.mean(axis=1)           # [B, H, K]
        exp_df = pd.DataFrame(
            {f"a{i}": avg_exp_dist[..., i].reshape(-1) for i in range(K)},
            index=idxs,
        )
        return estimation_df, exp_df


class HistoryBuilder:
    def __init__(
        self,
        B: int,
        N: int,
        horizon: int,
        num_actions: int,
        num_players: int,
        decentralized: bool,
        device: torch.device = torch.device("cpu"),
    ):
        del decentralized
        self.B = B
        self.N = N
        self.horizon = horizon
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device

        BN, H, K, P = B * N, horizon, num_actions, num_players
        self.exp_dist = torch.zeros((BN, H, K), dtype=torch.float32, device=device)

    def record_exploration_dist(self, h: int, dist: torch.Tensor):
        self.exp_dist[:, h] = dist

    def build(
        self,
        actions: torch.Tensor,
        position_history: torch.Tensor,
        rewards: torch.Tensor,
        exp_signal_location_idx: torch.Tensor,
        exp_optimal_reward: torch.Tensor,
        probs: torch.Tensor,
        policy_mean: torch.Tensor,
        signal_maps: torch.Tensor,
        stopping_time: torch.Tensor,
        true_signal_idx: torch.Tensor,
        signal_positions: Optional[torch.Tensor] = None,
    ) -> HistoryData:
        B, N, H, P, K = self.B, self.N, self.horizon, self.num_players, self.num_actions
        return HistoryData(
            policy_mean=policy_mean[::N],
            signal_maps=signal_maps[::N],
            exp_actions=actions.reshape(B, N, H, P),
            position_history=position_history.reshape(B, N, H, P, 2),
            exp_distribution=self.exp_dist.reshape(B, N, H, K),
            rew_exp_actions=rewards.reshape(B, N, H, 1),
            exp_signal_location_idx=exp_signal_location_idx.reshape(B, N, H, 1),
            exp_optimal_reward=exp_optimal_reward.reshape(B, N, H, 1),
            log_probs=probs.reshape(B, N, H, K, P),
            true_signal_idx=true_signal_idx.reshape(B, N),
            stopping_time=stopping_time.reshape(B, N),
            signal_positions=signal_positions[::N] if signal_positions is not None else None,
        )


__all__ = ["HistoryBuilder", "HistoryData"]
