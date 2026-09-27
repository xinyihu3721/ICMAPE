from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

@dataclass
class HistoryData:
    means: torch.Tensor                    # shape (B, K) where B = num_envs, K = num_actions
    caps: torch.Tensor                     # shape (B, K) where B = num_envs, K = num_actions
    exp_actions: torch.Tensor              # shape (B, N, H, P) where N = num_trajectories, H = horizon, P = num_players
    infer_probs: torch.Tensor              # shape (B, N, H, K, P)
    rew_exp_actions: torch.Tensor          # shape (B, N, H) or (B, N, H, num_rewards)
    rew_opt_actions: torch.Tensor          # shape (B, N, K) per-arm match to true capacity holder
    correct_capacity: torch.Tensor       # shape (B, N) 1.0 if all arms match on last step, else 0.0
    exp_distribution: torch.Tensor         # shape (B, N, H, K)
    rew_infer_capacity: torch.Tensor       # shape (B, N, H)
    stopping_time: torch.Tensor            # shape (B, N)
    returns: torch.Tensor = None           # shape (B, N, H) or None
    num_rewards: int = 1

    def to_numpy(self) -> HistoryData:
        """Convert all tensors to numpy arrays."""
        def t2np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        return HistoryData(
            means=t2np(self.means),
            caps=t2np(self.caps),
            exp_actions=t2np(self.exp_actions),
            rew_exp_actions=t2np(self.rew_exp_actions),
            rew_opt_actions=t2np(self.rew_opt_actions),
            correct_capacity=t2np(self.correct_capacity),
            exp_distribution=t2np(self.exp_distribution),
            infer_probs=t2np(self.infer_probs),
            rew_infer_capacity=t2np(self.rew_infer_capacity),
            stopping_time=t2np(self.stopping_time),
        )

    def statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute statistics on the trajectories and return as DataFrames."""
        traj = self.to_numpy()
        rews: np.ndarray = traj.correct_capacity  # (B, N): trajectory-level all-arms-correct
        B, N, H, K = traj.exp_distribution.shape

        avgs = [rews[b].mean() for b in range(B)] + [np.mean(rews)]
        stds = [np.nan_to_num(rews[b].std(ddof=1)) for b in range(B)] + [np.sqrt(np.nan_to_num(rews.var(-1, ddof=1)).mean() + np.nan_to_num(rews.mean(-1).var(ddof=1)))]
        indexes = [f'Env_{b}' for b in range(B)] + ['Overall']

        estimation_df = pd.DataFrame(
            {"avg": avgs,
             "std": stds}, index=indexes)

        idxs = pd.MultiIndex.from_tuples(
            [(b, h) for b in range(B) for h in range(H)],
            names=["Env", "Stage"])
        avg_exp_dist = traj.exp_distribution.mean(1)
        exp_df = pd.DataFrame({f'a{i}': avg_exp_dist[..., i].flatten() for i in range(K)},
                              index=idxs)
        return estimation_df, exp_df


class HistoryBuilder:
    def __init__(self, B: int, N: int, horizon: int, num_actions: int, num_players: int, decentralized: bool, device: torch.device = torch.device('cpu'), num_rewards: int = 1):
        self.B = B
        self.N = N
        self.horizon = horizon
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.decentralized = decentralized
        self.num_rewards = num_rewards

        shape_2d = (B * N, horizon)
        shape_exp_rewards = (B * N, horizon, num_rewards)
        shape_3d = (B * N, horizon, num_actions)
        shape_4d = (B * N, horizon, num_actions, num_players)
        shape_exp_actions = (B * N, horizon, num_players)

        self.exp_actions_one_hot = torch.zeros(shape_3d, dtype=torch.long, device=device)
        self.exp_dist = torch.zeros(shape_3d, dtype=torch.float32, device=device)
        self.opt_dist = torch.zeros(shape_3d, dtype=torch.float32, device=device)
        self.rew_exp_actions = torch.zeros(shape_exp_rewards, dtype=torch.float32, device=device)
        self.rew_opt_actions = torch.zeros((B * N, num_actions), dtype=torch.float32, device=device)
        self.correct_capacity = torch.zeros(B * N, dtype=torch.float32, device=device)
        self.exp_actions = torch.zeros(shape_exp_actions, dtype=torch.long, device=device)
        self.infer_probs = torch.zeros(shape_4d, dtype=torch.float32, device=device)
        self.inferenced_capacity = torch.zeros(shape_3d, dtype=torch.long, device=device)
        self.rew_infer_capacity = torch.zeros(shape_2d, dtype=torch.float32, device=device)
        self.stopping_time = torch.zeros(B * N, dtype=torch.float32, device=device)

        self._idxs = torch.arange(B * N, device=device)

    def record_exploration_dist(self, h: int, dist: torch.Tensor):
        """Record the exploration distribution at time step h (dist: B*N, K)."""
        self.exp_dist[:, h] = dist

    def record_optimal_dist(self, h: int, dist: torch.Tensor):
        """Record the optimal / target distribution at time step h (dist: B*N, K)."""
        self.opt_dist[:, h] = dist

    def build(self, actions: torch.Tensor, rewards: torch.Tensor, probs: torch.Tensor, means: torch.Tensor, caps: torch.Tensor, stopping_time: torch.Tensor, true_optimal: torch.Tensor, use_mle: bool = False) -> HistoryData:
        """Finalize and return a HistoryData object."""
        B, N, H, K, P = self.B, self.N, self.horizon, self.num_actions, self.num_players
        self.rew_exp_actions = rewards
        self.exp_actions = actions
        self.exp_actions_one_hot = F.one_hot(self.exp_actions, num_classes=K)

        self.inferenced_capacity = probs.argmax(dim=-1)  # (B*N, H, K)
        self.infer_probs = probs.exp()
        self.rew_infer_capacity = probs.max(dim=-1).values.sum(dim=-1).exp()

        optimal_actions = true_optimal.to(self.device)
        true_per_arm = optimal_actions.argmax(dim=-1)  # (B, K) or (B*N, K)

        # Use the prediction at the effective stopping step for each sample
        BN = B * N
        st_idx = stopping_time.long().clamp(0, H - 1)  # (B*N,)
        pred_per_arm = self.inferenced_capacity[torch.arange(BN, device=self.device), st_idx]  # (B*N, K)

        self.rew_opt_actions = (pred_per_arm == true_per_arm).float()
        self.correct_capacity = self.rew_opt_actions.all(dim=-1).float()

        self.stopping_time = stopping_time
        return HistoryData(
            means=means[::N],
            caps=caps[::N],
            exp_actions=self.exp_actions.reshape(B, N, H, P),
            rew_exp_actions=self.rew_exp_actions.reshape(B, N, H, self.num_rewards),
            rew_opt_actions=self.rew_opt_actions.reshape(B, N, K),
            correct_capacity=self.correct_capacity.reshape(B, N),
            exp_distribution=self.exp_dist.reshape(B, N, H, K),
            rew_infer_capacity=self.rew_infer_capacity.reshape(B, N, H),
            infer_probs=self.infer_probs.reshape(B, N, H, K, P),
            stopping_time=self.stopping_time.reshape(B, N),
        )
