"""Base off-policy Q-learning agent for the robot environment.

Mirrors RobotBaseOffPolicyAgent (SAC) but uses epsilon-greedy Q-value action
selection instead of a separate actor network. Supports uniform_action_only
mode for inference-only training (Q network is not trained).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import time
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

from icmape.dataclasses import Config
from icmape.agents.robot.buffers import RobotReplayBuffer, RobotReplayBufferSamples
from icmape.agents.robot.config import RobotBaseOffPolicyAgentConfig
from icmape.agents.robot.history import HistoryBuilder, HistoryData
from icmape.envs.robot.base import VecRobotEnv, BatchedRobotEnv2
from icmape.envs.robot.builder import robot_env_config_from_sequential, sample_envs_batched
from icmape.models.robot_net.builder import create_critic_network_qlearning, create_inference_network
from torch.optim.lr_scheduler import LinearLR

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_kwargs):
        return iterable


class RobotBaseOffPolicyAgentQlearning:
    """Off-policy Q-learning base agent for the robot signal-detector environment.

    Compared to the SAC base (``RobotBaseOffPolicyAgent``):
    - No actor network; the critic Q-values drive action selection directly.
    - Epsilon-greedy exploration (exponential decay) with ``uniform_action_only`` bypass.
    - Simplified ``_train_networks`` / ``train`` loop (no actor / alpha losses).

    Subclasses must implement:
      - ``_train_rl_nets``
      - ``_sample_actions_decentralized``
      - ``save`` / ``load``
    """

    cfg: Config
    agent_cfg: RobotBaseOffPolicyAgentConfig
    device: torch.device
    current_step: int

    def __init__(self, cfg: Config, device: torch.device | str):
        self.cfg = cfg
        self.agent_cfg = cfg.agent
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.current_step = 0
        self.gradient_step = 0
        self.i_gradient_step = 0

        self.num_actions = self.cfg.environment.num_actions + int(
            self.cfg.environment.enable_stopping_action
        )
        self.decentralized = self.cfg.environment.decentralized
        self.reward_struct = self.cfg.environment.reward_struct
        self.num_players = self.cfg.environment.num_players
        self.grid_size = self.cfg.environment.grid_size
        self.view_side = self.cfg.environment.view_side
        self.horizon = self.cfg.environment.horizon
        self.total_batch_size = (
            self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories
        )
        self.use_amp = self.device.type != "cpu"
        self.q_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.i_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.log_interval = int(getattr(self.agent_cfg, "log_interval", 100))

        self.buffer = RobotReplayBuffer(
            buffer_size=self.agent_cfg.buffer_size,
            grid_size=self.grid_size,
            num_actions=self.num_actions,
            view_side=self.view_side,
            num_players=self.num_players,
            num_traj=self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories,
            device=self.device,
            horizon=self.horizon,
            enable_last_transitions_sampling=False,
        )

        self.reset_networks()
        self.running_mean_est = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_training_started(self) -> bool:
        if self.uniform_action_only:
            return False
        return (
            len(self.buffer)
            > self.agent_cfg.inference_network_bootstrap_num_batches
            * self.agent_cfg.batch_size
            * self.agent_cfg.replay_ratio
        )

    @property
    def exploration_epsilon(self) -> float:
        """Exponential decay from start to end over ``exploration_epsilon_decay_steps``."""
        T = max(1, self.exploration_epsilon_decay_steps)
        if self.current_step >= T:
            return max(0.0, float(self.exploration_epsilon_end))
        progress = float(self.current_step) / float(T)
        k = -math.log(0.01)
        factor = math.exp(-k * progress)
        eps = self.exploration_epsilon_end + (
            self.exploration_epsilon_start - self.exploration_epsilon_end
        ) * factor
        return max(0.0, float(eps))

    @property
    def is_stopping_enabled(self) -> bool:
        return self.cfg.environment.enable_stopping_action

    @property
    def base_num_actions(self) -> int:
        return self.cfg.environment.num_actions

    @property
    def cost_per_timestep(self) -> float:
        return self._cost_multiplier.item()

    # ------------------------------------------------------------------
    # Network initialisation
    # ------------------------------------------------------------------

    def reset_networks(self):
        """Build critic, target critic, inference network, and optimisers.

        No actor is created; epsilon-greedy over Q-values replaces the actor.
        """
        bins = getattr(self.agent_cfg, "bins", 1)
        # Default to 1 for Q-learning (Double DQN still uses online+target copies).
        n_critic = getattr(self.agent_cfg, "n_critic", 1)
        architecture_type = self.agent_cfg.architecture_type
        robot_state_dim = self.num_players * self.grid_size * self.grid_size

        self.critic = create_critic_network_qlearning(
            robot_state_dim,
            output_dim=self.num_actions,
            num_actions=self.num_actions,
            num_players=self.num_players,
            horizon=self.horizon,
            config=self.agent_cfg.critic_network,
            grid_size=self.grid_size,
            bins=bins,
            architecture_type=architecture_type,
            belief_channels=1,
        ).to(self.device)
        self.critic_tgt = create_critic_network_qlearning(
            robot_state_dim,
            output_dim=self.num_actions,
            num_actions=self.num_actions,
            num_players=self.num_players,
            horizon=self.horizon,
            config=self.agent_cfg.critic_network,
            grid_size=self.grid_size,
            bins=bins,
            architecture_type=architecture_type,
            belief_channels=1,
        ).to(self.device)
        self.critic_tgt.hard_update(self.critic)

        self.i_net = create_inference_network(
            self.agent_cfg.inference_network, grid_size=self.grid_size
        ).to(self.device)
        self.i_tgt_net = create_inference_network(
            self.agent_cfg.inference_network, grid_size=self.grid_size
        ).to(self.device)
        self.i_tgt_net.hard_update(self.i_net)

        # Q-learning specific hyper-parameters (read from agent config / defaults)
        self.uniform_action_only = getattr(self.agent_cfg, "uniform_action_only", False)
        self.exploration_epsilon_start = getattr(
            self.agent_cfg, "exploration_epsilon_start", 1.0
        )
        self.exploration_epsilon_end = getattr(
            self.agent_cfg, "exploration_epsilon_end", 0.05
        )
        _decay = getattr(self.agent_cfg, "exploration_epsilon_decay_steps", None)
        self.exploration_epsilon_decay_steps = (
            _decay if _decay is not None else self.cfg.training.max_training_steps // 3
        )

        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.agent_cfg.critic_learning_rate
        )
        self.i_optimizer = torch.optim.AdamW(
            self.i_net.parameters(), lr=self.agent_cfg.inference_learning_rate
        )

        if "cost_learning_rate" not in self.agent_cfg:
            print("cost_learning_rate missing from config. Setting to 0")
            cost_lr = 0.0
        else:
            cost_lr = self.agent_cfg.cost_learning_rate
        self.min_cost = 0.0
        self._cost_multiplier = torch.tensor(
            self.min_cost, requires_grad=True, device=self.device, dtype=torch.float32
        )
        self.cost_optimizer = torch.optim.AdamW([self._cost_multiplier], lr=cost_lr)

        if "final_learning_rate" not in self.agent_cfg:
            final_lr = self.agent_cfg.critic_learning_rate
        else:
            final_lr = self.agent_cfg.final_learning_rate

        self.scheduler_exp = LinearLR(
            self.critic_optimizer,
            start_factor=1.0,
            end_factor=final_lr / self.agent_cfg.critic_learning_rate,
            total_iters=int(0.5 * self.cfg.training.max_training_steps),
        )
        self.scheduler_inf = LinearLR(
            self.i_optimizer,
            start_factor=1.0,
            end_factor=final_lr / self.agent_cfg.inference_learning_rate,
            total_iters=int(0.5 * self.cfg.training.max_training_steps),
        )
        self.scheduler_step = lambda: (
            self.scheduler_exp.step(),
            self.scheduler_inf.step(),
        )

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def sample_envs(
        self, B: int, N: int, seed_offset: int = 0
    ) -> VecRobotEnv | BatchedRobotEnv2:
        robot_cfg = robot_env_config_from_sequential(self.cfg.environment)
        base_seed = (
            int(self.cfg.experiment.seed)
            if getattr(self.cfg.experiment, "seed", None) is not None
            else 0
        )
        return sample_envs_batched(robot_cfg, B, N, base_seed + int(seed_offset))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _sample_actions_decentralized(
        self,
        state: torch.Tensor,
        location_history: torch.Tensor,
        timestep: torch.Tensor,
        greedy: bool = False,
        exploration_epsilon: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (actions [B,P], qvalues [B, P*A]) given the current observation."""
        raise NotImplementedError

    @abstractmethod
    def _train_rl_nets(self, batch: RobotReplayBufferSamples) -> Dict[str, List[float]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_estimation_net(self, batch: RobotReplayBufferSamples) -> Dict[str, List[float]]:
        self.i_net.train()

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            obs_map = batch.next_states
            logits = self.i_net(obs_map)          # [B, G, G] logits
            B, G, _ = logits.shape
            target_idx = batch.true_signal_maps.view(B, -1).argmax(dim=-1)  # [B]
            est_loss = F.cross_entropy(logits.view(B, -1), target_idx)

        self.i_optimizer.zero_grad()
        self.i_scaler.scale(est_loss).backward()
        self.i_scaler.unscale_(self.i_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.i_net.parameters(), self.agent_cfg.grad_clipping
        )
        self.i_scaler.step(self.i_optimizer)
        self.i_scaler.update()

        self.i_gradient_step += 1
        freq = int(
            self.agent_cfg.frequency_update_i_network * self.cfg.environment.horizon
        )
        if self.i_gradient_step % freq == 0:
            self.i_tgt_net.hard_update(self.i_net)

        return {"inference_loss": [est_loss.item()]}

    def _train_networks(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        inference_loss: Dict[str, List[float]] = {}
        qloss: Dict[str, List[float]] = {}

        initial_thresh = (
            self.agent_cfg.initial_buffer_bootstrap_num_batches
            * self.agent_cfg.batch_size
            * self.agent_cfg.replay_ratio
        )
        rl_thresh = (
            self.agent_cfg.inference_network_bootstrap_num_batches
            * self.agent_cfg.batch_size
            * self.agent_cfg.replay_ratio
        )

        if len(self.buffer) > initial_thresh:
            for _ in range(self.agent_cfg.replay_ratio):
                inference_loss = self._train_estimation_net(
                    self.buffer.sample(self.agent_cfg.batch_size, add_last_transtions=False)
                )

            if not self.uniform_action_only and len(self.buffer) > rl_thresh:
                for _ in range(self.agent_cfg.replay_ratio):
                    qloss = self._train_rl_nets(
                        self.buffer.sample(self.agent_cfg.batch_size, add_last_transtions=False)
                    )
                    self.gradient_step += 1

        return inference_loss, qloss

    def _update_cost(
        self, probs: torch.Tensor, true_signal_maps: torch.Tensor
    ) -> Dict[str, float]:
        estm_rewards_argmax = probs.max(dim=-1).values
        err = self.agent_cfg.confidence_level - estm_rewards_argmax.mean()

        if not self.uniform_action_only:
            cost_loss = self._cost_multiplier * err.detach()
            self.cost_optimizer.zero_grad()
            cost_loss.backward()
            self.cost_optimizer.step()
            with torch.no_grad():
                self._cost_multiplier.data.clamp_(min=self.min_cost, max=1.0)

        return {
            "cost": self._cost_multiplier.item(),
            "target_confidence_error": err.abs().item(),
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_statistics(
        self,
        step: int,
        training_statistics: Dict[str, List[float]],
        probs_i_network: torch.Tensor,
        true_signal_maps: torch.Tensor,
        qvalues: torch.Tensor,
        horizons: torch.Tensor = None,
    ):
        estm_rewards_argmax = probs_i_network.max(dim=-1).values.float().log()
        estm_rewards_probs = probs_i_network.max(dim=-1).values
        estm_rewards_argmax = estm_rewards_argmax.detach().cpu().numpy()
        estm_rewards_probs = estm_rewards_probs.detach().cpu().numpy()

        q_np = qvalues.detach().cpu().numpy()
        values = {
            "values/avg_qvalues": q_np.mean(),
            "values/std_avg_qvalues": q_np.mean(-1).std(),
            "values/max_qvalues": q_np.max(-1).mean(),
            "values/min_qvalues": q_np.min(-1).mean(),
        }
        performance = {
            "performance/est_rewards_argmax": estm_rewards_argmax.mean(),
            "performance/est_rewards_probs": estm_rewards_probs.mean(),
        }
        training_stats = {
            f"loss/{k}": np.mean(v) for k, v in training_statistics.items()
        }

        if horizons is not None:
            hf = horizons.to(dtype=torch.float32)
            performance["performance/avg_horizon"] = hf.mean().item()
            performance["performance/median_horizon"] = hf.median().item()

        wandb.log({**performance, **training_stats, **values}, step=step)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_trajectories(self, B: int, N: int, seed_offset: int = 0) -> HistoryData:
        """Sample robot trajectories for eval using greedy Q-value policy."""
        envs = self.sample_envs(B, N, seed_offset=seed_offset)
        H = self.horizon
        BN = B * N
        K = self.num_actions

        traj_builder = HistoryBuilder(
            B, N, H, K, self.num_players, self.decentralized, device=self.device
        )

        states, dones, stopped = envs.reset()
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        signal_maps = torch.as_tensor(
            envs.get_signal_maps(), dtype=torch.float32, device=self.device
        )
        true_signal_idx = signal_maps.view(BN, -1).argmax(dim=-1).to(dtype=torch.long)
        _fp_np = envs.get_signal_positions() if hasattr(envs, "get_signal_positions") else None
        signal_positions = (
            torch.as_tensor(_fp_np, dtype=torch.long, device=self.device)
            if _fp_np is not None else None
        )
        traj_current_timestep = torch.zeros(BN, device=self.device, dtype=torch.long)
        location_history = torch.zeros(
            (BN, self.num_players, H, 2), dtype=torch.long, device=self.device
        )
        location_history[:, :, 0, :] = torch.as_tensor(
            envs.get_positions_batch(), dtype=torch.long, device=self.device
        )

        actions_hist = torch.zeros(
            (BN, H, self.num_players), dtype=torch.long, device=self.device
        )
        position_hist = torch.zeros(
            (BN, H, self.num_players, 2), dtype=torch.long, device=self.device
        )
        log_probabilities = torch.zeros(
            (BN, H, K, self.num_players), dtype=torch.float32, device=self.device
        )
        rewards_hist = torch.zeros((BN, H, 1), dtype=torch.float32, device=self.device)
        exp_signal_location_idx_hist = torch.zeros(
            (BN, H, 1), dtype=torch.float32, device=self.device
        )
        exp_optimal_reward_hist = torch.zeros(
            (BN, H, 1), dtype=torch.float32, device=self.device
        )

        # For UNIFORM mode the env never receives a stop action; stopping is instead
        # determined by the inference net's confidence exceeding confidence_level,
        # mirroring the training-loop logic in self.train().
        already_done = torch.zeros(BN, dtype=torch.bool, device=self.device)

        for h in range(H):
            actions, qvalues = self._sample_actions_decentralized(
                states,
                location_history=location_history,
                timestep=traj_current_timestep,
                greedy=True,
            )
            actions = actions.view(BN, self.num_players).long()
            actions_hist[:, h] = actions

            # Store Q-value distribution as a proxy for the exploration policy.
            # qvalues: [BN, P*A]; reshape to [BN, P, A] then average over players -> [BN, A].
            q_reshaped = qvalues.view(BN, self.num_players, K)  # [BN, P, A]
            q_softmax = torch.softmax(q_reshaped, dim=-1)        # [BN, P, A]
            # HistoryBuilder expects [BN, K, P] log-prob-like tensor
            log_probs_h = q_softmax.permute(0, 2, 1).contiguous()  # [BN, A, P]
            log_probabilities[:, h] = log_probs_h

            informative_probs = q_softmax.mean(dim=1)  # [BN, A]
            traj_builder.record_exploration_dist(h, informative_probs)

            actions_np = actions.detach().cpu().numpy()
            next_states, dones_np, stopped_np = envs.step(actions_np)
            next_states = torch.as_tensor(
                next_states, dtype=torch.float32, device=self.device
            )
            dones = torch.from_numpy(dones_np).to(self.device)
            stopped = torch.from_numpy(stopped_np).to(self.device)

            probs_next = self.i_net.predict_proba(next_states).view(BN, -1)
            rewards_hist[:, h] = (
                probs_next.max(dim=-1).values.to(dtype=torch.float32).unsqueeze(-1)
            )
            exp_signal_location_idx_hist[:, h] = (
                probs_next.argmax(dim=-1).to(dtype=torch.float32).unsqueeze(-1)
            )
            exp_optimal_reward_hist[:, h] = probs_next.gather(
                dim=-1, index=true_signal_idx.unsqueeze(-1)
            )

            clipped_next_positions = torch.as_tensor(
                envs.get_positions_batch(), dtype=torch.long, device=self.device
            )
            position_hist[:, h] = clipped_next_positions
            next_t = torch.clamp(traj_current_timestep + 1, max=H - 1)
            batch_idx = torch.arange(BN, device=self.device)
            location_history[batch_idx, :, next_t, :] = clipped_next_positions

            # UNIFORM mode: stop when inference net confidence exceeds threshold,
            # matching the identical logic in self.train() (t >= 5 warmup).
            if self.uniform_action_only and self.is_stopping_enabled and h >= 5:
                conf_stop = probs_next.max(dim=-1).values > self.agent_cfg.confidence_level
                already_done = already_done | conf_stop

            stopped = stopped | already_done
            dones = dones | already_done

            states = next_states
            traj_current_timestep = traj_current_timestep + (1 - stopped.long())
            if torch.all(dones):
                break

        if self.is_stopping_enabled:
            stop_idx = self.num_actions - 1
            any_stop = (actions_hist == stop_idx).any(dim=-1)
            h_idx = torch.arange(H, device=actions_hist.device).view(1, -1)
            first_stop_h = torch.where(
                any_stop,
                h_idx.expand(BN, -1),
                torch.full_like(any_stop, H, dtype=torch.long),
            ).min(dim=1).values
            for bn in range(BN):
                fh = int(first_stop_h[bn].item())
                if fh >= H:
                    continue
                t0 = fh + 1
                if t0 < H:
                    actions_hist[bn, t0:] = 0
                    position_hist[bn, t0:] = 0
                    rewards_hist[bn, t0:] = 0
                    exp_signal_location_idx_hist[bn, t0:] = 0
                    exp_optimal_reward_hist[bn, t0:] = 0
                    log_probabilities[bn, t0:] = 0
                    traj_builder.exp_dist[bn, t0:] = 0

        mean_policy = traj_builder.exp_dist.mean(dim=1)

        return traj_builder.build(
            actions=actions_hist,
            position_history=position_hist,
            rewards=rewards_hist,
            exp_signal_location_idx=exp_signal_location_idx_hist,
            exp_optimal_reward=exp_optimal_reward_hist,
            probs=log_probabilities,
            policy_mean=mean_policy,
            signal_maps=signal_maps,
            stopping_time=traj_current_timestep.to(dtype=torch.float32),
            true_signal_idx=true_signal_idx,
            signal_positions=signal_positions,
        )

    def eval(self, B: int, N: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        traj = self.sample_trajectories(B, N)
        df_est, df_exp = traj.statistics()
        H = self.cfg.environment.horizon
        idenv = np.random.choice(B)

        Bn, Nn = traj.stopping_time.shape
        end_t = traj.stopping_time.to(dtype=torch.long).clamp(min=0, max=H - 1)
        b_idx = torch.arange(Bn, device=self.device).unsqueeze(1).expand(Bn, Nn)
        n_idx = torch.arange(Nn, device=self.device).unsqueeze(0).expand(Bn, Nn)
        pred_signal_idx = traj.exp_signal_location_idx[b_idx, n_idx, end_t, 0].to(dtype=torch.long)
        true_signal_idx = traj.true_signal_idx.to(dtype=torch.long)
        final_acc = (pred_signal_idx == true_signal_idx).to(dtype=torch.float32)
        overall_acc = final_acc.mean().item()
        avg_t = traj.stopping_time.to(dtype=torch.float32).mean().item()
        median_t = traj.stopping_time.to(dtype=torch.float32).median().item()

        print(
            f"[Eval] Agent: {self.agent_cfg.type} | Horizon: {H} | "
            f"Signal-loc acc: {overall_acc:.4f} | "
            f"Avg timestep: {avg_t:.2f} | Median: {median_t:.2f}"
        )
        return df_est, df_exp

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, save_path: str):
        B, N = self.agent_cfg.num_env_instances, self.agent_cfg.num_trajectories

        total_train_time = 0.0
        total_other_time = 0.0
        total_steps = 0

        step_iter = tqdm(
            range(self.cfg.training.max_training_steps),
            desc="train",
            unit="step",
            total=self.cfg.training.max_training_steps,
            dynamic_ncols=True,
            disable=False,
            # mininterval=300.0,
            # miniters=500,
            mininterval=10.0,
            miniters=100,
        )
       

        for step in step_iter:
            step_t0 = time.perf_counter()
   

            envs = self.sample_envs(B=B, N=N, seed_offset=step)
            states, dones, stopped = envs.reset()
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            true_signal_maps = torch.as_tensor(envs.get_signal_maps(), dtype=torch.float32, device=self.device)

            traj_current_timestep = torch.zeros(B * N, device=self.device, dtype=torch.long)
            location_history = torch.zeros((B * N, self.num_players, self.horizon, 2),dtype=torch.long,device=self.device,)

            location_history[:, :, 0, :] = torch.as_tensor(envs.get_positions_batch(), dtype=torch.long, device=self.device)

            for t in range(self.horizon):
                with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    actions, qvalues = self._sample_actions_decentralized(
                        states,
                        location_history=location_history,
                        timestep=traj_current_timestep,
                        exploration_epsilon=self.exploration_epsilon,
                    )

                actions = actions.view(-1, self.num_players).long()
                actions_np = actions.detach().cpu().numpy()
                next_states, dones, stopped = envs.step(actions_np)
                next_states = torch.as_tensor(
                    next_states, dtype=torch.float32, device=self.device
                )
                dones = torch.from_numpy(dones).to(self.device)
                stopped = torch.from_numpy(stopped).to(self.device)

                # uniform_action_only: override stopped/dones when inference is confident.
                # Mirrors the bandit base: after a short warmup the inference net's max
                # probability is compared to confidence_level.  Any trajectory that crosses
                # the threshold is immediately marked stopped AND done -- the env never
                # receives a stop action; the training loop simply ends that episode early.
                if self.uniform_action_only and self.is_stopping_enabled and t >= 5:
                    with torch.no_grad():
                        conf = self.i_net.predict_proba(next_states).view(
                            next_states.shape[0], -1
                        ).max(dim=-1).values          # [B*N]
                    conf_stop = conf > self.agent_cfg.confidence_level
                    stopped = stopped | conf_stop
                    dones = dones | conf_stop

                next_positions = torch.as_tensor(
                    envs.get_positions_batch(), dtype=torch.long, device=self.device
                )
                next_t = torch.clamp(traj_current_timestep + 1, max=self.horizon - 1)
                batch_idx = torch.arange(B * N, device=self.device)
                location_history[batch_idx, :, next_t, :] = next_positions

                self.buffer.add(
                    states,
                    actions,
                    dones,
                    next_states,
                    traj_current_timestep,
                    location_history=location_history,
                    true_signal_maps=true_signal_maps,
                )

                states = next_states
                traj_current_timestep = traj_current_timestep + (1 - stopped.long())

                if torch.all(dones):
                    break

            t_net0 = time.perf_counter()
            inference_loss, qloss = self._train_networks()
            t_net1 = time.perf_counter()
            total_train_time += t_net1 - t_net0

            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                B_ = states.shape[0]
                i_net_probs = self.i_net.predict_proba(states).view(B_, -1)
                qvalues = self.critic.forward(
                    states, location_history, timestep=traj_current_timestep
                )
                qvalues = qvalues.view(B_*self.num_players, -1)
                
                

            cost_updates: Dict[str, float] = {}
            if self.is_stopping_enabled:
                cost_updates = self._update_cost(i_net_probs, true_signal_maps)

            op_values, op_idx = i_net_probs.max(dim=-1)
            op = op_values.detach().cpu().float().numpy().round(4)
            op_idx_np = op_idx.detach().cpu().numpy()

            if self.current_step % self.log_interval == 0:
                print(f"\nQV: {qvalues.detach().cpu().float().numpy().round(2)[0]}", flush=True)
                print(f"OP: val={op[0]:.4f}, id={op_idx_np[0]} | Mean: {op.mean():.4f}", flush=True)
                true_signal_id = true_signal_maps.view(true_signal_maps.shape[0], -1).argmax(dim=-1)
                print(f"True signal location id: {true_signal_id[0].item()}", flush=True)
                print(f"Traj timestep: {traj_current_timestep[0]}", flush=True)
                print(f"Epsilon: {self.exploration_epsilon:.4f}\n", flush=True)

            learning_rates = {
                "lr_exploration_net": self.critic_optimizer.param_groups[0]["lr"],
                "lr_inference_net": self.i_optimizer.param_groups[0]["lr"],
                "exploration_epsilon": self.exploration_epsilon,
            }

            self._log_statistics(
                step=self.current_step,
                training_statistics={
                    **inference_loss,
                    **qloss,
                    **cost_updates,
                    **learning_rates,
                },
                probs_i_network=i_net_probs,
                true_signal_maps=true_signal_maps,
                qvalues=qvalues,
                horizons=traj_current_timestep,
            )
            self.current_step += 1

            if self.current_step % self.log_interval == 0:
                step_iter.set_postfix(
                    q_mean=f"{float(qvalues.mean().item()):.3f}",
                    conf=f"{float(op.mean()):.3f}",
                    eps=f"{self.exploration_epsilon:.3f}",
                    traj_t=int(traj_current_timestep.float().mean().item()),
                )

            self.scheduler_step()
            if self.current_step % self.cfg.training.save_frequency == 0:
                self.save(save_path)

            step_total = time.perf_counter() - step_t0
            total_other_time += max(0.0, step_total - (t_net1 - t_net0))
            total_steps += 1

        total_time = total_train_time + total_other_time
        train_pct = (100.0 * total_train_time / total_time) if total_time > 0 else 0.0
        print(
            "[Timing][training complete] "
            f"steps={total_steps} | "
            f"train_networks_total={total_train_time:.3f}s ({train_pct:.1f}%) | "
            f"avg_per_step={total_train_time / max(1, total_steps):.3f}s"
        )

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: str):
        raise NotImplementedError
