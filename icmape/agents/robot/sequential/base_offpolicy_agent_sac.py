from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import time
from abc import abstractmethod
from typing import Dict, List, Tuple

from icmape.dataclasses import Config
from icmape.agents.robot.buffers import RobotReplayBuffer, RobotReplayBufferSamples
from icmape.agents.robot.config import RobotBaseOffPolicyAgentConfig
from icmape.agents.robot.history import HistoryBuilder, HistoryData
from icmape.envs.robot.base import VecRobotEnv, BatchedRobotEnv2
from icmape.envs.robot.builder import robot_env_config_from_sequential, sample_envs_batched
from icmape.models.robot_net.builder import (
    create_actor_network,
    create_critic_network_sac,
    create_inference_network,
)
from torch.optim.lr_scheduler import LinearLR
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_kwargs):
        return iterable


class RobotBaseOffPolicyAgent:
    """RL + inference backbone for ``RobotSequentialEnvConfig`` + ``VecRobotEnv``.

    Rollouts stack per-env ``belief_maps`` from ``VecRobotEnv`` as ``[B, P, G, G]`` (see
    ``RobotState``). Subclasses implement decentralized sampling / RL losses; inference can
    train on the summed map (global view), matching ``obs['global_observation']`` in the env.
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

        self.num_actions = self.cfg.environment.num_actions + int(self.cfg.environment.enable_stopping_action)
        # self.base_num_actions is now a property (defined below)
        self.decentralized = self.cfg.environment.decentralized
        self.reward_struct = self.cfg.environment.reward_struct
        self.num_players = self.cfg.environment.num_players
        self.grid_size = self.cfg.environment.grid_size
        self.view_side = self.cfg.environment.view_side
        self.horizon = self.cfg.environment.horizon
        self.total_batch_size = self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories
        self.use_amp = (self.device != "cpu")
        self.q_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.i_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.actor_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        # TD3 (and similar) omit ``ent_coef``; SAC YAML supplies it. Struct configs forbid missing keys.
        self.ent_coef = OmegaConf.select(self.agent_cfg, "ent_coef", default=None)
        self.actor=None
        self.log_interval = int(OmegaConf.select(self.agent_cfg, "log_interval", default=100))
       

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
    
    @property
    def alpha(self):
        if hasattr(self, 'log_alpha'):
            return self.log_alpha.exp()
        raise AttributeError("alpha is only available in SAC mode (log_alpha not set)")
    
    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def has_training_started(self) -> bool:
        buffer_ready = len(self.buffer) > self.agent_cfg.inference_network_bootstrap_num_batches * self.agent_cfg.batch_size * self.agent_cfg.replay_ratio
        #actor_warmup_passed = self.current_step >= self.agent_cfg.actor_warmup_steps * self.cfg.training.max_training_steps
        return buffer_ready #and actor_warmup_passed
    
    @property
    def is_stopping_enabled(self) -> bool:
        return self.cfg.environment.enable_stopping_action

    @property
    def base_num_actions(self) -> int:
        return self.cfg.environment.num_actions

    @property
    def state_dim(self) -> int:
        # return  self.num_players*self.num_actions#(self.num_actions)*self.num_players  # concatenated one-hot encoding of the actions plus one reward for all
        return  self.num_players * self.num_actions#(self.num_actions)*self.num_players  # concatenated one-hot encoding of the actions plus one reward for all 
       
    
    @property
    def cost_per_timestep(self) -> float:
        return self._cost_multiplier.item()
        # # New: EMA of cost = alpha * current + (1 - alpha) * old_cost
        # current = self._cost_multiplier.item()
        # new_cost = self.cost_ema_alpha * current + (1.0 - self.cost_ema_alpha) * self._cost_per_timestep_ema
        # # self._cost_per_timestep_ema = new_cost
        # return new_cost


    def _create_actor_network(
        self, robot_state_dim: int, architecture_type: str
    ) -> "torch.nn.Module":
        """Factory for the actor network.  Override in subclasses (e.g. TD3)."""
        shared_critic = getattr(self.agent_cfg.actor_network, "shared_critic", False)
        return create_actor_network(
            robot_state_dim,
            self.base_num_actions,
            self.num_players,
            self.horizon,
            self.agent_cfg.actor_network,
            grid_size=self.grid_size,
            shared_critic=shared_critic,
            architecture_type=architecture_type,
            belief_channels=1,
        )

    def _init_algorithm_components(self):
        """Algorithm-specific setup called at the end of ``reset_networks``.

        Default: SAC entropy temperature.  Override in subclasses (e.g. TD3) to replace.
        """
        if self.ent_coef is None:
            raise ValueError(
                "agent.ent_coef is required for SAC; omit it only for agents that override "
                "_init_algorithm_components (e.g. MULTICPE_TD3_ROBOT)."
            )
        if self.ent_coef == 'auto':
            self.min_alpha = 0.005
            init_alpha = 0.1
            self.log_alpha = torch.nn.Parameter(
                torch.log(torch.tensor(init_alpha, device=self.device))
            )
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha], lr=self.agent_cfg.ent_learning_rate
            )
            self.target_entropy = -float(np.log(self.num_actions))
        else:
            self.ent_coef = torch.tensor(self.ent_coef, device=self.device)

    def reset_networks(self):
        """

        - **Actor** / **critic**: belief maps ``[B, P, grid_size, grid_size]`` (agents differ by
          sensor type in the env) plus optional ``location_history`` ``[B, P, T, 2]`` (or ``[B, T, P, 2]``).
        - **Inference**: single map ``[B, grid_size, grid_size]`` -> per-cell signal probability
          ``[B, grid_size, grid_size]``.

        ``robot_state_dim`` is only a legacy constructor argument; spatial forwards ignore it.
        """
        bins = getattr(self.agent_cfg, "bins", 1)
        n_critic = getattr(self.agent_cfg, "n_critic", 2)

        # ``icpe.models.robot_net.builder`` branches on ``architecture_type == "cnn_lstm"`` for
        # grid CNN + location LSTM; other values fall back to ``sequence_net`` builders.
        architecture_type = self.agent_cfg.architecture_type


        robot_state_dim = self.num_players * self.grid_size * self.grid_size

        self.critic = create_critic_network_sac(
            robot_state_dim,
            output_dim=self.num_actions,
            num_actions=self.num_actions,
            num_players=self.num_players,
            horizon=self.horizon,
            config=self.agent_cfg.critic_network,
            grid_size=self.grid_size,
            bins=bins,
            n_critic=n_critic,
            architecture_type=architecture_type,
            belief_channels=1,
        ).to(self.device)
        self.critic_tgt = create_critic_network_sac(
            robot_state_dim,
            output_dim=self.num_actions,
            num_actions=self.num_actions,
            num_players=self.num_players,
            horizon=self.horizon,
            config=self.agent_cfg.critic_network,
            grid_size=self.grid_size,
            bins=bins,
            n_critic=n_critic,
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

        self.actor = self._create_actor_network(robot_state_dim, architecture_type).to(self.device)
        

        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr = self.agent_cfg.critic_learning_rate)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr = self.agent_cfg.actor_learning_rate)
        self.i_optimizer = torch.optim.AdamW(self.i_net.parameters(), lr = self.agent_cfg.inference_learning_rate) 

        if 'cost_learning_rate' not in self.agent_cfg:
            print('cost_learning_rate missing from config. Setting to 0')
            cost_lr = 0
        else:
            cost_lr = self.agent_cfg.cost_learning_rate
        self.min_cost = 0 #1 / (self.num_actions * np.sqrt(self.horizon)*self.num_players)
        self._cost_multiplier = torch.tensor(self.min_cost, requires_grad=True, device=self.device, dtype=torch.float32)
        self._cost_per_timestep_ema = self.min_cost # for EMA in cost_per_timestep
        self.cost_optimizer = torch.optim.AdamW([self._cost_multiplier], lr=cost_lr)

        self._init_algorithm_components()

        if 'final_learning_rate' not in self.agent_cfg:
            final_learning_rate = self.agent_cfg.learning_rate
        else:
            final_learning_rate = self.agent_cfg.final_learning_rate

        # end_factor = final_learning_rate/self.agent_cfg.learning_rate
        # self.scheduler_exp = LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=end_factor, total_iters=int(0.5 * self.cfg.training.max_training_steps) )
        # self.scheduler_inf = LinearLR(self.e_optimizer, start_factor=1.0, end_factor=end_factor, total_iters=int(0.5 * self.cfg.training.max_training_steps))
        self.scheduler_exp = LinearLR(self.critic_optimizer, start_factor=1.0, end_factor= final_learning_rate/self.agent_cfg.critic_learning_rate, total_iters=int(0.5 * self.cfg.training.max_training_steps) )
        self.scheduler_inf = LinearLR(self.i_optimizer, start_factor=1.0, end_factor=final_learning_rate/self.agent_cfg.inference_learning_rate, total_iters=int(0.5 * self.cfg.training.max_training_steps))
        self.scheduler_actor = LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=min(final_learning_rate,self.agent_cfg.actor_learning_rate)/self.agent_cfg.actor_learning_rate, total_iters=int(0.5 * self.cfg.training.max_training_steps))

        self.scheduler_step = lambda: (
                self.scheduler_exp.step(),
                self.scheduler_inf.step(),
                self.scheduler_actor.step(),
                #self.scheduler_cost.step()
            )


    @torch.inference_mode()
    def sample_trajectories(
        self, B: int, N: int, seed_offset: int = 0
    ) -> HistoryData:
        """Sample robot trajectories for eval, mirroring bandit-style history outputs.

        ``seed_offset`` is added to ``experiment.seed`` when building batched envs (see
        ``sample_envs``). Use different offsets across repeated evals so each run draws
        independent signal/distractor placements; default ``0`` matches a single ``eval`` call.
        """
        envs = self.sample_envs(B, N, seed_offset=seed_offset)
        H = self.horizon
        BN = B * N
        K = self.num_actions

        traj_builder = HistoryBuilder(
            B,
            N,
            H,
            K,
            self.num_players,
            self.decentralized,
            device=self.device,
        )

        states, dones, stopped = envs.reset()
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        signal_maps = torch.as_tensor(envs.get_signal_maps(), dtype=torch.float32, device=self.device)
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

        actions_hist = torch.zeros((BN, H, self.num_players), dtype=torch.long, device=self.device)
        position_hist = torch.zeros((BN, H, self.num_players, 2), dtype=torch.long, device=self.device)
        # HistoryBuilder expects [BN, H, K, P] log-prob-like tensor.
        log_probabilities = torch.zeros((BN, H, K, self.num_players), dtype=torch.float32, device=self.device)
        rewards_hist = torch.zeros((BN, H, 1), dtype=torch.float32, device=self.device)
        exp_signal_location_idx_hist = torch.zeros((BN, H, 1), dtype=torch.float32, device=self.device)
        exp_optimal_reward_hist = torch.zeros((BN, H, 1), dtype=torch.float32, device=self.device)

        for h in range(H):
            actions, action_log_probs = self._sample_actions_decentralized(
                states,
                location_history=location_history,
                timestep=traj_current_timestep,
                network=self.actor,
                greedy=True,
            )
            actions = actions.view(BN, self.num_players).long()
            actions_hist[:, h] = actions

            # Convert [BN, P, K] -> [BN, K, P] for HistoryBuilder compatibility.
            log_probs_h = action_log_probs.permute(0, 2, 1).contiguous()
            log_probabilities[:, h] = log_probs_h

            informative_probs = action_log_probs.exp().mean(dim=1)  # [BN, K]
            traj_builder.record_exploration_dist(h, informative_probs)

            actions_np = actions.detach().cpu().numpy()
            next_states, dones_np, stopped_np = envs.step(actions_np)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones = torch.from_numpy(dones_np).to(self.device)
            stopped = torch.from_numpy(stopped_np).to(self.device)
            probs_next = self.i_net.predict_proba(next_states).view(BN, -1)
            # Critic reward history: max confidence value of inference distribution.
            rewards_hist[:, h] = probs_next.max(dim=-1).values.to(dtype=torch.float32).unsqueeze(-1)
            # Predicted signal-location index from inference distribution.
            exp_signal_location_idx_hist[:, h] = probs_next.argmax(dim=-1).to(dtype=torch.float32).unsqueeze(-1)
            # Expected reward at true signal location from inference distribution.
            exp_optimal_reward_hist[:, h] = probs_next.gather(dim=-1, index=true_signal_idx.unsqueeze(-1))

            # Positions returned by env after `step` are already boundary-clipped.
            clipped_next_positions = torch.as_tensor(
                envs.get_positions_batch(), dtype=torch.long, device=self.device
            )
            position_hist[:, h] = clipped_next_positions
            next_t = torch.clamp(traj_current_timestep + 1, max=H - 1)
            batch_idx = torch.arange(BN, device=self.device)
            location_history[batch_idx, :, next_t, :] = clipped_next_positions

            states = next_states
            traj_current_timestep = traj_current_timestep + (1 - stopped.long())
            if torch.all(dones):
                break

        # Once any player plays the stop action, that sub-env is terminal for the *episode*, but the
        # batched loop keeps stepping every row until **all** B*N envs are done. Clear tail timesteps
        # so stored history matches "episode ends when anyone stops" semantics for each trajectory.
        if self.is_stopping_enabled:
            stop_idx = self.num_actions - 1
            any_stop = (actions_hist == stop_idx).any(dim=-1)  # [BN, H]
            h_idx = torch.arange(H, device=actions_hist.device).view(1, -1)
            first_stop_h = torch.where(
                any_stop, h_idx.expand(BN, -1), torch.full_like(any_stop, H, dtype=torch.long)
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

        # Use averaged exploration policy as a coarse per-env summary.
        mean_policy = traj_builder.exp_dist.mean(dim=1)  # [BN, K]

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
        """Evaluate method over N trajectories per B environments

        Args:
            B (int): number of environments to sample
            N (int): number of trajectories per environment

        Returns:
            df (pandas.DataFrame): Pandas dataframe with the evaluation statistics
        """
        traj = self.sample_trajectories(B, N)
        df_est, df_exp = traj.statistics()
        H = self.cfg.environment.horizon
        idenv = np.random.choice(B)

        # Signal-location identification accuracy from inference prediction index
        # at each trajectory's actual end timestep.
        Bn, Nn = traj.stopping_time.shape
        end_t = traj.stopping_time.to(dtype=torch.long).clamp(min=0, max=H - 1)  # [B, N]
        b_idx = torch.arange(Bn, device=self.device).unsqueeze(1).expand(Bn, Nn)
        n_idx = torch.arange(Nn, device=self.device).unsqueeze(0).expand(Bn, Nn)
      
        pred_signal_idx = traj.exp_signal_location_idx[b_idx, n_idx, end_t, 0].to(dtype=torch.long)  # [B, N]
        true_signal_idx = traj.true_signal_idx.to(dtype=torch.long)                    # [B, N]
        final_acc = (pred_signal_idx == true_signal_idx).to(dtype=torch.float32)      # [B, N]
        acc_per_env = final_acc.mean(dim=1)
        overall_acc = final_acc.mean().item()

        avg_final_timestep = traj.stopping_time.to(dtype=torch.float32).mean().item()
        median_final_timestep = traj.stopping_time.to(dtype=torch.float32).median().item()

        print(f'----------------------------------')
        print(
            f'[Eval] Agent: {self.agent_cfg.type} - '
            f'Env: {getattr(self.cfg.environment.robot_env, "type", "RobotEnv")} - '
            f'Horizon: {self.cfg.environment.horizon}'
        )
        print(f'[Eval] Policy mean (env {idenv}): {traj.policy_mean[idenv]}')
       
        print(f'[Eval] Sample env_{idenv} trajectory[0] clipped location history (H x P x 2):')
        print(traj.position_history[idenv, 0].detach().cpu().numpy())
        # Actions for all players on the sampled environment.
        env_actions = traj.exp_actions[idenv]  # [N, H, P]
        # Show one representative trajectory action timeline: [H, P].
        print(f'[Eval] Sample env_{idenv} trajectory[0] actions (H * P):')
        print(env_actions[0].detach().cpu().numpy())
        # Per-player action histogram: only timesteps up to first stop per trajectory (excludes
        # padded zeros after episode end; otherwise full horizon if no stop).
        ea = env_actions.detach().cpu().numpy()
        if self.is_stopping_enabled:
            stop_ix = self.num_actions - 1
            parts: list[np.ndarray] = []
            for n in range(ea.shape[0]):
                ah = ea[n]
                first_stop_h = None
                for h in range(H):
                    if (ah[h] == stop_ix).any():
                        first_stop_h = h
                        break
                if first_stop_h is None:
                    parts.append(ah)
                else:
                    parts.append(ah[: first_stop_h + 1])
            flat_actions = np.concatenate(parts, axis=0) if parts else ea.reshape(0, self.num_players)
        else:
            flat_actions = ea.reshape(-1, self.num_players)
        for p in range(self.num_players):
            counts = np.bincount(flat_actions[:, p], minlength=self.num_actions)
            probs = counts / max(1, counts.sum())
            print(
                f'[Eval] env_{idenv} player_{p} action freq (valid steps only): '
                f'counts={counts.tolist()} probs={np.round(probs, 3).tolist()}'
            )

        # Overall: how often each player is among those who first sampled the stop action
        # (earliest timestep in the episode; ties at the same timestep all count).
        if self.is_stopping_enabled:
            stop_idx = self.num_actions - 1
            acts_all = traj.exp_actions.long()
            Bb, Nb, Hh, Pp = acts_all.shape
            dev = acts_all.device
            t_range = torch.arange(Hh, device=dev, dtype=torch.long).view(1, 1, Hh, 1)
            is_stop = acts_all == stop_idx
            masked_t = torch.where(
                is_stop,
                t_range.expand_as(is_stop),
                torch.full_like(acts_all, Hh),
            )
            first_stop_t = masked_t.min(dim=2).values  # [B, N, P]
            t_min = first_stop_t.min(dim=-1).values  # [B, N]
            valid = t_min < Hh
            tied_first = (first_stop_t == t_min.unsqueeze(-1)) & valid.unsqueeze(-1)
            first_stop_counts = tied_first.sum(dim=(0, 1)).detach().cpu().numpy().astype(int)
            n_traj = Bb * Nb
            n_any_stop = int(valid.sum().item())
            print(
                f'[Eval] First stop (action {stop_idx}) -- count per player (earliest timestep, ties ok): '
                f'{first_stop_counts.tolist()} | trajectories with any stop: {n_any_stop}/{n_traj}'
            )

        print(
            f'[Eval] Signal-location accuracy: overall={overall_acc:.4f} | '
            f'env_{idenv}={acc_per_env[idenv].item():.4f}'
        )
        print(
            f'[Eval] Final timestep: avg={avg_final_timestep:.3f} | '
            f'median={median_final_timestep:.3f} | horizon={H}'
        )

   
        # print(df_exp.loc[idenv].round(2))
        return df_est, df_exp
        
    
    def sample_envs(
        self, B: int, N: int, seed_offset: int = 0
    ) -> VecRobotEnv | BatchedRobotEnv2:
        """Build ``B * N`` robot envs: ``B`` independent seeds, each repeated ``N`` times."""
        robot_cfg = robot_env_config_from_sequential(self.cfg.environment)
        base_seed = (
            int(self.cfg.experiment.seed)
            if getattr(self.cfg.experiment, "seed", None) is not None
            else 0
        )
        return sample_envs_batched(robot_cfg, B, N, base_seed + int(seed_offset))
    
    @torch.no_grad()
    def _log_statistics(self, step: int, training_statistics: Dict[str, List[float]], # mle_rewards: torch.Tensor,
                        probs_i_network: torch.Tensor, true_signal_idx: torch.Tensor,
                        qvalues: torch.Tensor, exploration_probs: torch.Tensor = None, horizons: torch.Tensor = None, end_entropy:torch.Tensor = None):
            """Logs statistics to Wandb

            Args:
                step (int): current training step
                training_statistics (Dict[str, List[float]]): A dictonary containing training statistics
                # mle_rewards (torch.Tensor): The estimated maximum likelihood reward vector
                log_probs_e_network (torch.Tensor): log probabilities of the estimated optimal arm from the e network
                true_signal_idx (torch.Tensor): vector with the true optimal arms
                qvalues (torch.Tensor): qvalues of the exploration policy
                exploration_probs (Optional[torch.Tensor]): exploration probabilities of the true policy. Defaults to None
                horizon (Optional[torch.Tensor]): array of horizons at which the trajectory stopped
            """
            #=========================
           
            estm_rewards_argmax =(probs_i_network.max(dim=-1).values).float().log()
            

            estm_rewards_probs = (probs_i_network.max(dim=-1).values)

            #=========================
    
          
            estm_rewards_argmax = estm_rewards_argmax.detach().cpu().numpy()
            estm_rewards_probs = estm_rewards_probs.detach().cpu().numpy()

            values = {
                "values/avg_qvalues": qvalues.detach().cpu().numpy().mean(),
                "values/std_avg_qvalues": qvalues.detach().cpu().numpy().mean(-1).std(),
                "values/max_qvalues": qvalues.max(-1)[0].mean().detach().cpu().numpy(),
                "values/min_qvalues": qvalues.min(-1)[0].mean().detach().cpu().numpy(),
            }

            performance = {
                #"performance/mle_rewards": mle_rewards.mean().item(),
                "performance/est_rewards_argmax": estm_rewards_argmax.mean(),
                "performance/est_rewards_probs": estm_rewards_probs.mean(),
            }
            training_stats = {
                f"loss/{x}": np.mean(y) for x,y in training_statistics.items()
            }
            end_entropy_stats = {}
            if end_entropy is not None:
                end_entropy_stats = {
                    f"end_entropy/player_{p}": end_entropy[:, p].mean().item() for p in range(self.num_players)
                }
                end_entropy_stats["end_entropy/mean_all_players"] = end_entropy.mean().item()
            policy = {}
            if exploration_probs is not None:
                exploration_probs = exploration_probs.detach().cpu().numpy()
                # Geometric mean for probabilities is better suited
                avg_inf_prob = np.prod(exploration_probs, -1) ** (1 / exploration_probs.shape[-1])
                entropy = -(exploration_probs * np.log(exploration_probs + 1e-9)).sum(-1) /np.log(avg_inf_prob.shape[-1])
                
                policy = {
                    'policy/max_probability': exploration_probs.max(-1).mean(),
                    'policy/min_probability': exploration_probs.min(-1).mean(),
                    'policy/avg_probability': avg_inf_prob.mean(),
                    'policy/std_avg_probability': avg_inf_prob.std(),
                    'policy/entropy': entropy.mean()
                    }
            
            if horizons is not None:
                # `horizons` may come from `traj_current_timestep` which is `torch.long`.
                # Cast to float so `mean()`/`median()` work across PyTorch versions.
                horizons_f = horizons.to(dtype=torch.float32)
                _performance_horizons = {
                    'performance/avg_horizon': horizons_f.mean().item(),
                    'performance/median_horizon': horizons_f.median().item(),
                }
                performance = {**performance, **_performance_horizons}

            wandb.log({
                **performance, **training_stats, **values, **policy, **end_entropy_stats
            }, step=step)

    def _sample_actions(self, state: torch.Tensor, num_rewards: int = 1, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions 

        Args:
            state (torch.Tensor): compressed state
            num_rewards (int): number of rewards per timestep
            greedy (bool, optional): Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple (actions, qvalues, probabilities)
        """
        raise Exception('_sample_actions not implemented')


        



    def train(self, save_path: str):
        # Collect rollouts
        B, N = self.agent_cfg.num_env_instances, self.agent_cfg.num_trajectories
       
        total_train_networks_time = 0.0
        total_non_train_networks_time = 0.0
        total_steps_completed = 0


        step_iter = tqdm(
            range(self.cfg.training.max_training_steps),
            desc="train",
            unit="step",
            total=self.cfg.training.max_training_steps,
            dynamic_ncols=True,
            disable=False,
            mininterval=300.0,
            miniters=500,
        )
        for step in step_iter:
            step_t0 = time.perf_counter()
            # Sample environments
            envs = self.sample_envs(B=B, N=N, seed_offset=step)
            states, dones, stopped = envs.reset()  # [B, P, G, G]
            
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            true_signal_maps = torch.as_tensor(
                envs.get_signal_maps(), dtype=torch.float32, device=self.device
            )

            traj_current_timestep = torch.zeros(B*N, device=self.device, dtype=torch.long)
            location_history = torch.zeros((B * N, self.num_players, self.horizon, 2), dtype=torch.long, device=self.device)
            location_history[:, :, 0, :] = torch.as_tensor(envs.get_positions_batch(), dtype=torch.long, device=self.device)
          

            for t in range(self.horizon):
                with torch.no_grad(), torch.amp.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    
                    actions, action_log_probs = self._sample_actions_decentralized(
                        states, location_history=location_history, timestep=traj_current_timestep, network=self.actor)
                    

                actions = actions.view(-1, self.num_players).long()

                actions_np = actions.detach().cpu().numpy()
                next_states, dones, stopped = envs.step(actions_np)
                next_states = torch.as_tensor(
                    next_states, dtype=torch.float32, device=self.device
                )
                dones = torch.from_numpy(dones).to(self.device)
                stopped = torch.from_numpy(stopped).to(self.device)

                next_positions = torch.as_tensor(envs.get_positions_batch(), dtype=torch.long, device=self.device)
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

            
            t_train_nets_0 = time.perf_counter()
            inference_loss, qloss, max_q_diff, actor_loss, alpha_loss, alpha_value, train_entropy = self._train_networks()
            # Debug: show inference loss returned by `_train_estimation_net`
            
            t_train_nets_1 = time.perf_counter()
            train_networks_time = t_train_nets_1 - t_train_nets_0
            
            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                seq_state = states
                seq_state = seq_state#.sum(dim=1, keepdim=True)   # [B,1,G,G]
                B = seq_state.shape[0]
                i_net_probs = self.i_net.predict_proba(seq_state).view(B, -1)
                predicted_idx = i_net_probs.argmax(dim=-1)   # [B]
                target_idx = true_signal_maps.view(B, -1).argmax(dim=-1)   # [B]
                qvalues = self.critic.forward(states, location_history, timestep=traj_current_timestep)

            op_values, op_idx = i_net_probs.max(dim=-1)
            op = op_values.detach().cpu().float().numpy().round(4)
            op_id = op_idx.detach().cpu().numpy()
            op_tgt_probs,op_tgt_id = self.i_tgt_net.predict_proba(seq_state).view(B, -1).max(dim=-1)
            op_tgt = op_tgt_probs.detach().cpu().float().numpy().round(4)
            op_tgt_id = op_tgt_id.detach().cpu().numpy()
                # log_probs: [B, num_actions, num_players]; sum over actions (dim=1) -> entropy per player [B, P]
            end_entropy = -(action_log_probs.detach() * action_log_probs .detach().exp()).sum(dim=-1)
            if self.is_stopping_enabled:
                cost_updates = self._update_cost(i_net_probs, true_signal_maps)

            q_min = torch.stack(qvalues, dim=0).min(dim=0).values
                
            qv = q_min.detach().cpu().float().numpy().round(2)[0]
            if self.current_step % self.log_interval == 0:
                print(f'\nQV: {qv}', flush=True)
                print(f'OP      : val={op[0]:.4f}, id={op_id[0]} | Mean: {op.mean():.4f}', flush=True)
                print(f'OP  tgt : val={op_tgt[0]:.4f}, id={op_tgt_id[0]} | Mean: {op_tgt.mean():.4f}', flush=True)
                print(f"true_signal_idx: {true_signal_maps[:10].view(-1).argmax(dim=-1)}", flush=True)
                print(f'Traj current timestep: {traj_current_timestep[0]}\n', flush=True)
            # print(f"States[action,reward]: {np.round(states[0].cpu().numpy(), 2)}")


            learning_rates = {
                "lr_exploration_net": self.critic_optimizer.param_groups[0]['lr'],
                "lr_inference_net": self.i_optimizer.param_groups[0]['lr'],
                "lr_actor_net": self.actor_optimizer.param_groups[0]['lr'],
            }
            if hasattr(self, '_annealed_gumbel_tau'):
                learning_rates["gumbel_tau"] = self._annealed_gumbel_tau(self.current_step)
            # Use returned alpha_value if available, otherwise compute it
            if not alpha_value and hasattr(self, 'log_alpha'):
                if self.ent_coef == 'auto':
                    alpha_value = {"alpha_value": [self.alpha.item()]}
                else:
                    alpha_val = self.ent_coef.item() if isinstance(self.ent_coef, torch.Tensor) else self.ent_coef
                    alpha_value = {"alpha_value": [alpha_val]}
            # train_entropy is dict like {'train_entropy': [scalar]}; convert list values to CPU floats for logging
            train_entropy_stats = {
                k: [x.cpu().item() if isinstance(x, torch.Tensor) else float(x) for x in v]
                for k, v in train_entropy.items()
            }

            self._log_statistics(
                step=self.current_step,
                training_statistics={**inference_loss, **qloss, **max_q_diff, **actor_loss, **alpha_loss, **cost_updates, **learning_rates, **alpha_value, **train_entropy_stats},
                # mle_rewards=mle_rewards,
                probs_i_network=i_net_probs,
                true_signal_idx=true_signal_maps,
                qvalues=q_min,
                end_entropy=end_entropy,
                exploration_probs=None,
                horizons=traj_current_timestep)
            self.current_step += 1

            self.scheduler_step()
            if self.current_step % self.cfg.training.save_frequency == 0:
                self.save(save_path)


# ========================================= print time comparison 
            step_total = time.perf_counter() - step_t0
            non_train_networks_time = max(0.0, step_total - train_networks_time)
            total_train_networks_time += train_networks_time
            total_non_train_networks_time += non_train_networks_time
            total_steps_completed += 1
        total_time = total_train_networks_time + total_non_train_networks_time
        train_networks_pct = (100.0 * total_train_networks_time / total_time) if total_time > 0 else 0.0
        non_train_networks_pct = (100.0 * total_non_train_networks_time / total_time) if total_time > 0 else 0.0
        avg_train_networks_time = (
            total_train_networks_time / total_steps_completed if total_steps_completed > 0 else 0.0
        )
        avg_non_train_networks_time = (
            total_non_train_networks_time / total_steps_completed if total_steps_completed > 0 else 0.0
        )
        print(
            "[Timing][training complete] "
            f"steps={total_steps_completed} | "
            f"train_networks_total={total_train_networks_time:.3f}s ({train_networks_pct:.1f}%) | "
            f"non_train_networks_total={total_non_train_networks_time:.3f}s ({non_train_networks_pct:.1f}%) | "
            f"train_networks_avg_per_step={avg_train_networks_time:.3f}s | "
            f"non_train_networks_avg_per_step={avg_non_train_networks_time:.3f}s"
        )
         # =========================================  =========================================


                
        

    def _update_cost(self, probs: torch.Tensor, true_signal_idx: torch.Tensor):


       
        estm_rewards_argmax =(probs.max(dim=-1).values)


        err = self.agent_cfg.confidence_level-estm_rewards_argmax.mean()

        cost_loss = self._cost_multiplier * err.detach()
        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()
        with torch.no_grad():
            # self._cost_multiplier.data.clamp_(min=self.min_cost, max=1)
            self._cost_multiplier.data.clamp_(min=self.min_cost,max=1)

        return {'cost': self._cost_multiplier.item(), 'target_confidence_error': err.abs().item()}


    @abstractmethod
    def _train_rl_nets(self, batch: RobotReplayBufferSamples) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        raise Exception('_train_rl_nets not implemented')

    def _train_estimation_net(self, batch: RobotReplayBufferSamples) -> Dict[str, List[float]]:
       
        self.i_net.train()
        
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # Global map for inference: sum over players -> [B, G, G]
            # obs_map = batch.next_states.sum(dim=1, keepdim=True)   
            obs_map = batch.next_states 
            logits = self.i_net(obs_map)                           # [B,G,G]
            B, G, _ = logits.shape

            target_idx = batch.true_signal_maps.view(B, -1).argmax(dim=-1)  # [B]


            est_loss = F.cross_entropy(
                logits.view(B, -1),   # [B, G*G]
                target_idx
            )

        self.i_optimizer.zero_grad()
        # scale -> backward -> unscale -> clip -> step -> update
        self.i_scaler.scale(est_loss).backward()
        self.i_scaler.unscale_(self.i_optimizer)
        torch.nn.utils.clip_grad_norm_(self.i_net.parameters(),
                                       self.agent_cfg.grad_clipping)
        self.i_scaler.step(self.i_optimizer)
        self.i_scaler.update()
 

        self.i_gradient_step += 1
        if self.i_gradient_step % int(self.agent_cfg.frequency_update_i_network * self.cfg.environment.horizon) == 0:
            self.i_tgt_net.hard_update(self.i_net)
        
        return {'inference_loss': [est_loss.item()]}

    def _train_networks(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        inference_loss, qloss, max_q_diff, actor_loss, alpha_loss, alpha_value, train_entropy = {}, {}, {}, {}, {}, {}, {}
        
        if len(self.buffer) >  self.agent_cfg.initial_buffer_bootstrap_num_batches * self.agent_cfg.batch_size * self.agent_cfg.replay_ratio:
            # Don't add last transitions to avoid leakage! if the capacity of the E network is sufficiently large it will overfit new samples
            for _ in range(self.agent_cfg.replay_ratio):
                inference_loss = self._train_estimation_net(self.buffer.sample(self.agent_cfg.batch_size, add_last_transtions=False))

            if len(self.buffer) > self.agent_cfg.inference_network_bootstrap_num_batches * self.agent_cfg.batch_size * self.agent_cfg.replay_ratio:
                for _ in range(self.agent_cfg.replay_ratio):
                    
                    qloss, max_q_diff, actor_loss, alpha_loss, alpha_value, train_entropy = self._train_rl_nets(self.buffer.sample(self.agent_cfg.batch_size, add_last_transtions=False) )
                    self.gradient_step += 1
        return inference_loss, qloss, max_q_diff, actor_loss, alpha_loss, alpha_value, train_entropy
            
    
    @abstractmethod
    def save(self, path: str):
        raise Exception('Save method not implemented in BaseOffPolicyAgent')

    @staticmethod
    @abstractmethod
    def load(path: str):
        raise Exception('Load method not implemented in BaseOffPolicyAgent')


# Backward-compat alias -- existing code that imports RobotBaseOffPolicyAgentSAC still works.
RobotBaseOffPolicyAgentSAC = RobotBaseOffPolicyAgent