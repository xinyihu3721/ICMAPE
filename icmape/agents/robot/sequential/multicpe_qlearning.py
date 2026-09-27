"""MULTICPE Q-learning agent for the robot wildfire environment.

Implements decentralised Double DQN on top of ``RobotBaseOffPolicyAgentQlearning``.

Supports two aggregation modes:
  - ``VDN_mode=True``  : Q-values are summed over players (VDN).
  - ``VDN_mode=False`` : Per-player Q-targets (independent learners with shared reward).

Also supports ``uniform_action_only`` (inference training only, Q-network never updated).
"""
from __future__ import annotations

import pickle
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from icmape.dataclasses import Config
from icmape.agents.robot.buffers import RobotReplayBufferSamples
from icmape.agents.robot.sequential.base_offpolicy_agent_qlearning import (
    RobotBaseOffPolicyAgentQlearning,
)


class MULTICPE_QLEARNING_ROBOT(RobotBaseOffPolicyAgentQlearning):
    """Multi-player Constrained Exploration with Q-learning for the robot environment.

    Action selection:
      - Epsilon-greedy over the min of the multi-head critic Q-values.
      - During the bootstrap phase (``uniform_action_only`` or buffer not yet large
        enough) uniform random actions are used.

    Critic update (Double DQN):
      - Online network selects next action; target network evaluates its value.
      - VDN mode: Q-values summed over players -> scalar target per transition.
      - Per-player mode: per-player Q-target broadcast from a shared scalar reward.

    No Q-mixer is used (robot setting uses the CNN-LSTM critic directly).
    """

    def __init__(self, cfg: Config, device: torch.device | str):
        super().__init__(cfg, device)

        # Re-read after super().__init__ because reset_networks was already called.
        self.num_players = self.cfg.environment.num_players
        self.num_actions = self.cfg.environment.num_actions + int(
            self.cfg.environment.enable_stopping_action
        )
        self.decentralized = self.cfg.environment.decentralized

        # VDN mode: aggregate Q-values by summing over players.
        self.VDN_mode: bool = getattr(
            self.agent_cfg.critic_network, "VDN_mode", False
        )

        # Pre-build player one-hot cache for potential decentralised trajectory use.
        self._player_ids_onehot_cache = self._build_player_ids_onehot_cache()

    # ------------------------------------------------------------------
    # One-hot player ID helpers (kept for compatibility / future use)
    # ------------------------------------------------------------------

    def _build_player_ids_onehot_cache(self) -> torch.Tensor:
        """Return ``[B_cap, P, P]`` on device, expanded at call time."""
        P = self.num_players
        B_cap = max(int(self.agent_cfg.batch_size), int(self.total_batch_size))
        row = torch.eye(P, dtype=torch.float32, device=self.device)
        return row.unsqueeze(0).expand(B_cap, P, P)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        model = {
            "cfg": self.cfg,
            "critic": self.critic.state_dict(),
            "Enet": self.i_net.state_dict(),
            "step": self.current_step,
            "cost_multiplier": self._cost_multiplier.data,
        }
        filename = (
            f"{path}/{self.cfg.agent.type}_{self.cfg.agent.architecture_type}"
            f"_{self.current_step}_seed{self.cfg.experiment.seed}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str, device: torch.device) -> "MULTICPE_QLEARNING_ROBOT":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = MULTICPE_QLEARNING_ROBOT(data["cfg"], device)
        agent.critic.load_state_dict(data["critic"])
        agent.critic_tgt.load_state_dict(data["critic"])
        agent.i_net.load_state_dict(data["Enet"])
        agent.i_tgt_net.load_state_dict(data["Enet"])
        if "step" in data:
            agent.current_step = data["step"]
        if "cost_multiplier" in data:
            with torch.no_grad():
                agent._cost_multiplier.copy_(data["cost_multiplier"])
        return agent

    # ------------------------------------------------------------------
    # Q-value computation
    # ------------------------------------------------------------------

    def _get_decentralized_qvalues(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        network: torch.nn.Module,
        location_history: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Forward pass through the Q-network.

        Args:
            state:            ``[B, P, G, G]`` belief maps.
            timestep:         ``[B]`` current trajectory timestep.
            network:          Online or target Q-network.
            location_history: ``[B, P, H, 2]`` position history.

        Returns:
            One-element list containing ``[B, P, num_actions]`` Q-values.
        """
        return network.forward(state, location_history=location_history, timestep=timestep)

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_actions_decentralized(
        self,
        state: torch.Tensor,
        location_history: torch.Tensor,
        timestep: torch.Tensor,
        greedy: bool = False,
        exploration_epsilon: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Epsilon-greedy action selection using the min-over-heads Q-values.

        Returns:
            actions:  ``[B, P]`` int64
            qvalues:  ``[B, P*A]`` float (for logging / target computation)
        """
        B = state.shape[0]

        if (self.has_training_started or greedy) and not self.uniform_action_only:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                qvalues = self._get_decentralized_qvalues(
                    state, timestep, self.critic, location_history=location_history
                )
            
            actions = qvalues.argmax(dim=-1)                         # [B, P]

            if (
                not greedy
                and exploration_epsilon is not None
                and exploration_epsilon > 0.0
            ):
                mask = torch.rand(B, device=self.device) < exploration_epsilon
                n_explore = int(mask.sum().item())
                if n_explore > 0:
                    random_acts = torch.randint(
                        0, self.num_actions, (n_explore, self.num_players),
                        device=self.device,
                    )
                    actions = actions.clone()
                    actions[mask] = random_acts

            return actions.to(dtype=torch.long), qvalues.view(B, -1)

        else:
            # Uniform random over base actions (no stop action during exploration).
            A_base = self.cfg.environment.num_actions
            probs = torch.full(
                (B * self.num_players, A_base), 1.0 / A_base, device=self.device
            )
            actions = torch.multinomial(probs, num_samples=1).view(
                B, self.num_players
            )
            # Compute Q-values for logging even under uniform policy.
            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                qvalues = self._get_decentralized_qvalues(
                    state, timestep, self.critic, location_history=location_history
                )
            return actions.to(dtype=torch.long), qvalues.view(B, -1)

    # ------------------------------------------------------------------
    # RL network update (Double DQN)
    # ------------------------------------------------------------------

    def _train_rl_nets(self, batch: RobotReplayBufferSamples) -> Dict[str, List[float]]:
        """Double DQN update with optional VDN aggregation.

        Target formula (per player unless VDN):
          y = r_t + gamma (1 - d) Q_tgt(s', argmax_a Q_online(s', a))

        where ``r_t = cost + d * log P_inference(fire_location | s')``.
        """
        num_players = self.num_players
        B = batch.states.shape[0]

        # ----------------------------------------------------------------
        # Build target values (no gradient)
        # ----------------------------------------------------------------
       
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=self.use_amp
        ):
            # --- Double DQN: online net selects best next action ---
            next_q_online = self._get_decentralized_qvalues(
                batch.next_states,
                batch.traj_current_timestep + 1,
                self.critic,
                location_history=batch.location_history,
            )
            # min over critic heads -> [B, P, A]
            next_act = next_q_online.argmax(dim=-1, keepdim=True)  # [B, P, 1]

            # --- Target net evaluates that action ---
            next_q_tgt = self._get_decentralized_qvalues(
                batch.next_states,
                batch.traj_current_timestep + 1,
                self.critic_tgt,
                location_history=batch.location_history,
            )

            next_q = next_q_tgt.gather(-1, next_act).squeeze(-1)  # [B, P]

            # --- Terminal reward: log-confidence from inference target net ---
            log_probs_next = torch.log(
                self.i_tgt_net.predict_proba(batch.next_states).view(B, -1).max(dim=-1).values
                + 1e-9
            )
            rewards_next = log_probs_next.detach()  # [B]

            # --- Per-step cost for non-terminal transitions ---
            if self.is_stopping_enabled:
                cost = -(1.0 - batch.dones.float()) * self.cost_per_timestep
            else:
                cost = torch.zeros_like(rewards_next)

            d = batch.dones.float()  # [B]

            if self.VDN_mode:
                # Sum Q-values over players; scalar target per transition.
                r_t = cost + d * rewards_next                    # [B]
                next_q_agg = next_q.sum(dim=-1)                  # [B]
                tgt_val = r_t + self.agent_cfg.gamma * (1 - d) * next_q_agg  # [B]
            else:
                # Per-player targets; broadcast scalar reward to each player.
                r_scalar = cost + d * rewards_next               # [B]
                r_t = r_scalar.unsqueeze(1).expand(-1, num_players)  # [B, P]
                d_p = d.unsqueeze(1)                             # [B, 1]
                tgt_val = r_t + self.agent_cfg.gamma * (1 - d_p) * next_q  # [B, P]

        # ----------------------------------------------------------------
        # Critic loss
        # ----------------------------------------------------------------
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            cur_q = self._get_decentralized_qvalues(
                batch.states,
                batch.traj_current_timestep,
                self.critic,
                location_history=batch.location_history,
            )

            if self.VDN_mode:
                # Sum gathered Q over players -> [B] per critic head
                cur_q_sel = cur_q.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
                    
            else:
                # Per-player gathered Q -> [B, P] per critic head
                cur_q_sel = cur_q.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
                    

            qloss =  torch.nn.functional.mse_loss(cur_q_sel, tgt_val.detach())
        # if self.VDN_mode:
        #         # Prevent double accounting
        #         idxs = ~(batch.actions == self.num_actions - 1).any(dim=-1)
        #         qnext = self._get_decentralized_qvalues(batch.next_states[idxs], batch.traj_current_timestep[idxs]+1, self.critic, location_history=batch.location_history[idxs])[..., -1].mean(dim=-1)
        #         qloss_stopping_current = torch.nn.functional.mse_loss(qnext, rewards_next[idxs])
        #         qloss = qloss +  qloss_stopping_current

        self.critic_optimizer.zero_grad()
        self.q_scaler.scale(qloss).backward()
        self.q_scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.agent_cfg.grad_clipping
        )
        self.q_scaler.step(self.critic_optimizer)
        self.q_scaler.update()

        self.critic_tgt.soft_update(self.critic, self.agent_cfg.polyak_update_q_network)

        return {"value_loss": [qloss.item()]}
