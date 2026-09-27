from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from icmape.dataclasses import Config
from icmape.agents.robot.buffers import RobotReplayBufferSamples
from icmape.agents.robot.config import RobotMULTICPE_SAC_Config
from icmape.agents.robot.sequential.base_offpolicy_agent_sac import RobotBaseOffPolicyAgent
from icmape.models.robot_net.builder import create_actor_network_td3


class MULTICPE_TD3_ROBOT(RobotBaseOffPolicyAgent):
    """Twin-critic TD3 for the robot wildfire exploration environment.

    Differences from ``MULTICPE_SAC_ROBOT``:
    - No entropy temperature alpha -- critic target is plain Q (no soft-Q).
    - Delayed actor updates every ``td3_policy_delay`` gradient steps.
    - Actor uses ``ActorLSTMTD3``: returns ``(gumbel_actions, policy_probs, logits)``.
    - Actor loss: ``-(gumbel_actions * Q_min).sum()`` with straight-through gradient
      when ``hard=True`` (same pattern as bandit ``MULTICPE_TD3``).
    """

    agent_cfg: RobotMULTICPE_SAC_Config

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.num_players = self.cfg.environment.num_players
        self.num_actions = (
            self.cfg.environment.num_actions
            + int(self.cfg.environment.enable_stopping_action)
        )
        self.decentralized = self.cfg.environment.decentralized
        self.policy_delay = getattr(self.agent_cfg.actor_network, "td3_policy_delay", 2)
        self._actor_loss_logged = 0.0
        self._train_entropy_logged = 0.0

    # ------------------------------------------------------------------
    # Actor factory hook -- returns ActorLSTMTD3 instead of ActorLSTM.
    # ------------------------------------------------------------------
    def _create_actor_network(
        self, robot_state_dim: int, architecture_type: str
    ) -> torch.nn.Module:
        shared_critic = getattr(self.agent_cfg.actor_network, "shared_critic", False)
        return create_actor_network_td3(
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

    # ------------------------------------------------------------------
    # Algorithm-specific init: TD3 -- no log_alpha; store Gumbel tau params.
    # ------------------------------------------------------------------
    def _init_algorithm_components(self):
        self.tau_start = float(
            getattr(self.agent_cfg.actor_network, "gumbel_temperature", 1.0)
        )
        self.tau_min = float(
            getattr(self.agent_cfg.actor_network, "gumbel_tau_min", 0.1)
        )
        self.tau_anneal_total_steps = self.cfg.training.max_training_steps // 2

        # Actor target network (TD3 standard: use lagged actor for Bellman targets).
        robot_state_dim = self.num_players * self.grid_size * self.grid_size
        architecture_type = self.agent_cfg.architecture_type
        self.actor_tgt = self._create_actor_network(
            robot_state_dim, architecture_type
        ).to(self.device)
        self.actor_tgt.hard_update(self.actor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _annealed_gumbel_tau(self, step: int) -> float:
        if self.tau_anneal_total_steps <= 1:
            return self.tau_min
        t = min(int(step), self.tau_anneal_total_steps - 1)
        frac = t / (self.tau_anneal_total_steps - 1)
        tau = self.tau_start + (self.tau_min - self.tau_start) * frac
        return max(self.tau_min, float(tau))

    def _set_actor_gumbel_tau(self, step: int) -> None:
        self.actor._gumbel_tau = self._annealed_gumbel_tau(step)

    def _get_decentralized_qvalues(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        network: torch.nn.Module,
        location_history: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        others_actions: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        act = actions if actions is not None else others_actions
        return network.forward(
            state, location_history=location_history, timestep=timestep, action=act
        )

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------
    def _sample_actions_decentralized(
        self,
        state: torch.Tensor,
        location_history: torch.Tensor,
        timestep: torch.Tensor,
        network: torch.nn.Module = None,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(actions [B,P], log_probs [B,P,A+1])``.

        During warm-up, samples uniformly. Once training has started, calls
        the TD3 actor (no ``hard``; straight-through only used inside
        ``_train_rl_nets``).
        """
        if self.has_training_started or greedy:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                B = state.shape[0]
                P = self.num_players
                # hard=False for rollout -- we just need action samples.
                _gumbel, policy_probs, _ = network.forward(
                    state, timestep, location_history, hard=False
                )  # policy_probs: [B,P,A+1]
                log_pi = policy_probs.clamp_min(1e-8).log()  # [B,P,A+1]
                if greedy:
                    sampled_actions = policy_probs.argmax(dim=-1)  # [B,P]
                else:
                    sampled_actions = torch.distributions.Categorical(
                        probs=policy_probs
                    ).sample()  # [B,P]
            return sampled_actions, log_pi
        else:
            B = state.shape[0]
            P = self.num_players
            A = self.cfg.environment.num_actions
            probs = torch.full((B, P, A), 1.0 / A, device=self.device)
            actions = torch.multinomial(
                probs.view(-1, A), num_samples=1, replacement=True
            ).view(B, P).to(device=self.device, dtype=torch.long)
            return actions, probs.log()

    # ------------------------------------------------------------------
    # TD3 training loss
    # ------------------------------------------------------------------
    def _train_rl_nets(
        self, batch: RobotReplayBufferSamples
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, List[float]],
    ]:
        """Twin-critic TD3 update with delayed actor (Gumbel straight-through)."""

        self._set_actor_gumbel_tau(self.current_step)

        num_players = self.num_players
        num_actions = self.num_actions
        B = batch.states.shape[0]

        # -- BELLMAN TARGET (no entropy term) -----------------------------
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=self.use_amp
        ):
            next_states = batch.next_states
            current_states = batch.states

            # Greedy next actions from target actor (TD3: use lagged network for targets).
            next_actions_full, _ = self._sample_actions_decentralized(
                next_states,
                location_history=batch.location_history,
                timestep=batch.traj_current_timestep + 1,
                network=self.actor_tgt,
            )  # [B, P]

            next_q_all_list = self._get_decentralized_qvalues(
                next_states,
                timestep=batch.traj_current_timestep + 1,
                network=self.critic_tgt,
                location_history=batch.location_history,
                actions=next_actions_full,
            )  # list of [B, P, A+1]

            next_q_all = torch.min(
                torch.stack(next_q_all_list, dim=0), dim=0
            ).values  # [B, P, A+1]
            next_q = next_q_all.gather(
                -1, next_actions_full.unsqueeze(-1)
            ).squeeze(-1)  # [B, P]

            probs_next = self.i_tgt_net.predict_proba(next_states).view(B, -1)
            max_probs = probs_next.max(dim=-1).values
            log_probs_next = torch.log(torch.clamp(max_probs, min=1e-10))
            rewards_next = log_probs_next.detach()

            if self.is_stopping_enabled:
                cost = -(1.0 - batch.dones.float()) * self.cost_per_timestep
            else:
                cost = torch.zeros_like(rewards_next)

            d = batch.dones.float()
            r_t = (cost + d * rewards_next).unsqueeze(1).expand(-1, num_players)
            r_t_check = (d * rewards_next).unsqueeze(1).expand(-1, num_players)

            tgt_val = r_t + self.agent_cfg.gamma * (1.0 - d.unsqueeze(1)) * next_q
            tgt_val_check = (
                r_t_check + self.agent_cfg.gamma * (1.0 - d.unsqueeze(1)) * next_q
            )
            tgt_diff = tgt_val - tgt_val_check

        # -- CRITIC UPDATE -------------------------------------------------
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            qvalues_list = self._get_decentralized_qvalues(
                current_states,
                timestep=batch.traj_current_timestep,
                network=self.critic,
                location_history=batch.location_history,
                actions=batch.actions,
            )  # list of [B, P, A+1]

            max_q_diff = (
                torch.abs(qvalues_list[0] - qvalues_list[1]).max().item()
                if len(qvalues_list) == 2
                else 0.0
            )

            critic_loss = sum(
                F.mse_loss(
                    qv.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1),  # [B, P]
                    tgt_val.detach(),
                )
                for qv in qvalues_list
            )

        self.critic_optimizer.zero_grad()
        self.q_scaler.scale(critic_loss).backward()
        self.q_scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.agent_cfg.grad_clipping
        )
        self.q_scaler.step(self.critic_optimizer)
        self.q_scaler.update()

        # -- DELAYED ACTOR UPDATE (Gumbel straight-through) ----------------
        if self.gradient_step % self.policy_delay == 0:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # hard=True -> one-hot Gumbel sample with straight-through gradient.
                gumbel_actions, policy_probs, policy_logits = self.actor.forward(
                    current_states,
                    batch.traj_current_timestep,
                    batch.location_history,
                    hard=True,
                )  # each [B, P, A+1]

                q_list = self._get_decentralized_qvalues(
                    current_states,
                    timestep=batch.traj_current_timestep,
                    network=self.critic,
                    location_history=batch.location_history,
                    actions=gumbel_actions.argmax(dim=-1),  # [B, P] integer actions
                )
                q_min = torch.min(
                    torch.stack(q_list, dim=0), dim=0
                ).values  # [B, P, A+1]
                q_min = q_min.view(B, num_players, num_actions)

                # Actor loss: -Q(s, a_gumbel) via straight-through inner product.
                q_pi = (q_min * gumbel_actions).sum(dim=-1)  # [B, P]
                actor_loss_t = -q_pi.sum(dim=-1).mean()

                # Logit regularisation (same as bandit TD3).
                actor_loss_t = actor_loss_t + 1e-3 * (policy_logits ** 2).mean()

                log_pi = F.log_softmax(policy_logits, dim=-1)
                entropy_t = -(policy_probs * log_pi).sum(dim=-1).mean()

                if self.current_step % self.log_interval == 0:
                    print("[Batch[0] Info]")
                    print(f"  Timestep:       {batch.traj_current_timestep[0].item()}")
                    print(f"  Q-values:       {q_min[0].detach().cpu().numpy().round(4)}")
                    print(f"  Policy probs:   {policy_probs[0].detach().cpu().numpy().round(4)}")
                    print("=" * 40 + "\n")

            self.actor_optimizer.zero_grad()
            self.actor_scaler.scale(actor_loss_t).backward()
            self.actor_scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.agent_cfg.grad_clipping
            )
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()
            self._actor_loss_logged = float(actor_loss_t.detach().item())
            self._train_entropy_logged = float(entropy_t.detach().item())
            self.actor_tgt.soft_update(self.actor, self.agent_cfg.polyak_update_q_network)

        self.critic_tgt.soft_update(self.critic, self.agent_cfg.polyak_update_q_network)

        return (
            {"value_loss": critic_loss.item(), "tgt_diff": [tgt_diff.mean().item()]},
            {"max_q_diff": [max_q_diff]},
            {"actor_loss": self._actor_loss_logged},
            {},  # alpha_loss -- TD3 has no entropy temperature
            {},  # alpha_value -- TD3 has no entropy temperature
            {"train_entropy": [self._train_entropy_logged]},
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        model = {
            "cfg": self.cfg,
            "critic": self.critic.state_dict(),
            "Enet": self.i_net.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "step": self.current_step,
            "cost_multiplier": self._cost_multiplier.data,
            "cost_optimizer_state_dict": self.cost_optimizer.state_dict(),
        }
        filename = (
            f"{path}/{self.cfg.agent.type}_{self.cfg.agent.architecture_type}"
            f"_ts{self.tau_start}_te{self.tau_min}"
            f"_{self.current_step}_seed{self.cfg.experiment.seed}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str, device: torch.device) -> "MULTICPE_TD3_ROBOT":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = MULTICPE_TD3_ROBOT(data["cfg"], device)
        agent.critic.load_state_dict(data["critic"])
        agent.i_net.load_state_dict(data["Enet"])
        agent.critic_tgt.load_state_dict(data["critic"])
        agent.i_tgt_net.load_state_dict(data["Enet"])
        agent.actor.load_state_dict(data["actor"])
        actor_tgt_state = data.get("actor_tgt", data["actor"])
        agent.actor_tgt.load_state_dict(actor_tgt_state)
        if "step" in data:
            agent.current_step = data["step"]
        return agent
