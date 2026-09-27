from __future__ import annotations
import torch
import pickle
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from icmape.dataclasses import Config

from icmape.agents.bandit.config import BaseOffPolicyAgentConfig
from icmape.agents.bandit.buffers import ReplayBufferSamples
from icmape.agents.bandit.sequential.base_offpolicy_agent_td3 import BaseOffPolicyAgent, HistoryData

class MULTICPE_TD3(BaseOffPolicyAgent):
    """
    Multi-agent TD3-style training: twin Q-targets, deterministic greedy next actions in the Bellman target,
    and a delayed actor that maximizes expected Q under the current softmax policy (Q detached).
    """
    agent_cfg: BaseOffPolicyAgentConfig

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.num_players = self.cfg.environment.num_players
        self.num_actions = self.cfg.environment.num_actions +  int(self.cfg.environment.enable_stopping_action)  
        self.decentralized = self.cfg.environment.decentralized
        self.policy_delay = self.agent_cfg.actor_network.td3_policy_delay
        self._actor_loss_logged = 0.0
        self._train_entropy_logged = 0.0
        self._player_ids_onehot_cache = self.get_player_ids_onehot_cache()
        P = self.num_players
        self._player_reward_gather_idx = torch.arange(P, device=self.device, dtype=torch.long).view(1, 1, P, 1)

    def get_player_ids_onehot_cache(self) -> torch.Tensor:
        """[B_max, P, P] on ``self.device``; B_max = max(replay batch, rollout batch)."""
        P = self.num_players
        device = self.device
        B_cap = max(int(self.agent_cfg.batch_size), int(self.total_batch_size))
        row = torch.eye(P, dtype=torch.float32, device=device)
        return row.unsqueeze(0).expand(B_cap, P, P)

    def save(self, path: str):
        model = {'cfg': self.cfg,
                 'critic': self.critic.state_dict(),
                 'Enet': self.i_net.state_dict(),
                 'actor': self.actor.state_dict(),
                 'actor_tgt': self.actor_tgt.state_dict(),
                 'step': self.current_step,
                 'cost_multiplier': self._cost_multiplier.data,
                 'cost_optimizer_state_dict': self.cost_optimizer.state_dict()}
        filename = f'{path}/{self.cfg.agent.type}_{self.cfg.agent.architecture_type}_{self.current_step}_seed{self.cfg.experiment.seed}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str, device: torch.device) -> 'MULTICPE_TD3':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = MULTICPE_TD3(data['cfg'], device)
        agent.critic.load_state_dict(data['critic'])
        agent.i_net.load_state_dict(data['Enet'])
        agent.critic_tgt.load_state_dict(data['critic'])
        agent.i_tgt_net.load_state_dict(data['Enet'])
        agent.actor.load_state_dict(data['actor'])
        actor_tgt_state = data.get('actor_tgt', data['actor'])
        agent.actor_tgt.load_state_dict(actor_tgt_state)
        if 'step' in data:
            agent.current_step = data['step']
        return agent

    def _onehot_player_ids(self, batch_size: int, horizon: int, num_players: int, device: torch.device) -> torch.Tensor:
        """Return [B, H, P, P] one-hot player-id tensor.

        Uses the pre-built cache when B fits; generates on-the-fly for larger
        batch sizes (e.g. during evaluation with BATCH_SIZE > training batch).
        """
        B, H, P = int(batch_size), int(horizon), int(num_players)
        c = self._player_ids_onehot_cache
        if B <= c.shape[0]:
            return c[:B].unsqueeze(1).expand(B, H, P, P)
        row = torch.eye(P, dtype=torch.float32, device=device)
        return row.view(1, 1, P, P).expand(B, H, P, P)

    def _get_others_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        For every player, returns a vector containing the actions of all OTHER players (raw integer action indices).
        Vectorized: no Python loop, single indexed gather.

        Output shape: [B, P, P-1], where B = batch size, P = num_players.
        """
        B, P = actions.shape
        device = actions.device
        # other_idx[p, k] = index of k-th "other" player for player p: [0..p-1, p+1..P-1]
        k_idx = torch.arange(P - 1, device=device).unsqueeze(0).expand(P, -1)
        other_idx = k_idx + (k_idx >= torch.arange(P, device=device).unsqueeze(1)).long()
        return actions[:, other_idx]

    def _get_decetralized_trajectory(self, state: torch.Tensor, num_rewards: int = 1) -> torch.Tensor:
        """
        Returns decentralized trajectory representation for each player.
        """

        device = self.device
        B, H, _ = state.shape
        P = self.num_players
        A = self.base_num_actions
        state = state.to(device)

        with torch.amp.autocast(device, enabled=self.use_amp):
            actions_part = state[..., :P*A]  # one hot encoded action P*A for all players
            rewards_part = state[..., -num_rewards:]  # last num_rewards dimensions are rewards
           
            actions = actions_part.view(B, H, P, A).contiguous()

            # Create one-hot encoded player IDs: [B, H, P, P]
            player_ids_onehot = self._onehot_player_ids(B, H, P, device)
            

            rewards_part = rewards_part.unsqueeze(2).expand(B, H, P, num_rewards)
            rewards_p = rewards_part.gather(-1, index=self._player_reward_gather_idx.expand(B, H, P, 1))

            state_p = torch.cat([player_ids_onehot, actions, rewards_p], dim=-1)
            ind_traj = state_p.permute(0, 2, 1, 3).contiguous().view(B * P, H, P + A + 1)
        return ind_traj

    def _get_decentralized_qvalues(self, state: torch.Tensor, timestep: torch.Tensor, network: torch.nn.Module, num_rewards: int = 1,decentralize_state: bool = True, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if decentralize_state:
            inputs = self._get_decetralized_trajectory(state, num_rewards)
            timestep = timestep.unsqueeze(1).expand(-1, self.num_players).reshape(state.shape[0]*self.num_players)
        else:
            player_ids_onehot = self._onehot_player_ids(state.shape[0], state.shape[1], self.num_players, state.device)
            state_repeat = state.unsqueeze(-1).expand(-1, -1, -1, self.num_players)
            rest_dim = state.shape[2:]
            state_p = torch.cat([player_ids_onehot, state_repeat], dim=-2).permute(0,3,1,2)
            state_dim = state_p.shape[2:]
            inputs = state_p.reshape(state_p.shape[0] * state_p.shape[1], *state_dim)
            timestep = timestep.unsqueeze(1).expand(-1, self.num_players).reshape(state.shape[0]*self.num_players)
        if actions is not None:
            q_flat = network.forward(inputs, timestep, action=actions)
        else:
            q_flat = network.forward(inputs, timestep)
        return q_flat

    def _get_decentralized_qvalues_step(self, state: torch.Tensor, timesteps: torch.Tensor, hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]], feat_hists: Optional[List[torch.Tensor]], network: torch.nn.Module, num_rewards: int = 1, decentralize_state: bool = True) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get decentralized Q-values using step-by-step processing (for LSTM networks).
        
        Args:
            state: [B, H, state_dim] state tensor
            timesteps: [B] current timesteps
            hidden_states: List of hidden states for each critic, or None
            feat_hists: List of feature histories for each critic, or None
            network: Critic network (should have step method)
            num_rewards: Number of rewards
            decentralize_state: Whether to decentralize the state
            
        Returns:
            qvalues_list: List of [B*P, num_actions] Q-values for each critic
            feat_hists_updated: List of updated feature histories
            hidden_states_updated: List of updated hidden states
        """
        B = timesteps.shape[0]
        P = self.num_players
        Bflat = B * P
        device = self.device
        t_player = timesteps.unsqueeze(1).expand(B, P).reshape(Bflat)
        
        if decentralize_state:
            inputs = self._get_decetralized_trajectory(state, num_rewards)
        else:
            player_ids_onehot = self._onehot_player_ids(state.shape[0], state.shape[1], self.num_players, state.device)
            state_repeat = state.unsqueeze(-1).expand(-1, -1, -1, self.num_players)
            rest_dim = state.shape[2:]
            state_p = torch.cat([player_ids_onehot, state_repeat], dim=-2).permute(0,3,1,2)
            state_dim = state_p.shape[2:]
            inputs = state_p.reshape(state_p.shape[0] * state_p.shape[1], *state_dim)
        
        row_idx = torch.arange(Bflat, device=device)
        x_t = inputs[row_idx, t_player, :]  # [Bflat, d]
        
        # Call network.step() which returns (q_values_list, feat_hists_updated, hidden_states_updated)
        qvalues_list, feat_hists_updated, hidden_states_updated = network.step(
            x_t=x_t,
            hidden_states=hidden_states,
            feat_hists=feat_hists,
            timesteps=t_player,
        )
        
        return qvalues_list, feat_hists_updated, hidden_states_updated

 
    def _train_rl_nets(self, batch: ReplayBufferSamples) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """Twin critic TD target with greedy next actions; delayed actor maximizes E_pi[Q] with Q detached."""
        num_players = self.cfg.environment.num_players
        num_actions = self.cfg.environment.num_actions + 1
        
        B = batch.states.shape[0]

        # Bellman target: y = r + gamma(1-d) Q_tgt(s', argmax pi(*|s'))
        with torch.no_grad(), torch.amp.autocast(self.device, enabled=self.use_amp):
            next_states = batch.next_states
            current_states = batch.states

            next_actions_full, _, _ = self._sample_actions_decentralized(
                next_states,
                timestep=batch.traj_current_timestep + 1,
                network=self.actor_tgt,
                num_rewards=num_players,
            )
            next_actions = next_actions_full.reshape(-1)

            next_q_all_list = self._get_decentralized_qvalues(
                next_states,
                timestep=batch.traj_current_timestep + 1,
                network=self.critic_tgt,
                num_rewards=num_players,
                decentralize_state=False,
                actions=next_actions_full,
            )
            next_q_all = torch.min(torch.stack(next_q_all_list, dim=0), dim=0).values
            next_q = next_q_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1).view(B, num_players)

            rewards_next = (
                self.i_tgt_net.forward(next_states, batch.traj_current_timestep + 1).max(dim=-1).values
            ).sum(dim=-1).detach()

            if self.is_stopping_enabled:
                cost = -(1.0 - batch.dones.float()) * self.cost_per_timestep
            else:
                cost = torch.zeros_like(rewards_next)

            d = batch.dones.float()
            r_scalar = cost + d * rewards_next
            r_t = r_scalar.unsqueeze(1).expand(-1, num_players)
            r_scalar_check = d * rewards_next
            r_t_check = r_scalar_check.unsqueeze(1).expand(-1, num_players)

            tgt_val = r_t + self.agent_cfg.gamma * (1.0 - d.unsqueeze(1)) * next_q
            tgt_val_check = r_t_check + self.agent_cfg.gamma * (1.0 - d.unsqueeze(1)) * next_q
            tgt_diff = tgt_val - tgt_val_check
            



        # ================== CRITIC UPDATE ==================
        with torch.amp.autocast(self.device, enabled=self.use_amp):

            qvalues_list = self._get_decentralized_qvalues(
                current_states,
                timestep=batch.traj_current_timestep,
                network=self.critic,
                num_rewards=num_players,
                decentralize_state=False,
                actions=batch.actions,
            )
            
            # Calculate max difference between two Q network outputs
            max_q_diff = torch.abs(qvalues_list[0] - qvalues_list[1]).max().item() if len(qvalues_list) == 2 else 0.0

            B = current_states.shape[0]  # batch size
            critic_loss = sum(
                F.mse_loss(
                    qv.view(B, num_players, num_actions).gather(
                        -1, batch.actions.unsqueeze(-1)
                    ).squeeze(-1),
                    tgt_val.detach(),
                )
                for qv in qvalues_list
            )

        self.critic_optimizer.zero_grad()
        self.q_scaler.scale(critic_loss).backward()
        self.q_scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.agent_cfg.grad_clipping)
        self.q_scaler.step(self.critic_optimizer)
        self.q_scaler.update()

        # Actor: policy delay + logged losses (see branch below).
        # When the actor is skipped (policy delay), keep logging actor_loss / train_entropy from the most recent update (both start at 0).
        if self.gradient_step % self.policy_delay == 0:
            with torch.amp.autocast(self.device, enabled=self.use_amp):
                # Old: expected Q under full policy -- actor_loss_t = -(probs * q_min).sum(-1).sum(-1).mean()

                actions_gumbel, policy_probs, policy_logits = self._sample_actions_decentralized(
                    current_states,
                    timestep=batch.traj_current_timestep,
                    network=self.actor,
                    num_rewards=num_players,
                    hard=True,
                )  # [B, P, A+1]
              
                q_list = self._get_decentralized_qvalues(
                    current_states,
                    timestep=batch.traj_current_timestep,
                    network=self.critic,
                    num_rewards=num_players,
                    decentralize_state=False,
                    actions=actions_gumbel.argmax(dim=-1),
                )
                q_min = torch.min(torch.stack(q_list, dim=0), dim=0).values
                q_min = q_min.view(B, num_players, num_actions)
                # probs = policy_probs.view(B, num_players, num_actions)
                # actor_loss_t = -(probs * q_min).sum(dim=-1).sum(dim=-1).mean()
                #probs = policy_probs.view(B, num_players, num_actions)
              
                q_pi = (q_min * actions_gumbel).sum(dim=-1)
                actor_loss_t = -q_pi.sum(dim=-1).mean()
                policy_logits_regularized = 1e-3 * (policy_logits**2).mean()
                actor_loss_t = actor_loss_t + policy_logits_regularized
                #actor_loss_t = -(probs * q_min).sum(dim=-1).sum(dim=-1).mean()
                log_pi = (policy_probs.clamp_min(1e-5)).log()
                
                entropy_t = -(policy_probs * log_pi).sum(dim=-1).mean()

            self.actor_optimizer.zero_grad()
            self.actor_scaler.scale(actor_loss_t).backward()
            self.actor_scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.agent_cfg.grad_clipping)
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()
            self._actor_loss_logged = float(actor_loss_t.detach().item())
            self._train_entropy_logged = float(entropy_t.detach().item())
            self.actor_tgt.soft_update(self.actor, self.agent_cfg.polyak_update_q_network*10)

        self.critic_tgt.soft_update(self.critic, self.agent_cfg.polyak_update_q_network)
        return (
            {"value_loss": critic_loss.item(), "tgt_diff": [tgt_diff.mean().item()]},
            {"max_q_diff": [max_q_diff]},
            {"actor_loss": self._actor_loss_logged},
            {"train_entropy": [self._train_entropy_logged]},
        )



   





    def _sample_actions_decentralized(self, state: torch.Tensor,  timestep: torch.Tensor, network: torch.nn.Module = None, num_rewards: int = 1, greedy: bool = False, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
       
        inputs = self._get_decetralized_trajectory(state, num_rewards)
        timestep = timestep.unsqueeze(1).expand(-1, self.num_players).reshape(state.shape[0]*self.num_players)

        if self.has_training_started or greedy:
            with torch.amp.autocast(self.device, enabled=self.use_amp):
                B = state.shape[0]
                P = self.num_players
                # if network is self.actor and hasattr(network, "_gumbel_tau"):
                #     #network._gumbel_tau = self._annealed_gumbel_tau(self.current_step)
                gumbel_actions, policy_probs, policy_logits = network.forward(inputs, timestep, hard=hard)
                if greedy:
                    sampled_actions = policy_probs.argmax(dim=-1).view(B, P)
                else:
                    # dist = torch.distributions.Categorical(probs=policy_probs)
                    # sampled_actions = dist.sample().view(B, P)
                    if hard:
                        sampled_actions = gumbel_actions.view(B, P, -1)
                    else:
                        sampled_actions = gumbel_actions.argmax(dim=-1).view(B, P)
                # [B*P, A] -> [B, P, A] for callers (entropy logging, actor loss)
                policy_probs_batched = policy_probs.view(B, P, -1)
            return sampled_actions, policy_probs_batched, policy_logits
            
        else:
            B = state.shape[0]
            P = self.cfg.environment.num_players
            A = self.cfg.environment.num_actions 

            # 1. Uniform probability distribution for each (batch, player)
            probs = torch.full((B, P, A), 1.0 / A, device=self.device)
            probs_full = torch.full((B, P, A+1), 1.0 / (A+1), device=self.device)
            actions = torch.multinomial(
                probs.view(-1, A),  # flatten B*P groups
                num_samples=1,
                replacement=True).view(B, P)   
    
            actions = actions.to(device=self.device, dtype=torch.long)
        return actions, probs, None
            


