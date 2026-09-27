from __future__ import annotations
import torch
import pickle
from typing import Optional, Dict, Any, List, Tuple
from icmape.dataclasses import Config

from icmape.agents.bandit.config import ICPEConfig
from icmape.agents.bandit.buffers import ReplayBufferSamples
from icmape.agents.bandit.sequential.base_offpolicy_agent import BaseOffPolicyAgent, HistoryData

class MULTICPE_QLEARNING(BaseOffPolicyAgent):
    """
    Information Constrained Exploration with DQN
    This implementation uses a transformer architecture and handles both informative and optimal policies.
    
    :param cfg: Configuration object containing all parameters
    :param transformer: Transformer model for policy and value prediction
    :param device: Device (cpu, cuda) on which the code should be run
    """
    agent_cfg: ICPEConfig

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.num_players = self.cfg.environment.num_players
        self.num_actions = self.cfg.environment.num_actions +  int(self.cfg.environment.enable_stopping_action)  
        self.decentralized = self.cfg.environment.decentralized
        self._player_ids_onehot_cache = self.get_player_ids_onehot_cache()
        P = self.num_players
        self._player_reward_gather_idx = torch.arange(P, device=self.device, dtype=torch.long).view(1, 1, P, 1)

    @property
    def critic_lambda(self) -> float:
        """Linearly from ``1e-4`` to ``1`` over the first half of ``max_training_steps``; then ``1``."""
        max_steps = self.cfg.training.max_training_steps
        half = max(int(max_steps) // 2, 1)
        z = min(1.0, float(self.current_step) / float(half))
        return 1e-3 + (1.0 - 1e-3) * z

    def get_player_ids_onehot_cache(self) -> torch.Tensor:
        """[B_max, P, P] on ``self.device``; B_max = max(replay batch, rollout batch)."""
        P = self.num_players
        device = self.device
        B_cap = max(int(self.agent_cfg.batch_size), int(self.total_batch_size))
        row = torch.eye(P, dtype=torch.float32, device=device)
        return row.unsqueeze(0).expand(B_cap, P, P)

    def save(self, path: str):
        model: Dict[str, Any] = {
            'cfg': self.cfg,
            'Inet': self.i_net.state_dict(),
            'step': self.current_step,
        }
        if not self.uniform_action_only:
            model['critic'] = self.critic.state_dict()
        filename = f'{path}/icpe_{self.current_step}_{self.cfg.experiment.seed}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str, device: torch.device) -> ICPE:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = ICPE(data['cfg'], device)
        critic_sd = data.get('critic')
        if critic_sd is not None:
            agent.critic.load_state_dict(critic_sd)
            agent.critic_tgt.load_state_dict(critic_sd)
        # Support both 'Inet' and 'Enet' for backward compatibility
        inference_state = data.get('Inet', data.get('Enet'))
        agent.i_net.load_state_dict(inference_state)
        agent.i_tgt_net.load_state_dict(inference_state)
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
        # B exceeds cache size -- build directly (no data copy, just a view expand)
        row = torch.eye(P, dtype=torch.float32, device=device)
        return row.view(1, 1, P, P).expand(B, H, P, P)

    def _get_decetralized_trajectory(self, state: torch.Tensor, num_rewards: int) -> torch.Tensor:
        """
        Returns decentralized trajectory representation for each player (same pattern as MULTICPE_SAC_CentralQ).
        """
        device = self.device
        B, H, _ = state.shape
        P = self.num_players
        A = self.base_num_actions
        state = state.to(device)

        with torch.amp.autocast(device, enabled=self.use_amp):
            actions_part = state[..., :P * A]
            rewards_part = state[..., -num_rewards:]

            actions = actions_part.view(B, H, P, A).contiguous()
            player_ids_onehot = self._onehot_player_ids(B, H, P, device)

            rewards_part = rewards_part.unsqueeze(2).expand(B, H, P, num_rewards)
            rewards_p = rewards_part.gather(-1, index=self._player_reward_gather_idx.expand(B, H, P, 1))

            state_p = torch.cat([player_ids_onehot, actions, rewards_p], dim=-1)
            ind_traj = state_p.permute(0, 2, 1, 3).contiguous().view(B * P, H, P + A + 1)
        return ind_traj

    def _get_decentralized_qvalues(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        network: torch.nn.Module,
        num_rewards: int,
        decentralize_state: bool = True,
        decentralized_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
      
        if decentralize_state:
            if decentralized_inputs is None:
                inputs = self._get_decetralized_trajectory(state, num_rewards=num_rewards)
            else:
                inputs = decentralized_inputs
            timestep = timestep.unsqueeze(1).expand(-1, self.num_players).reshape(state.shape[0] * self.num_players)
        else:
            player_ids_onehot = self._onehot_player_ids(state.shape[0], state.shape[1], self.num_players, state.device)
            state_repeat = state.unsqueeze(-1).expand(-1, -1, -1, self.num_players)
            state_p = torch.cat([player_ids_onehot, state_repeat], dim=-2).permute(0, 3, 1, 2)
            state_dim = state_p.shape[2:]
            inputs = state_p.reshape(state_p.shape[0] * state_p.shape[1], *state_dim)
            timestep = timestep.unsqueeze(1).expand(-1, self.num_players).reshape(state.shape[0] * self.num_players)

        with torch.amp.autocast(self.device, enabled=self.use_amp):
            q_flat = network.forward(inputs, timestep)

        B = state.shape[0]
        P = self.num_players
        qvalues = q_flat.view(B, P, -1)
        return qvalues


    def _train_rl_nets(self, batch: ReplayBufferSamples) -> Dict[str, List[float]]:
        """Perform DQN  update."""
        with torch.no_grad(), torch.amp.autocast(self.device, enabled=self.use_amp):
            num_players = self.num_players

            ind_next = self._get_decetralized_trajectory(batch.next_states, num_rewards=num_players)
            ts_next = batch.traj_current_timestep + 1

            next_act = self._get_decentralized_qvalues(
                batch.next_states, ts_next, self.critic,
                num_rewards=num_players, decentralize_state=True, decentralized_inputs=ind_next,
            ).argmax(-1, keepdim=True)

            if self.use_q_mixer:
                next_q = self._get_decentralized_qvalues(
                    batch.next_states, ts_next, self.critic_tgt,
                    num_rewards=num_players, decentralize_state=True, decentralized_inputs=ind_next,
                ).gather(-1, next_act).squeeze(-1)
                next_q = self.q_mixer_tgt.forward(next_q, batch.next_states, ind_states=ind_next)
            elif self.VDN_mode:
                next_q = self._get_decentralized_qvalues(
                    batch.next_states, ts_next, self.critic_tgt,
                    num_rewards=num_players, decentralize_state=True, decentralized_inputs=ind_next,
                ).gather(-1, next_act).squeeze(-1).sum(dim=-1)

            else:
                next_q = self._get_decentralized_qvalues(
                    batch.next_states, ts_next, self.critic_tgt,
                    num_rewards=num_players, decentralize_state=True, decentralized_inputs=ind_next,
                ).gather(-1, next_act).squeeze(-1)
            

            log_probs_next = (self.i_tgt_net.forward(batch.next_states, batch.traj_current_timestep + 1).max(dim=-1).values).sum(dim=-1)
            rewards_next = log_probs_next.detach()
                

            cost = 0
            if self.is_stopping_enabled:
                cost = -(1-batch.dones.float()) * self.cost_per_timestep

            next_q = next_q.squeeze(-1).squeeze(-1)
 
            
            d = batch.dones.float()

            if self.VDN_mode:
                r_t = cost + d * rewards_next
            else:
                r_scalar = cost + d * rewards_next      # [B]
                r_t = r_scalar.unsqueeze(1).expand(-1, num_players)        # [B, P]
                d = d.unsqueeze(1)
            
            tgt_val = r_t + self.agent_cfg.gamma * (1 - d) * next_q

            

        with torch.amp.autocast(self.device, enabled=self.use_amp):
            ind_cur = self._get_decetralized_trajectory(batch.states, num_rewards=num_players)
            qvalues = self._get_decentralized_qvalues(
                batch.states, batch.traj_current_timestep, self.critic,
                num_rewards=num_players, decentralize_state=True, decentralized_inputs=ind_cur,
            )
            if self.use_q_mixer:
                qvalues_act = qvalues.view(-1, self.num_players, self.num_actions).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
                qvalues_act = self.q_mixer.forward(qvalues_act, batch.states, ind_states=ind_cur)
            elif self.VDN_mode:
                qvalues_act = qvalues.view(-1, self.num_players, self.num_actions).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            else:
                qvalues_act = qvalues.view(-1, self.num_players, self.num_actions).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)

            
            qloss = torch.nn.functional.mse_loss(qvalues_act, tgt_val.detach())

            # if self.VDN_mode:
            #     # Prevent double accounting
            #     idxs = ~(batch.actions == self.num_actions - 1).any(dim=-1)
            #     qnext = self._get_decentralized_qvalues(batch.next_states[idxs], batch.traj_current_timestep[idxs]+1, self.critic, num_rewards=num_players, decentralize_state=True)[..., -1].mean(dim=-1)
                
            #     qloss_stopping_current = torch.nn.functional.mse_loss(qnext, rewards_next[idxs])
            #     qloss = qloss +  qloss_stopping_current

        self.critic_optimizer.zero_grad()
        self.q_scaler.scale(qloss).backward()
        self.q_scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                    self.agent_cfg.grad_clipping)
        self.q_scaler.step(self.critic_optimizer)
        self.q_scaler.update()
                

        self.critic_tgt.soft_update(self.critic, self.agent_cfg.polyak_update_q_network)
        # if self.use_q_mixer:
        #     self.q_mixer_tgt.soft_update(self.q_mixer, self.agent_cfg.polyak_update_q_network*0.01)

        return {'value_loss': qloss.item()}


    @torch.inference_mode()
    def _sample_actions_decentralized(self, state: torch.Tensor, timestep: Optional[torch.Tensor] = None, num_rewards: int = 1, greedy: bool = False, stopping_prob: float = 0.05, exploration_epsilon: float = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if timestep is None:
            timestep = torch.zeros(state.shape[0], device=self.device, dtype=torch.float32)
        if (self.has_training_started or greedy) and not self.uniform_action_only:
            qvalues = self._get_decentralized_qvalues(state, timestep, self.critic, num_rewards=num_rewards, decentralize_state=True)
            if exploration_epsilon is not None: #and exploration_prob>self.exploration_epsilon_end:
                idxs = torch.rand(state.shape[0], device=self.device) < exploration_epsilon
                actions = qvalues.argmax(dim=-1)
                n_explore = idxs.sum().item()
                actions[idxs] = torch.randint(0, self.num_actions, (n_explore, self.num_players), device=self.device)
                qvalues = qvalues.view(state.shape[0], -1)
            else: 
                actions = qvalues.argmax(dim=-1)
                qvalues = qvalues.view(state.shape[0], -1)

        else:
            # # Old method (uniform all actions):
            probs = torch.full((state.shape[0], self.cfg.environment.num_actions), 1.0 / self.cfg.environment.num_actions, device=self.device)

            probs = probs / probs.sum(dim=-1, keepdim=True)
            actions = torch.multinomial(probs, num_samples=self.cfg.environment.num_players, replacement=True)
            qvalues = self._get_decentralized_qvalues(state, timestep, self.critic, num_rewards=num_rewards, decentralize_state=True)
            qvalues = qvalues.view(state.shape[0], -1)
            actions = actions.to(device=self.device, dtype=torch.long)
        return actions, qvalues, None
            

