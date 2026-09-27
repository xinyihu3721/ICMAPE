from __future__ import annotations
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional, Dict, Any, List, Tuple
from icmape.dataclasses import Config

from icmape.agents.robot.config import RobotMULTICPE_SAC_Config
from icmape.agents.robot.buffers import RobotReplayBufferSamples
from icmape.agents.robot.history import HistoryBuilder, HistoryData
from icmape.agents.robot.sequential.base_offpolicy_agent_sac import RobotBaseOffPolicyAgentSAC
from icmape.envs.robot.state import RobotState
from datetime import datetime


class MULTICPE_SAC_ROBOT(RobotBaseOffPolicyAgentSAC):
    """
    Information Constrained Exploration with DQN
    This implementation uses a transformer architecture and handles both informative and optimal policies.
    
    :param cfg: Configuration object containing all parameters
    :param transformer: Transformer model for policy and value prediction
    :param device: Device (cpu, cuda) on which the code should be run
    """
    agent_cfg: RobotMULTICPE_SAC_Config

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.num_players = self.cfg.environment.num_players
        self.num_actions = self.cfg.environment.num_actions +  int(self.cfg.environment.enable_stopping_action)  
        self.decentralized = self.cfg.environment.decentralized
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
                 'step': self.current_step,
                 'cost_multiplier': self._cost_multiplier.data,
                 'cost_optimizer_state_dict': self.cost_optimizer.state_dict()}
        # Save entropy/alpha parameter and optimizer if they exist (only when ent_coef == 'auto')
        if hasattr(self, 'log_alpha'):
            model['log_alpha'] = self.log_alpha.data
        if hasattr(self, 'alpha_optimizer'):
            model['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        filename = f'{path}/{self.cfg.agent.type}_{self.cfg.agent.architecture_type}_{self.current_step}_seed{self.cfg.experiment.seed}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str, device: torch.device) -> "MULTICPE_SAC_ROBOT":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = MULTICPE_SAC_ROBOT(data['cfg'], device)
        agent.critic.load_state_dict(data['critic'])
        agent.i_net.load_state_dict(data['Enet'])
        agent.critic_tgt.load_state_dict(data['critic'])
        agent.i_tgt_net.load_state_dict(data['Enet'])
        agent.actor.load_state_dict(data['actor'])
        if 'step' in data:
            agent.current_step = data['step']
        return agent

    def _onehot_player_ids(self, batch_size: int, horizon: int, num_players: int, device: torch.device) -> torch.Tensor:
        """Return [B, H, P, P] from ``_player_ids_onehot_cache``; ``H`` via ``unsqueeze(1).expand``."""
        B, H, P = int(batch_size), int(horizon), int(num_players)
        c = self._player_ids_onehot_cache
        return c[:B].unsqueeze(1).expand(B, H, P, P)

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

    def _get_decentralized_qvalues(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        network: torch.nn.Module,
        location_history: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        others_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        act = actions if actions is not None else others_actions
        qvalues = network.forward(
            state, location_history=location_history, timestep=timestep, action=act
        )
        return qvalues
        
        #import pdb; pdb.set_trace()
       # qvalues = q_flat.view(inputs.shape[0], self.num_players, -1)

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

 
    def _train_rl_nets(self, batch: RobotReplayBufferSamples) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """Perform SAC update with proper entropy regularization."""
        num_players = self.cfg.environment.num_players
        num_actions = self.cfg.environment.num_actions + 1
        B = batch.states.shape[0]  # batch size

        # Temperature alpha
        if self.ent_coef == 'auto':
            alpha = self.log_alpha.exp().detach()
        else:
            alpha = self.ent_coef

        # ================== BUILD TARGET y_t ==================
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=self.use_amp
        ):
            
            next_states = batch.next_states
            current_states = batch.states

            # Sample next actions from current policy
            # next_actions_full: [B, P]
            # next_log_prob:     [B, P, num_actions] (log-probs over actions)
           
            next_actions_full, next_log_prob = self._sample_actions_decentralized(
                next_states, location_history=batch.location_history, timestep=batch.traj_current_timestep+1, network=self.actor)

            # Flatten to [B*P] for Q lookup
            


            # Q_{\bartheta_i}(s_{t+1}, *) for each critic, decentralized
            # each element in list: [B*P, num_actions]
            # next_others_actions = self._get_others_actions(next_actions_full)
            next_q_all_list = self._get_decentralized_qvalues(
                next_states,
                timestep=batch.traj_current_timestep + 1,
                network=self.critic_tgt,
                location_history=batch.location_history,
                actions=next_actions_full,
            )  # [B, num_player, actions]
 

            # Take min over critics: [n_critic, B,P, num_actions] -> [B,P, num_actions]
            next_q_all = torch.stack(next_q_all_list, dim=0)        # [n_critic, B,P, num_actions]
            next_q_all = torch.min(next_q_all, dim=0).values        # [B,P, num_actions]

            # Q_min(s_{t+1}, a') for sampled a' (per player)
            # next_actions: [B*P]
            
            next_q = next_q_all.gather(-1, next_actions_full.unsqueeze(-1)).squeeze(-1)  # [B,P]
   
            next_log_prob_selected = next_log_prob.gather(-1, next_actions_full.unsqueeze(-1)).squeeze(-1)  
            # next_log_prob_selected = next_log_prob.gather(
            #     -1, next_actions_full
            # ).squeeze(-1)  


            # === old version ===
            soft_q_next = (next_q - alpha * next_log_prob_selected)#[B,P].sum(dim=-1)  # [B] #global soft value
            # soft_q_next = (next_q).sum(dim=-1) 

            # Rewards:
            # cost: per-step penalty for non-terminal transitions
            # rewards_next: terminal posterior reward (only matters when done)
            
            # log_probs_next = torch.log(self.i_tgt_net.predict_proba(next_states.sum(dim=1, keepdim=True)).view(B, -1).max(dim=-1).values)
            log_probs_next = torch.log(self.i_tgt_net.predict_proba(next_states).view(B, -1).max(dim=-1).values)

            rewards_next = log_probs_next.detach()  # [B]
            

            # Cost for non-terminal states
            if self.is_stopping_enabled:
                cost = -(1.0 - batch.dones.float()) * self.cost_per_timestep  # [B]
            else:
                cost = torch.zeros_like(rewards_next)

            d = batch.dones.float()  # [B], 1 if episode terminated (stop action), else 0
           

            # Total immediate reward r_t = cost + d * rewards_next
            # r_t = cost + d * rewards_next  # [B]
            r_scalar = cost + d * rewards_next      # [B]
            r_t = r_scalar.unsqueeze(1).expand(-1, num_players)        # [B, P]

            r_scalar_check =  d * rewards_next      # [B]
            r_t_check = r_scalar_check.unsqueeze(1).expand(-1, num_players)        # [B, P]


            # ===== SAC-style target: y_t = r_t + (1-d) * gamma * (Q_min - alpha log pi) =====
            #tgt_val = r_t + self.agent_cfg.gamma * (1 - d) * soft_q_next        # [B,P]
            tgt_val = r_t + self.agent_cfg.gamma * (1 - d.unsqueeze(1)) * next_q       # [B,P]
            tgt_val_check = r_t_check + self.agent_cfg.gamma * (1 - d.unsqueeze(1)) * next_q      # [B,P]
            tgt_diff = tgt_val - tgt_val_check
   
            



        # ================== CRITIC UPDATE ==================
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # Q_theta(s_t, *) for each critic, decentralized
            # qvalues: [B*P, num_actions]
            # batch_others_actions = self._get_others_actions(batch.actions)
            qvalues_list = self._get_decentralized_qvalues(
                current_states,
                timestep=batch.traj_current_timestep,
                network=self.critic,
                location_history=batch.location_history,
                actions=batch.actions,
            )
            
            # Calculate max difference between two Q network outputs
            max_q_diff = torch.abs(qvalues_list[0] - qvalues_list[1]).max().item() if len(qvalues_list) == 2 else 0.0

        

            # For each critic, gather Q(s_t, a_t) and match to y_t (full-batch, old version)
            # critic_loss = sum(
            #     torch.nn.functional.mse_loss(
            #         qvalues.view(B, num_players, num_actions)
            #         .gather(-1, batch.actions.unsqueeze(-1))      # [B, P, 1]
            #         .squeeze(-1)                                  # [B, P]
            #         .sum(dim=-1),                                 # [B]
            #         tgt_val.detach()
            #     )
            #     for qvalues in qvalues_list
            # )

            critic_loss = sum(
                torch.nn.functional.mse_loss(
                    qvalues
                    .gather(-1, batch.actions.unsqueeze(-1))      # [B, P, 1]
                    .squeeze(-1),                                  # [B, P]
                    tgt_val.detach()
                )
                for qvalues in qvalues_list
            )


            

            # if self.is_stopping_enabled:
            #     idxs = ~(batch.actions == self.num_actions - 1).any(dim=-1)
            #     qnext_nostop = self._get_decentralized_qvalues(next_states[idxs], timestep=batch.traj_current_timestep[idxs]+1, network=self.critic, num_rewards=num_players, decentralize_state=False)
            #     qloss_stopping_current = sum(torch.nn.functional.mse_loss(q_next.view(idxs.sum().item(), num_players, num_actions)[..., -1].mean(dim=-1), rewards_next[idxs]) for q_next in qnext_nostop)
            #     critic_loss = critic_loss + self.critic_lambda * qloss_stopping_current
          



        self.critic_optimizer.zero_grad()
        self.q_scaler.scale(critic_loss).backward()
        self.q_scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.agent_cfg.grad_clipping)
        self.q_scaler.step(self.critic_optimizer)
        self.q_scaler.update()
        # self.critic_tgt.soft_update(self.critic, self.agent_cfg.polyak_update_q_network)

        # (Your actor / alpha updates would follow below, unchanged)





        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            actions_pi, log_prob_full = self._sample_actions_decentralized(current_states, location_history=batch.location_history, timestep=batch.traj_current_timestep, network=self.actor)
    
            log_pi_a = log_prob_full.gather(-1, actions_pi.unsqueeze(-1)).squeeze(-1)  # [B, P]
            assert log_pi_a.requires_grad
            assert log_prob_full.requires_grad
            pi = log_prob_full.exp()
            # actions_pi_others = self._get_others_actions(actions_pi)

            q_values_pi_list = self._get_decentralized_qvalues(
                current_states,
                timestep=batch.traj_current_timestep,
                network=self.critic,
                location_history=batch.location_history,
                actions=actions_pi,
            )
            q_values_all = torch.min(torch.stack(q_values_pi_list, dim=0),dim=0).values
            q_values_all = q_values_all.view(B, self.num_players, self.num_actions)
            #actor_loss = (pi * (alpha * log_prob_full - q_values_all)).sum(dim=-1).mean()
            actor_loss = (pi * (alpha * log_prob_full - q_values_all)).sum(dim=-1).sum(dim=-1).mean()


            

            if self.current_step % 10 == 0:
                print("[Batch[0] Info]")
                print(f"  Timestep:       {batch.traj_current_timestep[0].item()}")
                print(f"  Q-values:       {q_values_all[0].detach().cpu().numpy().round(4)}")
                print(f"  Policy probs:   {log_prob_full[0].detach().exp().cpu().numpy().round(4)}")
                print("="*40 + "\n")


        self.actor_optimizer.zero_grad()
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.agent_cfg.grad_clipping
        )
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()



        # ===================== ALPHA (ENTROPY TEMP) UPDATE =====================
        if self.ent_coef == 'auto':
        
            # with torch.no_grad():
            #     _, log_probs = self._sample_actions_decentralized(current_states, self.actor, num_rewards=num_players)
            #     entropy = -(log_probs.exp() * log_probs).sum(-1).mean(0)
            # print(f"Entropy per player: {entropy} | Target: {self.target_entropy} | Alpha: {self.alpha.item()}")
            log_pi_detached = log_pi_a.detach()  # [B, P]
            alpha_loss = -(self.log_alpha * (log_pi_detached + self.target_entropy)).mean()
            
            log_prob_full_detached = log_prob_full.detach()
            entropy = -(log_prob_full_detached * log_prob_full_detached.exp()).sum(-1).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=np.log(self.min_alpha))
            # Get the updated alpha value after optimization
            alpha_value = self.log_alpha.exp().detach().item()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
            alpha_value = self.ent_coef.item() if isinstance(self.ent_coef, torch.Tensor) else self.ent_coef

        # ===================== TARGET CRITIC UPDATE =====================

        self.critic_tgt.soft_update(self.critic, self.agent_cfg.polyak_update_q_network)
        return (
            {'value_loss': critic_loss.item(), 'tgt_diff': [tgt_diff.mean().item()]},
            {'max_q_diff': [max_q_diff]},
            {'actor_loss': actor_loss.item()},
            {'alpha_loss': alpha_loss.item()},
            {'alpha_value': [alpha_value]},
            {'train_entropy': [entropy]},
        )



    def _sample_actions_decentralized_step(self,state: torch.Tensor,   timesteps: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor], feat_hist: torch.Tensor,network: torch.nn.Module, greedy: bool = False,num_rewards: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    
        B = timesteps.shape[0]
        P = self.num_players
        Bflat = B * P
        device = self.device
        t_player = timesteps.unsqueeze(1).expand(B, P).reshape(Bflat)
        inputs = self._get_decetralized_trajectory(state, num_rewards)   
        #t_player = t_player.clamp(0, self.cfg.environment.horizon - 1)
        row_idx = torch.arange(Bflat, device=device)
        x_t = inputs[row_idx, (t_player-1).clamp(0, self.horizon), :]  # [Bflat, d]
       #import pdb; pdb.set_trace()
 

        if self.has_training_started or greedy:
           
           
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                policy_probs_flat, log_pi_flat, feat_hist_updated, hidden_state_updated = network.step(
                    x_t=x_t,
                    hidden_state=hidden_state,
                    feat_hist=feat_hist,
                    timesteps=t_player,
                )  
                policy_probs = policy_probs_flat.view(B, P, -1)  # [B,P,A+1]
                log_pi = log_pi_flat.view(B, P, -1)              # [B,P,A+1]

                dist = torch.distributions.Categorical(probs=policy_probs)

                if greedy:
                    actions = policy_probs.argmax(dim=-1) 
                else:
                    actions = dist.sample()               
            return actions, log_pi, feat_hist_updated, hidden_state_updated
        else:

            A = self.cfg.environment.num_actions  # base actions (no stop)
            probs = torch.full((B, P, A), 1.0 / A, device=self.device)
            probs_full = torch.full((B, P, A + 1), 1.0 / (A + 1), device=self.device)

            actions = torch.multinomial(
                probs.view(-1, A), 
                num_samples=1,
                replacement=True
            ).view(B, P).to(device=self.device, dtype=torch.long)

            return actions, probs_full.log(), feat_hist, hidden_state





    def _sample_actions_decentralized(self, state: torch.Tensor, location_history: torch.Tensor, timestep: torch.Tensor, network: torch.nn.Module = None, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
       
       
        if self.has_training_started or greedy:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):

                B = state.shape[0]
                P = self.num_players
                policy_probs, log_pi= network.forward(state, timestep, location_history)
                
                policy_probs_reshaped = policy_probs.view(B, P, -1)
                log_pi_reshaped = log_pi.view(B, P, -1) #[B,P,A+1]
                dist = torch.distributions.Categorical(probs=policy_probs_reshaped)
                if greedy:
                    
                    sampled_actions = policy_probs_reshaped.argmax(dim=-1)
                else:
                    sampled_actions = dist.sample()
                #log_pi = log_pi.gather(-1, sampled_actions.unsqueeze(-1)).squeeze(-1)
            return sampled_actions,log_pi_reshaped
            
        else:
            B = state.shape[0]
            P = self.cfg.environment.num_players
            A = self.cfg.environment.num_actions 
            probs = torch.full((B, P, A), 1.0 / A, device=self.device)
            probs_full = torch.full((B, P, A+1), 1.0 / (A+1), device=self.device)
            actions = torch.multinomial(
                probs.view(-1, A),  # flatten B*P groups
                num_samples=1,
                replacement=True).view(B, P)   
            # actions = torch.multinomial(
            #     probs_full.view(-1, A+1),  # flatten B*P groups
            #     num_samples=1,
            #     replacement=True).view(B, P)   
            actions = actions.to(device=self.device, dtype=torch.long)
        return actions, probs.log()
            


    @torch.inference_mode()
    def _sample_actions(self, state: torch.Tensor, num_rewards: int = 1, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            qvalues = self.critic.forward(state)
        if self.has_training_started or greedy:
            # qvalues shape: (batch_size, num_actions * num_players)
            # For each group of num_actions, select the argmax index (i.e., best action for each player)
            actions = qvalues.view(-1, self.num_players, self.num_actions).argmax(dim=-1)

        else:
            # ================================== 
            # actions = np.random.choice(self.cfg.environment.num_actions, size=state.shape[0])
            # actions = torch.from_numpy(actions).to(self.device)
            probs = torch.full((state.shape[0], self.cfg.environment.num_actions), 1.0 / self.cfg.environment.num_actions, device=self.device)
            actions = torch.multinomial(probs, num_samples=self.cfg.environment.num_players, replacement=True)
    
        actions = actions.to(device=self.device, dtype=torch.long)
        return actions, qvalues, None