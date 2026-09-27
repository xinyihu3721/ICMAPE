from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import wandb
from abc import abstractmethod
from typing import Tuple, List, Dict
from icmape.dataclasses import Config
from icmape.agents.bandit.config import BaseOffPolicyAgentConfig
from icmape.envs.bandit.base import VecBanditEnv
from icmape.envs.bandit.builder import create_bandit_env, sample_env
from icmape.agents.bandit.history import HistoryData
from icmape.agents.bandit.buffers import ReplayBuffer, ReplayBufferSamples
from icmape.models.sequence_net.builder import create_critic_network_td3, create_inference_network, create_actor_network_td3
from icmape.agents.bandit.history import HistoryBuilder
from torch.optim.lr_scheduler import LinearLR

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


class BaseOffPolicyAgent(object):
    cfg: Config
    agent_cfg: BaseOffPolicyAgentConfig
    device: torch.device
    current_step: int

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.agent_cfg = cfg.agent
        self.device = device
        self.current_step = 0
        self._lr_sched_step_inference = False
        self._lr_sched_step_rl = False
        self.gradient_step = 0
        self.i_gradient_step = 0

        self.num_actions = self.cfg.environment.num_actions + int(self.cfg.environment.enable_stopping_action)
        self.decentralized = self.cfg.environment.decentralized
        self.reward_struct = self.cfg.environment.reward_struct
        self.num_players = self.cfg.environment.num_players
        self.horizon = self.cfg.environment.horizon
        self.total_batch_size = self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories
        self.use_amp = (self.device != "cpu")
        self.q_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.i_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.actor_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.tau_start = self.agent_cfg.actor_network.gumbel_temperature
        self.tau_min = self.agent_cfg.actor_network.gumbel_tau_min
        self.tau_anneal_total_steps = self.cfg.training.max_training_steps// 2
    

        if self.reward_struct == 'per_player':
            self.num_rewards = self.num_players
        else:
            self.num_rewards = 1
        self.buffer = ReplayBuffer(
            buffer_size=self.agent_cfg.buffer_size,
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_traj=self.agent_cfg.num_env_instances * self.agent_cfg.num_trajectories,
            device=device,
            horizon=self.horizon,
            enable_last_transitions_sampling=False
        )

        self.reset_networks()
        self.running_mean_est = 0
    
    def _annealed_gumbel_tau(self, step: int) -> float:
        if self.tau_anneal_total_steps <= 1:
            return self.tau_min
        t = min(int(step), self.tau_anneal_total_steps - 1)
        # Cosine decay: step=0 -> tau_start, step=total_steps-1 -> tau_min
        frac = t / (self.tau_anneal_total_steps - 1)
        tau = self.tau_min + 0.5 * (self.tau_start - self.tau_min) * (1.0 + np.cos(np.pi * frac))
        return max(self.tau_min, float(tau))

    def _set_actor_gumbel_tau(self, step: int) -> None:
        self.actor._gumbel_tau = self._annealed_gumbel_tau(step)
    
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
        return self.num_players * self.num_actions
       
    
    @property
    def cost_per_timestep(self) -> float:
        return self._cost_multiplier.item()


    def reset_networks(self):
        cNet_state_dim = self.num_players + (self.base_num_actions + 1 )*self.num_players 
        architecture_type = getattr(self.agent_cfg, 'architecture_type', 'transformer')
        
        self.critic = create_critic_network_td3(cNet_state_dim, output_dim=self.num_actions, num_actions=self.num_actions, num_players=self.num_players, horizon=self.horizon, config=self.agent_cfg.critic_network, architecture_type=architecture_type).to(self.device)
        self.critic_tgt = create_critic_network_td3(cNet_state_dim, output_dim=self.num_actions, num_actions=self.num_actions, num_players=self.num_players, horizon=self.horizon, config=self.agent_cfg.critic_network, architecture_type=architecture_type).to(self.device)
        self.critic_tgt.hard_update(self.critic)

        
        self.i_net = create_inference_network(self.state_dim, self.base_num_actions, self.num_players, self.horizon, self.agent_cfg.inference_network, architecture_type=architecture_type).to(self.device)
        self.i_tgt_net = create_inference_network(self.state_dim, self.base_num_actions, self.num_players, self.horizon, self.agent_cfg.inference_network, architecture_type=architecture_type).to(self.device)
        self.i_tgt_net.hard_update(self.i_net)

        # Create actor network and target network (similar to critic pattern)
        # State dim: player_id (one-hot, P dims) + actions (A dims) + reward (1 dim)
        actor_state_dim = self.num_players + self.base_num_actions + 1 
        shared_critic = getattr(self.agent_cfg.actor_network, 'shared_critic', False) if hasattr(self.agent_cfg, 'actor_network') else False
        self.actor = create_actor_network_td3(
            actor_state_dim,
            self.base_num_actions,
            self.num_players,
            self.horizon,
            self.agent_cfg.actor_network,
            shared_critic=shared_critic,
            architecture_type=architecture_type,
        ).to(self.device)
        self.actor_tgt = create_actor_network_td3(
            actor_state_dim,
            self.base_num_actions,
            self.num_players,
            self.horizon,
            self.agent_cfg.actor_network,
            shared_critic=shared_critic,
            architecture_type=architecture_type,
        ).to(self.device)
        self.actor_tgt.hard_update(self.actor)
        self._set_actor_gumbel_tau(self.current_step)

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
        self.cost_optimizer = torch.optim.AdamW([self._cost_multiplier], lr=cost_lr)

        if 'final_learning_rate' not in self.agent_cfg:
            final_learning_rate = self.agent_cfg.learning_rate
        else:
            final_learning_rate = self.agent_cfg.final_learning_rate

        self.scheduler_exp = LinearLR(self.critic_optimizer, start_factor=1.0, end_factor= final_learning_rate/self.agent_cfg.critic_learning_rate, total_iters=int(0.5 * self.cfg.training.max_training_steps) )
        self.scheduler_inf = LinearLR(self.i_optimizer, start_factor=1.0, end_factor=final_learning_rate/self.agent_cfg.inference_learning_rate, total_iters=int(0.5 * self.cfg.training.max_training_steps))
        self.scheduler_actor = LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=min(final_learning_rate,self.agent_cfg.actor_learning_rate)/self.agent_cfg.actor_learning_rate, total_iters=int(0.5 * self.cfg.training.max_training_steps))

    def step_lr_schedulers(self) -> None:
        """Step schedulers only for optimizers that ran this iteration (avoids PyTorch warning / wrong schedule)."""
        if self._lr_sched_step_inference:
            self.scheduler_inf.step()
        if self._lr_sched_step_rl:
            self.scheduler_exp.step()
            self.scheduler_actor.step()

    @torch.inference_mode()
    def sample_trajectories(self, B: int, N: int) -> HistoryData:
        """Sample trajectories according to the current policy.
        
        Args:A
            B (int): number of environments to sample
            N (int): number of trajectories per environment
        
        Returns:
            HistoryData: Sampled trajectories
        """
        # Sample environments
        envs = self.sample_envs(B, N)
        horizon = self.cfg.environment.horizon
        
        num_actions = self.base_num_actions
        num_players = self.cfg.environment.num_players
        
        traj_builder = HistoryBuilder(B, N, horizon, num_actions, self.cfg.environment.num_players, self.decentralized, device=self.device, num_rewards=self.num_rewards)
        
        states, dones, stopped = envs.reset()
        states.to_tensor(self.decentralized,self.device)
        
        #true_optimal = torch.tensor(envs.means, device=self.device).argmax(-1)
        true_optimal = self._find_optimal_capacity(envs.caps)
        traj_current_timestep = torch.zeros(B*N, device=self.device, dtype=torch.float32)

        #log_probabilities = torch.zeros(B*N, horizon, self.base_num_actions, device=self.device)
        log_probabilities = torch.zeros(B*N, horizon, self.base_num_actions, self.num_players, device=self.device)


        for h in range(self.horizon):    

            # seq_state = states.to_transformer_input(self.base_num_actions, traj_current_timestep).state
            seq_state = states.to_transformer_input(self.base_num_actions, traj_current_timestep)
            
            #actions, qvalues, probs = self._sample_actions(seq_state, num_rewards=states.num_rewards, greedy=True)
            actions, policy_probs, _ = self._sample_actions_decentralized(seq_state, timestep=traj_current_timestep, network=self.actor, num_rewards=states.num_rewards, greedy=True)
            # Get qvalues separately using the critic network
            qvalues_list = self._get_decentralized_qvalues(seq_state, timestep=traj_current_timestep, network=self.critic, num_rewards=states.num_rewards, decentralize_state=False, actions=actions)
            # Handle both list and tensor returns
            if isinstance(qvalues_list, list):
                qvalues = torch.min(torch.stack(qvalues_list, dim=0), dim=0).values  # Take min over critics
            else:
                qvalues = qvalues_list
            qvalues = qvalues.view(seq_state.shape[0], self.num_players, -1)  # [B, P, num_actions]
            log_probs = self.i_net.forward(seq_state, traj_current_timestep)    
            log_probabilities[:, h] = log_probs
            

            # log_probabilities[:, h] = log_probs
            

            informative_probs: torch.Tensor = torch.nn.functional.softmax(qvalues, dim=-1)  # [B, P, num_actions]
            # Aggregate across players by taking mean to get shape [B, num_actions]
            informative_probs = informative_probs.mean(dim=1)  # [B, num_actions]
            traj_builder.record_exploration_dist(h, informative_probs[...,:self.base_num_actions])
           
            # Execute action and get rewards
            next_states, dones, stopped = envs.step(actions.tolist())
            
            next_states = next_states.to_tensor(self.device)
            dones = torch.from_numpy(dones).to(self.device)
            stopped = torch.from_numpy(stopped).to(self.device)

            

            states = next_states
            traj_current_timestep = traj_current_timestep + (1-dones.float())
            if torch.all(dones):               
                break
        
        means = torch.from_numpy(envs.means).to(dtype=torch.float32, device=self.device)
        caps = torch.from_numpy(envs.caps).to(dtype=torch.float32, device=self.device)
        seq_state = states.to_transformer_input(self.base_num_actions, traj_current_timestep)
        log_probs = self.i_net.forward(seq_state, traj_current_timestep)
        log_probabilities[:, -1] = log_probs
        

        traj = traj_builder.build(states.actions, states.rewards, log_probabilities , means,  caps, traj_current_timestep, true_optimal)

        return traj
    


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

        print(f'----------------------------------')
        print(f'[Eval] Agent: {self.agent_cfg.type} - Env: {self.cfg.environment.bandit_env.type} - Horizon: {self.cfg.environment.horizon}')
        print(f'[Eval] Env means: {traj.means[idenv]}')
        print(df_exp.loc[idenv].round(2))
        return df_est, df_exp
        
    
    def sample_envs(self, B: int, N: int) -> VecBanditEnv:
        """Generate B*N environments. B environments are sampled at random, and then
           each env is repeated N times.

        Args:
            B (int): Number of environments to sample
            N (int): Number of repetitions per environment

        Returns:
            BatchedBanditEnv: A vectorized environment
        """
        # Sample B environment
        envs = [sample_env(self.cfg.environment) for _ in range(B)]

        # Repeat each env N times
        repeated_envs = [create_bandit_env(env.env_type, **env.parameters()) for env in envs for _ in range(N)]
        return VecBanditEnv(repeated_envs)
    
    @torch.no_grad()
    def _log_statistics(self, step: int, training_statistics: Dict[str, List[float]], # mle_rewards: torch.Tensor,
                        log_probs_i_network: torch.Tensor, true_optimal: torch.Tensor,
                        qvalues: torch.Tensor, exploration_probs: torch.Tensor = None, horizons: torch.Tensor = None, end_entropy:torch.Tensor = None):
            """Logs statistics to Wandb

            Args:
                step (int): current training step
                training_statistics (Dict[str, List[float]]): A dictonary containing training statistics
                # mle_rewards (torch.Tensor): The estimated maximum likelihood reward vector
                log_probs_e_network (torch.Tensor): log probabilities of the estimated optimal arm from the e network
                true_optimal (torch.Tensor): vector with the true optimal arms
                qvalues (torch.Tensor): qvalues of the exploration policy
                exploration_probs (Optional[torch.Tensor]): exploration probabilities of the true policy. Defaults to None
                horizon (Optional[torch.Tensor]): array of horizons at which the trajectory stopped
            """
            #=========================
            # estm_rewards_argmax = (log_probs_e_network.argmax(-1) == true_optimal).float()
            # estm_rewards_probs = log_probs_e_network.exp().gather(-1, true_optimal.unsqueeze(-1)).flatten()


            #estm_rewards_argmax = (log_probs_e_network * true_optimal).sum(dim=(-1,-2)).float()
            estm_rewards_argmax =(log_probs_i_network.max(dim=-1).values).sum(dim=-1).float()
            

            # estm_rewards_probs: sum of predicted probabilities where optimal == 1
            #estm_rewards_probs = (log_probs_e_network * true_optimal).sum(dim=(-1,-2)).exp().flatten()
            estm_rewards_probs = (log_probs_i_network.max(dim=-1).values).sum(dim=-1).exp() 

            #=========================
    
            # final_action_correct = mle_rewards
            # mu = np.round(final_action_correct.mean().item(), 3)
            # std = np.round(final_action_correct.std().item(), 3)
            estm_rewards_argmax = estm_rewards_argmax.detach().cpu().numpy()
            estm_rewards_probs = estm_rewards_probs.detach().cpu().numpy()

            print(f"Step {self.current_step + 1} completed.")
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
                _performance_horizons = {'performance/avg_horizon': horizons.mean().item(), 'performance/median_horizon': horizons.median().item()}
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

    def _onehot_action(self, action: torch.Tensor, num_classes:Optional[int] = None) -> torch.Tensor:
        """
        Embed the action into a one-hot encoded tensor.
        """
        if num_classes == None:
            return torch.nn.functional.one_hot(action, num_classes=self.num_actions).view(-1,num_classes*self.num_players)
        else:
            return torch.nn.functional.one_hot(action, num_classes=num_classes)
    
    def _find_optimal_capacity(self, capacity: np.ndarray):
        """
        Convert the capacity vector to a matrix of shape [B, K, M].

        Args:
            capacity (np.ndarray or torch.Tensor): Array of shape [B, K], values in {1, ..., M} representing
                the capacity per arm of each batch B.

        Returns:
            torch.Tensor: Tensor of shape [B, K, M], where for each [b, k, m], entry is 1 if arm k in batch b
                has capacity m+1, else 0.
        """
        # Convert to tensor if not already
        capacity = torch.as_tensor(capacity, device=self.device)
        B, K = capacity.shape
        M = self.num_players
        # capacity values are in {1, ..., M}, so subtract 1 for zero-based indexing
        capacity_index = (capacity - 1).unsqueeze(-1)  # [B, K, 1]
        m_range = torch.arange(M, device=self.device).view(1, 1, M)  # [1, 1, M]
        selected = (capacity_index == m_range).long()  # [B, K, M]
        return selected

        



    def train(self, save_path: str):
        # Collect rollouts
        B, N = self.agent_cfg.num_env_instances, self.agent_cfg.num_trajectories

        max_steps = self.cfg.training.max_training_steps
        min_save_step = max_steps // 3
        step_iter = tqdm(
            range(max_steps),
            desc="train",
            unit="step",
            total=max_steps,
            dynamic_ncols=True,
            disable=False,
            mininterval=300.0,
            miniters=500,
        )
        for step in step_iter:
            #self._set_actor_gumbel_tau(self.current_step)
            # Sample environments
            envs = self.sample_envs(B=B, N=N)

            states, dones, stopped = envs.reset()

            states = states.to_tensor(self.decentralized,self.device)

            true_optimal = self._find_optimal_capacity(envs.caps)
            
            traj_current_timestep = torch.zeros(B*N, device=self.device, dtype=torch.long)

            # Collect transitions
            for t in range(self.horizon):
                with torch.no_grad(), torch.amp.autocast(self.device, enabled=self.use_amp):
                   
                    seq_state = states.to_transformer_input(self.base_num_actions, traj_current_timestep)
                    actions, true_policy_probs,_ = self._sample_actions_decentralized(seq_state,timestep=traj_current_timestep, network=self.actor, num_rewards=states.num_rewards)



                    actions = actions.view(-1, self.num_players)
                    # others_actions = self._get_others_actions(actions)           
                    qvalues = self._get_decentralized_qvalues(seq_state, timestep=traj_current_timestep, network=self.critic, num_rewards=states.num_rewards, decentralize_state=False, actions=actions
                    #others_actions=others_actions
                    )
                    
                  
                  
                 # Execute action and get rewards
            
                next_states, dones, stopped= envs.step(actions.tolist())
                next_states = next_states.to_tensor(self.device)
                dones = torch.from_numpy(dones).to(self.device)
                stopped = torch.from_numpy(stopped).to(self.device)
            
             

                self.buffer.add(states=seq_state,
                                actions=actions,
                                rewards=next_states.rewards[torch.arange(next_states.rewards.size(0)), traj_current_timestep.long().to(next_states.rewards.device)],
                                env_rewards=next_states.rewards[torch.arange(next_states.rewards.size(0)), traj_current_timestep.long().to(next_states.rewards.device)],
                                dones=dones,
                                next_states=next_states.to_transformer_input(self.base_num_actions, (traj_current_timestep + 1)),
                                optimal_actions=true_optimal,
                                stopped=stopped,
                                traj_current_timestep=traj_current_timestep)
                
                states = next_states
                traj_current_timestep = traj_current_timestep + (1-stopped.float())

                if torch.all(dones):
                    break

            
            inference_loss, qloss, max_q_diff, actor_loss, train_entropy = self._train_networks()
            
            with torch.no_grad(), torch.amp.autocast(self.device, enabled=self.use_amp):
                seq_state = states.to_transformer_input(self.base_num_actions, traj_current_timestep).to(self.device)
                log_probs = self.i_net.forward(seq_state,traj_current_timestep)
                op = log_probs.exp().detach().cpu().numpy().round(2)
                op_tgt = self.i_tgt_net.forward(seq_state,traj_current_timestep).exp().detach().cpu().numpy().round(2)
            # Policy probs: [B, P, A] or [B*P, A] from actor; we need [B, P] for per-player logging
            end_entropy = -(true_policy_probs.detach() * true_policy_probs.detach().log()).sum(dim=-1)
            if end_entropy.dim() == 1:
                end_entropy = end_entropy.view(-1, self.num_players)



            

            cost_updates = {}
            if self.is_stopping_enabled:
                cost_updates = self._update_cost(log_probs, true_optimal)

            q_min = torch.stack(qvalues, dim=0).min(dim=0).values
            if step % 1000 == 0:
                qv = q_min.view(qvalues[0].shape[0]//self.num_players, self.num_players,-1).detach().cpu().numpy().round(2)[0]
                print('\nQV: ', qv, flush=True)
                print(f'OP     : {op[0]} ', flush=True)
                print(f'OP  tgt: {op_tgt[0]}', flush=True)
                print(f'Traj current timestep: {traj_current_timestep[0]}', flush=True)
                print(f"States[action,reward]: {np.round(states.state[0].cpu().numpy(), 2)}", flush=True)


            learning_rates = {
                "lr_exploration_net": self.critic_optimizer.param_groups[0]['lr'],
                "lr_inference_net": self.i_optimizer.param_groups[0]['lr'],
                "lr_actor_net": self.actor_optimizer.param_groups[0]['lr'],
            }
            # train_entropy is dict like {'train_entropy': [scalar]}; convert list values to CPU floats for logging
            train_entropy_stats = {
                k: [x.cpu().item() if isinstance(x, torch.Tensor) else float(x) for x in v]
                for k, v in train_entropy.items()
            }

            self._log_statistics(
                step=self.current_step,
                training_statistics={**inference_loss, **qloss, **max_q_diff, **actor_loss, **cost_updates, **learning_rates, **train_entropy_stats},
                # mle_rewards=mle_rewards,
                log_probs_i_network=log_probs,
                true_optimal=true_optimal,
                qvalues=q_min,
                end_entropy=end_entropy,
                exploration_probs=None,
                horizons=traj_current_timestep)
            self.current_step += 1

            self.step_lr_schedulers()
            if self.current_step >= min_save_step and self.current_step % self.cfg.training.save_frequency == 0:
                self.save(save_path)

    def _update_cost(self, log_probs: torch.Tensor, true_optimal: torch.Tensor):

       
        estm_rewards_argmax =(log_probs.max(dim=-1).values).sum(dim=-1).exp()


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
    def _train_rl_nets(self, batch: ReplayBufferSamples) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        raise Exception('_train_rl_nets not implemented')

    def _train_estimation_net(self, batch: ReplayBufferSamples) -> Dict[str, List[float]]:
       
        self.i_net.train()
        
        with torch.amp.autocast(self.device, enabled=self.use_amp):
   
            log_probs = self.i_net.forward(batch.next_states,batch.traj_current_timestep+1) # add 1 to the timestep to account for the next timestep

            est_loss = -(log_probs * batch.optimal_actions).sum(dim=-1).sum(dim=-1).mean()

            

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

    def _train_networks(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        inference_loss, qloss, max_q_diff, actor_loss, train_entropy = {}, {}, {}, {}, {}
        self._lr_sched_step_inference = False
        self._lr_sched_step_rl = False

        if len(self.buffer) >  self.agent_cfg.initial_buffer_bootstrap_num_batches * self.agent_cfg.batch_size * self.agent_cfg.replay_ratio:
            self._lr_sched_step_inference = True
            # Don't add last transitions to avoid leakage! if the capacity of the E network is sufficiently large it will overfit new samples
            for _ in range(self.agent_cfg.replay_ratio):
                inference_loss = self._train_estimation_net(self.buffer.sample(self.agent_cfg.batch_size, add_last_transtions=False))

            # 30 so that the enetwork can bootstrap in the meantime.
            if len(self.buffer) > self.agent_cfg.inference_network_bootstrap_num_batches * self.agent_cfg.batch_size * self.agent_cfg.replay_ratio:
                self._lr_sched_step_rl = True
                for _ in range(self.agent_cfg.replay_ratio):
                    qloss, max_q_diff, actor_loss, train_entropy = self._train_rl_nets(self.buffer.sample(self.agent_cfg.batch_size, add_last_transtions=False) )
                    self.gradient_step += 1

        return inference_loss, qloss, max_q_diff, actor_loss, train_entropy
            
    
    @abstractmethod
    def save(self, path: str):
        raise Exception('Save method not implemented in BaseOffPolicyAgent')

    @staticmethod
    @abstractmethod
    def load(path: str):
        raise Exception('Load method not implemented in BaseOffPolicyAgent')