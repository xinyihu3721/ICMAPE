import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
from icmape.envs.bandit.gaussian_bandit import GaussianBanditEnv
from icmape.agents.bandit.sequential.base_offpolicy_agent import BaseOffPolicyAgent
from icmape.agents.bandit.sequential.icpe import ICPE
from icmape.agents.bandit.sequential.multicpe_a2c import MULTICPE_A2C
import os

class BaseBanditController:
    """Base class for bandit controllers"""
    
    def __init__(self, env: GaussianBanditEnv, num_envs: int = 1, device: str = "cuda"):
        self.env = env
        self.num_envs = num_envs
        self.num_actions = env.num_actions
        self.horizon = env.horizon
        self.current_step = 0
        self.device = device
        self.action_counts = np.zeros((num_envs, self.num_actions))
        self.action_rewards = np.zeros((num_envs, self.num_actions))
        self.stopped = np.zeros(num_envs, dtype=bool)
        
    def update(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """Update internal state after taking actions and receiving rewards"""
        for i in range(self.num_envs):
            if not self.stopped[i]:
                action = actions[i]
                self.action_counts[i, action] += 1
                self.action_rewards[i, action] += rewards[i]
                if dones[i]:
                    self.stopped[i] = True
        self.current_step += 1

    def get_action_means(self) -> np.ndarray:
        """Get current estimates of action means"""
        return self.action_rewards / (self.action_counts + 1e-8)
        
    def select_actions(self, states) -> np.ndarray:
        """Select next actions to take for all environments"""
        raise NotImplementedError
        
    def reset(self):
        """Reset controller state"""
        self.current_step = 0
        self.action_counts = np.zeros((self.num_envs, self.num_actions))
        self.action_rewards = np.zeros((self.num_envs, self.num_actions))
        self.stopped = np.zeros(self.num_envs, dtype=bool)

class ThompsonSamplingController(BaseBanditController):
    """Thompson Sampling controller for Gaussian bandits"""
    
    def __init__(self, env: GaussianBanditEnv, num_envs: int = 1, num_actions: int = 5, 
                 prior_std: float = 0.7, device: str = "cuda"):
        super().__init__(env, num_envs, device)
        self.prior_means = np.ones((num_envs, num_actions)) * 1.5  # Default prior mean centered at 1.5 for non-magic arms
        self.prior_means[:, 0] = 0.5  # Prior mean for magic arm centered at 0.5
        
        # Prior standard deviations should reflect our uncertainty
        self.prior_stds = np.ones((num_envs, num_actions)) * prior_std
        self.num_actions = num_actions

    def update(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        super().update(actions, rewards, dones)
        # Update posterior parameters
        for i in range(self.num_envs):
            if not self.stopped[i]:
                action = actions[i]
                n = self.action_counts[i, action]
                if n > 0:
                    mean = self.action_rewards[i, action] / n
                    # Update posterior using conjugate prior formulas
                    self.prior_means[i, action] = (self.prior_means[i, action] * self.prior_stds[i, action]**2 + mean * n) / (self.prior_stds[i, action]**2 + n)
                    self.prior_stds[i, action] = 1.0 / np.sqrt(1.0 / self.prior_stds[i, action]**2 + n)
        
    def select_actions(self, states) -> np.ndarray:
        actions = np.zeros(self.num_envs, dtype=int)
        for i in range(self.num_envs):
            if not self.stopped[i]:
                # Sample from posterior distributions
                sampled_values = np.random.normal(self.prior_means[i], self.prior_stds[i])
                actions[i] = np.argmax(sampled_values)
        return actions

class UCBController(BaseBanditController):
    """Upper Confidence Bound controller"""
    
    def __init__(self, env: GaussianBanditEnv, num_envs: int = 1, num_actions: int = 5, 
                 alpha: float = 2.0, prior_std: float = 0.7, device: str = "cuda"):
        super().__init__(env, num_envs, device)
        self.alpha = alpha  # Exploration parameter
        self.num_actions = num_actions
        self.prior_std = prior_std
        
        # Initialize action value estimates with appropriate priors
        self.q_values = np.ones((num_envs, num_actions)) * 1.5  # Default to 1.5 for regular actions
        self.q_values[:, 0] = 0.5  # Prior for magic arm centered at 0.5
        
        print(f"UCB controller initialized with alpha={alpha}, prior_std={prior_std}")
        
    def select_actions(self, states) -> np.ndarray:
        actions = np.zeros(self.num_envs, dtype=int)
        for i in range(self.num_envs):
            if not self.stopped[i]:
                if self.current_step < self.num_actions:
                    # Initial exploration phase - try each arm once
                    actions[i] = self.current_step
                else:
                    # Use the action rewards and counts we've accumulated
                    means = self.action_rewards[i] / (self.action_counts[i] + 1e-8)
                    # Fill in any unvisited arms with prior values
                    for a in range(self.num_actions):
                        if self.action_counts[i, a] == 0:
                            means[a] = self.q_values[i, a]
                    
                    # UCB formula with tuned alpha
                    # Incorporate prior_std into the exploration term for unvisited arms
                    exploration = np.sqrt(self.alpha * np.log(self.current_step + 1) / (self.action_counts[i] + 1e-8))
                    # Add more exploration for unvisited arms based on prior_std
                    for a in range(self.num_actions):
                        if self.action_counts[i, a] == 0:
                            exploration[a] = exploration[a] * (1 + self.prior_std)
                            
                    ucb_values = means + exploration
                    actions[i] = np.argmax(ucb_values)
        return actions
    
    def update(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        super().update(actions, rewards, dones)
        # Also update q_values with observed rewards
        for i in range(self.num_envs):
            if not self.stopped[i]:
                action = actions[i]
                n = self.action_counts[i, action]
                if n > 0:
                    # Update q-values based on observed rewards
                    self.q_values[i, action] = self.action_rewards[i, action] / n

class IDSController(BaseBanditController):
    """Information-Directed Sampling controller for Gaussian bandits"""
    
    def __init__(self, env: GaussianBanditEnv, num_envs: int = 1, num_actions: int = 5, 
                 M: int = 10000, threshold: float = 0.99, prior_std: float = 0.7, 
                 reward_std: float = None, device: str = "cuda"):
        super().__init__(env, num_envs, device)
        self.M = M
        self.num_actions = num_actions
        self.threshold = threshold
        self.flag = np.zeros(num_envs, dtype=bool)
        self.optimal_arms = np.zeros(num_envs, dtype=int)
        
        # Initialize priors to match the problem structure
        self.prior_means = np.ones((num_envs, self.num_actions)) * 1.5  # Regular actions centered at 1.5
        self.prior_means[:, 0] = 0.5  # Magic action centered at 0.5
        
        self.prior_stds = np.ones((num_envs, self.num_actions)) * prior_std
        self.reward_std = reward_std if reward_std is not None else 0.75
        print(f"IDS controller initialized with M={M}, threshold={threshold}, prior_std={prior_std}, reward_std={self.reward_std}")
        
    def update_posterior(self, env_idx: int, action: int, reward: float):
        """Update posterior for a single environment"""
        eta = self.reward_std  # Use single standard deviation for all actions
        mu = self.prior_means[env_idx]
        sigma = self.prior_stds[env_idx]
        
        mu[action] = (eta**2 * mu[action] + reward * sigma[action]**2) / (eta**2 + sigma[action]**2)
        sigma[action] = np.sqrt((eta * sigma[action])**2 / (eta**2 + sigma[action]**2))
        
        return mu, sigma
        
    def computeVIDS(self, env_idx: int):
        """Compute V-IDS for a single environment"""
        if self.flag[env_idx]:
            return self.optimal_arms[env_idx]
            
        thetas = np.array([np.random.normal(self.prior_means[env_idx, a], self.prior_stds[env_idx, a], self.M) 
                          for a in range(self.num_actions)])
        
        Maap = np.zeros((self.num_actions, self.num_actions))
        p_a = np.zeros(self.num_actions)
        
        mu = np.mean(thetas, axis=1)
        theta_hat = np.argmax(thetas, axis=0)
        
        for a in range(self.num_actions):
            for ap in range(self.num_actions):
                t = thetas[ap, np.where(theta_hat == a)]
                Maap[ap, a] = np.nan_to_num(np.mean(t))
                if ap == a:
                    p_a[a] = t.shape[1] / self.M
                    
        if np.max(p_a) >= self.threshold:
            self.flag[env_idx] = True
            self.optimal_arms[env_idx] = np.argmax(p_a)
            return self.optimal_arms[env_idx]
            
        rho_star = sum([p_a[a] * Maap[a, a] for a in range(self.num_actions)])
        delta = rho_star - mu
        v = np.array([sum([p_a[ap] * (Maap[a, ap] - mu[a])**2 
                          for ap in range(self.num_actions)]) 
                     for a in range(self.num_actions)])
        
        return np.argmin(delta**2 / (v + 1e-8))
        
    def update(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        super().update(actions, rewards, dones)
        for i in range(self.num_envs):
            if not self.stopped[i]:
                self.prior_means[i], self.prior_stds[i] = self.update_posterior(
                    i, actions[i], rewards[i]
                )
        
    def select_actions(self, states) -> np.ndarray:
        actions = np.zeros(self.num_envs, dtype=int)
        for i in range(self.num_envs):
            if not self.stopped[i]:
                actions[i] = self.computeVIDS(i)
        return actions
        
    def reset(self):
        super().reset()
        self.flag = np.zeros(self.num_envs, dtype=bool)
        self.optimal_arms = np.zeros(self.num_envs, dtype=int)
        self.prior_means = np.zeros((self.num_envs, self.num_actions))
        self.prior_stds = np.ones((self.num_envs, self.num_actions))

class ETCController(BaseBanditController):
    """Explore-Then-Commit controller using a trained model"""
    
    def __init__(self, env: GaussianBanditEnv, model_horizon: int, model_path: str, device: str = "cuda"):
        num_envs = env.batch_size
        super().__init__(env, num_envs, device)
        self.model_path = model_path
        self.model_horizon = model_horizon
        self.committed_actions = np.zeros(num_envs, dtype=int)
        self.committed = np.zeros(num_envs, dtype=bool)
        self.traj_current_timestep = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.agent = ICPE.load(model_path, device)
        self.step=0
        
    def select_actions(self, states) -> np.ndarray:
        self.step+=1
        if self.step > self.model_horizon:
            return self.committed_actions

        actions = np.zeros(self.num_envs, dtype=int)
        
        seq_states = states.to_transformer_input(self.agent.base_num_actions, self.traj_current_timestep)
        with torch.no_grad():
            qvalues = self.agent.critic.forward(seq_states)
            log_probs = self.agent.i_net.forward(seq_states)

        for i in range(self.num_envs):
            if not self.stopped[i]:
                if self.committed[i]:
                    actions[i] = self.committed_actions[i]
                else:
                    if self.env.enable_stopping_action:
                        if torch.argmax(qvalues[i]).item() == self.env.num_actions - 1:
                            best_action = log_probs[i].argmax(-1).item()
                            self.committed[i] = True
                            self.committed_actions[i] = best_action
                            actions[i] = best_action
                            continue
                    actions[i] = torch.argmax(qvalues[i]).item()
        if self.step == self.model_horizon:
            self.committed_actions = log_probs.argmax(-1).cpu().numpy()
        return actions
        
    def update(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        super().update(actions, rewards, dones)
        for i in range(self.num_envs):
            if not self.stopped[i]:
                self.traj_current_timestep[i] += 1
                
    def reset(self):
        super().reset()
        self.committed_actions = np.zeros(self.num_envs, dtype=int)
        self.committed = np.zeros(self.num_envs, dtype=bool)
        self.traj_current_timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32) 