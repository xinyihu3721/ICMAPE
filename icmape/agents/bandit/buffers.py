import numpy as np
import torch
from typing import Optional, Generator
from dataclasses import dataclass

@dataclass
class RolloutBufferSamples:
    states: torch.Tensor               # Shape: [batch_size, horizon, 2 * action_size]
    current_timestep: torch.Tensor     # Shape: [batch_size]
    actions: torch.Tensor              # Shape: [batch_size]
    optimal_actions: torch.Tensor      # Shape: [batch_size]
    rewards_optimal: torch.Tensor      # Shape: [batch_size] - rewards from optimal actions
    values: torch.Tensor               # Shape: [batch_size]
    dones: torch.Tensor               # Shape: [batch_size]
    log_probs_informative: torch.Tensor  # Shape: [batch_size]
    advantages: torch.Tensor          # Shape: [batch_size]
    returns: torch.Tensor             # Shape: [batch_size]
    means: torch.Tensor               # Shape: [batch_size, num_actions]
    next_states: torch.Tensor               # Shape: [batch_size, horizon, 2 * action_size]

@dataclass
class ReplayBufferSamples:
    states: torch.Tensor      # Shape: [batch_size, horizon]
    actions: torch.Tensor      # Shape: [batch_size, horizon]
    rewards: torch.Tensor # Shape: [batch_size, horizon]
    env_rewards: torch.Tensor   # Shape: [batch_size]
    next_states: torch.Tensor # Shape: [batch_size, horizon]
    dones: torch.Tensor     # Shape: [batch_size]
    optimal_actions: torch.Tensor # Shape: [batch_size]
    traj_current_timestep: torch.Tensor # Shape: [batch_size]

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        num_actions: int,
        num_players: int,
        num_traj: int,
        device: torch.device,
        horizon: Optional[int] = None,
        enable_last_transitions_sampling: Optional[bool] = False
    ):
        self.buffer_size = buffer_size + (buffer_size % num_traj)
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_players = num_players
        self.num_traj = num_traj
        self.device = device
        self.horizon = horizon
        self.enable_last_transitions_sampling = enable_last_transitions_sampling

        self.reset()
    
    def reset(self):
        # History buffers
        if self.horizon is None:
            raise ValueError("ReplayBuffer requires a finite horizon for sequence storage.")
        else:
            self.states = np.zeros((self.buffer_size, self.horizon, self.state_dim), dtype=np.float32)
            self.next_states = np.zeros((self.buffer_size, self.horizon, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_players), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.num_players), dtype=np.float32)
        self.env_rewards = np.zeros((self.buffer_size, self.num_players), dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        self.optimal_actions  = np.zeros((self.buffer_size, self.num_actions-1, self.num_players), dtype=np.int64)
        self.traj_current_timestep = np.zeros(self.buffer_size, dtype=np.float32)
        self.new_transitions_idxs = []

        # Position and full flags
        self.pos = 0
        self.full = False

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        env_rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        optimal_actions: torch.Tensor,
        stopped: torch.Tensor,
        traj_current_timestep: Optional[torch.Tensor] = None
    ) -> None:
        """Add a new transition to the buffer"""

        not_stopped = ~stopped
        n_transitions = int(not_stopped.sum())

        space_left = min(n_transitions, self.buffer_size - self.pos)
        remaining = n_transitions - space_left
        
        # Move not_stopped to the same device as the tensors being indexed
        not_stopped_cpu = not_stopped.cpu()
        
        self.states[self.pos:self.pos+space_left] = states[not_stopped_cpu][:space_left].detach().cpu().numpy()
        self.actions[self.pos:self.pos+space_left, :] = actions[not_stopped_cpu][:space_left].detach().cpu().numpy()
        self.rewards[self.pos:self.pos+space_left, :] = rewards[not_stopped_cpu][:space_left].detach().cpu().numpy()
        self.env_rewards[self.pos:self.pos+space_left, :] = env_rewards[not_stopped_cpu][:space_left].detach().cpu().numpy()
        self.dones[self.pos:self.pos+space_left] = dones[not_stopped_cpu][:space_left].detach().cpu().numpy()
        self.next_states[self.pos:self.pos+space_left] = next_states[not_stopped_cpu][:space_left].detach().cpu().numpy()
        self.optimal_actions[self.pos:self.pos+space_left] = optimal_actions[not_stopped_cpu][:space_left].detach().cpu().numpy()
        
        if traj_current_timestep is not None:
            self.traj_current_timestep[self.pos:self.pos+space_left] = traj_current_timestep[not_stopped_cpu][:space_left].detach().cpu().numpy()
        if self.enable_last_transitions_sampling:
            self.new_transitions_idxs.extend([x for x in range(self.pos, self.pos+space_left)])
        self.pos += space_left

        if remaining > 0:
            self.full = True
            self.states[:remaining] = states[not_stopped_cpu][space_left:].detach().cpu().numpy()
            self.actions[:remaining] = actions[not_stopped_cpu][space_left:].detach().cpu().numpy()
            self.rewards[:remaining] = rewards[not_stopped_cpu][space_left:].detach().cpu().numpy()
            self.env_rewards[:remaining] = env_rewards[not_stopped_cpu][space_left:].detach().cpu().numpy()
            self.dones[:remaining] = dones[not_stopped_cpu][space_left:].detach().cpu().numpy()
            self.next_states[:remaining] = next_states[not_stopped_cpu][space_left:].detach().cpu().numpy()
            self.optimal_actions[:remaining] = optimal_actions[not_stopped_cpu][space_left:].detach().cpu().numpy()
            if traj_current_timestep is not None:
                self.traj_current_timestep[:remaining] = traj_current_timestep[not_stopped_cpu][space_left:].detach().cpu().numpy()
            if self.enable_last_transitions_sampling:
                self.new_transitions_idxs.extend([x for x in range(remaining)])
            self.pos = remaining
       
    

    def get_last_inds(self, size: int):
        L = len(self.new_transitions_idxs)
        if size < L:
            idxs = np.array(self.new_transitions_idxs[:size])
            self.new_transitions_idxs = self.new_transitions_idxs[size:]
        else:
            idxs = np.array(self.new_transitions_idxs)
            self.new_transitions_idxs = []
        return idxs


    def sample(self, batch_size: int, add_last_transtions: bool = True) -> ReplayBufferSamples:
        """Sample a batch of transitions"""
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
    
        if add_last_transtions and self.enable_last_transitions_sampling:  # only if we have transitions
            last_inds = self.get_last_inds(batch_size)
            n_last = min(batch_size,len(last_inds))
            batch_inds[:n_last] = last_inds  # force inclusion of the last transitions
        np.random.shuffle(batch_inds)

        return ReplayBufferSamples(
            states=torch.tensor(self.states[batch_inds], device=self.device, dtype=torch.float32),
            actions=torch.tensor(self.actions[batch_inds], device=self.device, dtype=torch.long),
            rewards=torch.tensor(self.rewards[batch_inds], device=self.device, dtype=torch.float32),
            env_rewards=torch.tensor(self.env_rewards[batch_inds], device=self.device, dtype=torch.float32),
            next_states=torch.tensor(self.next_states[batch_inds], device=self.device, dtype=torch.float32),
            dones=torch.tensor(self.dones[batch_inds], device=self.device, dtype=torch.float32),
            optimal_actions=torch.tensor(self.optimal_actions[batch_inds], device=self.device, dtype=torch.long),
            traj_current_timestep=torch.tensor(self.traj_current_timestep[batch_inds], device=self.device, dtype=torch.float32)
        )
    

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_actions: int,
        horizon: int,
        device: torch.device,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        assert buffer_size <= horizon , 'Buffer size should be <= than the horizon'
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        self.horizon = horizon
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.reset()

    def reset(self):
        self.states = np.zeros((self.buffer_size, self.n_envs, 2 * self.num_actions), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.n_envs, 2 * self.num_actions), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.current_timestep = np.zeros(self.buffer_size, dtype=np.int64)
        
        
        self.optimal_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.rewards_optimal = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.log_probs_informative = np.zeros((self.buffer_size, self.n_envs, self.num_actions), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.means = np.zeros((self.buffer_size, self.n_envs, self.num_actions), dtype=np.float32)
        
        self.pos = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,    # Shape: [horizon]
        action: int,
        current_timestep: int,
        optimal_action: int,
        done: bool,
        reward_optimal: float,
        value: torch.Tensor,
        next_value: torch.Tensor,
        log_prob_informative: torch.Tensor,
        means: np.ndarray,
        next_state
    ) -> None:
        """Add a new transition to the buffer"""
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.current_timestep[self.pos] = current_timestep
        self.dones[self.pos] = done
        self.actions[self.pos] = action
        self.optimal_actions[self.pos] = optimal_action
        self.rewards_optimal[self.pos] = reward_optimal
        self.values[self.pos] = value
        self.next_values[self.pos] = next_value
        self.log_probs_informative[self.pos] = log_prob_informative
        self.means[self.pos] = means

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantage(self) -> None:
        """Compute returns and advantages for each transition independently"""
        last_gae_lam = np.zeros(self.n_envs)
        returns = np.zeros_like(self.returns)
        advantages = np.zeros_like(self.advantages)

        for step in reversed(range(self.size())):
            delta = self.rewards_optimal[step] + self.gamma * (1 - self.dones[step]) * self.next_values[step] - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * last_gae_lam
            self.advantages[step] = last_gae_lam
            self.returns[step] = self.advantages[step] + self.values[step]

        return self.advantages, self.returns


    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        buffer_size = self.size()
        indices = np.random.permutation(buffer_size * self.n_envs)
        
        batch_size = buffer_size if batch_size is None else batch_size
        start_idx = 0
        while start_idx < buffer_size:
            buffer_idx = indices[start_idx:start_idx + batch_size]
            batch_indices = buffer_idx // self.n_envs
            env_idx = buffer_idx % self.n_envs
            yield RolloutBufferSamples(
                states=torch.tensor(self.states[batch_indices, env_idx], device=self.device),
                current_timestep=torch.tensor(self.current_timestep[batch_indices], device=self.device),        
                actions=torch.tensor(self.actions[batch_indices, env_idx], device=self.device),
                optimal_actions=torch.tensor(self.optimal_actions[batch_indices, env_idx], device=self.device),
                rewards_optimal=torch.tensor(self.rewards_optimal[batch_indices, env_idx], device=self.device),
                values=torch.tensor(self.values[batch_indices, env_idx], device=self.device),
                dones=torch.tensor(self.dones[batch_indices, env_idx], device=self.device),
                log_probs_informative=torch.tensor(self.log_probs_informative[batch_indices, env_idx], device=self.device),
                advantages=torch.tensor(self.advantages[batch_indices, env_idx], device=self.device),
                returns=torch.tensor(self.returns[batch_indices, env_idx], device=self.device),
                means=torch.tensor(self.means[batch_indices, env_idx], device=self.device),
                next_states=torch.tensor(self.next_states[batch_indices, env_idx], device=self.device),
            )
            start_idx += batch_size

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos