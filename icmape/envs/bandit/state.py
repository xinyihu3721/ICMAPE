from __future__ import annotations
import numpy as np
import torch
from typing import Generic, TypeVar, Tuple, NamedTuple, List

class Experience(NamedTuple):
    step: int
    reward: np.ndarray | torch.Tensor
    action: np.ndarray | torch.Tensor
    done: bool


T = TypeVar("T", np.ndarray, torch.Tensor)

class State(Generic[T]):
    def __init__(self, state: T, num_actions: int, num_rewards: int = 1,decentralized: bool=False):
        """Initialize network state that tracks action vectors and reward vectors

        Args:
            state: Tensor of shape (..., H, num_actions + num_rewards) where:
                  - First num_actions channels contain action vectors (one per player)
                  - Last num_rewards channels contain reward vectors
            num_actions: Number of action channels (equals num_players in multi-player)
            num_rewards: Number of rewards observed per round
        """
        self.state = state
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.decentralized = decentralized

    @staticmethod
    def zeros_numpy(shape: tuple[int], num_actions: int, num_rewards: int = 1,decentralized: bool = False) -> State[np.ndarray]:
        """Create a state with zeros

        Args:
            shape: Shape of the state (typically just horizon H)
            num_actions (int): Number of action channels (equals num_players in multi-player)
            num_rewards (int): Number of rewards observed per round
        """
        if isinstance(shape, int):
            shape = (shape, )
        
        return State(
            np.zeros(tuple(shape) + (num_actions + num_rewards,), dtype=np.float64, order='C'),
            num_actions, num_rewards, decentralized)
    
    @property
    def actions(self) -> T:
        if isinstance(self.state, np.ndarray):
            return torch.from_numpy(self.state[..., :self.num_actions].astype(np.int64)).type(torch.long)
        elif isinstance(self.state, torch.Tensor):
            return self.state[..., :self.num_actions].type(torch.long)#.squeeze(-1)
        raise Exception('Type not found')

    @property
    def rewards(self) -> T:
        if isinstance(self.state, np.ndarray):
            return torch.from_numpy(self.state[..., -self.num_rewards:]).type(torch.float32)
        elif isinstance(self.state, torch.Tensor):
            return self.state[..., -self.num_rewards:]
        raise Exception('Type not found')
    
    def add(self, step: Experience) -> State:
        self.state[..., step.step, :self.num_actions] = step.action
        self.state[..., step.step, -self.num_rewards:] = step.reward
        return self
    
    def to_tensor(self, decentralized: bool, device: torch.device = torch.device('cpu')) -> State[torch.Tensor]:
        return State(torch.from_numpy(self.state).to(device).type(torch.float32), self.num_actions, self.num_rewards, decentralized)

    @staticmethod
    def stack(states: List[State[T]]) -> State[T]:
        if isinstance(states[0].state, np.ndarray):
            return State(np.stack([states[i].state for i in range(len(states))]),
                         num_actions=states[0].num_actions,
                         num_rewards=states[0].num_rewards)
        elif isinstance(states[0].state, torch.Tensor):
            return State(torch.stack([states[i].state for i in range(len(states))]),
                         num_actions=states[0].num_actions,
                         num_rewards=states[0].num_rewards)
        raise Exception('Type not found')

    
    def to_transformer_input(self, num_players: int, timestep: torch.Tensor | int):
        """
        Note: self.num_actions is not the same as the passed parameter num_actions.
            In this function, num_actions refers to the number of num_players in multi-player. 
        """

        assert self.actions.shape[-1] == self.num_actions, f"Expected last dim of self.actions ({self.actions.shape[-1]}) to be self.num_actions ({self.num_actions})"

        batch_size, horizon = self.rewards.shape[0], self.rewards.shape[1]
       

        device = timestep.device 
        actions = self.actions.to(device)
        rewards = self.rewards.to(device)

        timestep_mask = torch.arange(horizon, device=device).unsqueeze(0).expand(batch_size, -1)
        timestep_mask = timestep_mask < timestep.unsqueeze(-1).to(device)

        actions = torch.nn.functional.one_hot(actions, num_players)
        actions = actions.view(*actions.shape[:-2], -1) 
        masked_actions = actions * timestep_mask.unsqueeze(-1)
       
        masked_rewards = rewards * timestep_mask.unsqueeze(-1)
        result = torch.cat([masked_actions, masked_rewards], dim=-1)
        result = result.to(device)
        

        return result