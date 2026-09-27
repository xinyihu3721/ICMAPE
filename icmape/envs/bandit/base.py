from __future__ import annotations
import gymnasium
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Tuple, List
from icmape.envs.bandit.state import State, Experience



class BanditEnv(gymnasium.Env, ABC):
    optimal_action: int
    means: NDArray[np.float64]
    horizon: int
    current_state: State
    env_type: str = None
    enable_stopping_action: bool
    decentralized: bool
    num_actions: int
    num_players: int
    caps: NDArray[np.int32]
    reward_struct: str
    def __init__(self, horizon: int, means: NDArray[np.float64], enable_stopping_action: bool, decentralized: bool, num_players: int, caps: NDArray[np.int32], reward_struct: str = 'per_player'):
        """Create an abstract bandit environment

        Args:
            means (NDArray[np.float64]): vector of average rewards
            horizon (int): maximum episode length
            num_players (int): number of players in the environment
        """
        super().__init__()
        self.means = means
        self.caps = caps
        self.horizon = horizon
        self.enable_stopping_action = enable_stopping_action
        self.decentralized = decentralized
        self.num_players = num_players
        self.reward_struct = reward_struct
        if self.reward_struct == 'per_player':
            self.num_rewards = self.num_players
        elif self.reward_struct == 'sum':
            self.num_rewards = 1
        else:
            raise ValueError(f"Invalid reward structure: {self.reward_struct}")
        self.dim = len(means)
        self.num_actions = self.dim + int(enable_stopping_action)
        self.observation_space = gymnasium.spaces.Box(low=1, high=1, shape=(1,))
        self.action_space = gymnasium.spaces.Discrete(n = self.dim, start = 0)
        self.current_step = 0
        self.optimal_action = np.argmax(means).astype(np.longlong)
        self.current_state = State.zeros_numpy(self.horizon, num_actions=self.num_players, num_rewards=self.num_rewards, decentralized=self.decentralized)
        self.stopped = False

    def reset(self) -> Tuple[State, bool, bool]:
        self.current_step = 0
        
        self.current_state = State.zeros_numpy(self.horizon, num_actions=self.num_players, num_rewards=self.num_rewards, decentralized=self.decentralized)
        return self.current_state, False, False

    def step(self, actions: list[int]) -> Tuple[State, bool, bool]:
        actions = np.array(actions, dtype=int).flatten()

        assert np.all((0 <= actions) & (actions < self.num_actions)), \
            f"Actions {actions} need to be in 0, 1, ..., {self.num_actions-1}"
        
        if self.stopped:
            return self.current_state, True, True
        
        # self.stopped = self.enable_stopping_action and np.any(actions == self.num_actions - 1)
        self.stopped = self.enable_stopping_action and np.any(actions == self.num_actions - 1)


        done = (self.current_step + 1 >= self.horizon) or self.stopped
        if not self.stopped:
            action_counts = np.zeros(len(self.means))
            np.add.at(action_counts, actions, 1)

            individual_rewards, reward = self._step_with_caps(action_counts,actions) 
            if self.decentralized:
                exp = Experience(self.current_step, individual_rewards, actions, done)
            else:
                exp = Experience(self.current_step, reward, actions, done)
            
            self.current_state.add(exp)
            self.current_step += 1
        return self.current_state, done, False #self.stopped #False

    @abstractmethod
    def _step(self, action: int) -> float:
        """Computes the reward given an action

        Args:
            action (int): action taken
        Returns:
            float: reward
        """
        raise Exception('Method _step not implemented!')

    @abstractmethod
    def _step_with_caps(self, action: np.ndarray) -> float:
        """Computes the reward given an action

        Args:
            action (np.ndarray): 'action' is a vector where each entry specifies the number of players pulling each arm.
        Returns:
            float: reward
        """
        raise Exception('Method _step not implemented!')
    @abstractmethod
    def parameters(self) -> dict[int | float| np.ndarray]:
        raise Exception('Not implemeneted')





class VecBanditEnv(object):
    num_actions: int
    enable_stopping_action: bool
    decentralized: bool
    horizon: int

    def __init__(self, envs: List[BanditEnv]):
        """
        Create a batched bandit environment.

        Args:
            envs (List[BanditEnv]): List of individual bandit environments.
        """
        
        super().__init__()
        self.envs = envs
        self.batch_size = len(envs)
        self.optimal_actions = np.stack([env.optimal_action for env in envs]).astype(np.longlong)
        self.means = np.stack([env.means for env in envs]).astype(np.float64)
        self.caps = np.stack([env.caps for env in envs]).astype(np.int32)

        ## We Assume the environments are homogeneous!!
        self.num_actions = envs[0].num_actions
        self.enable_stopping_action = envs[0].enable_stopping_action
        self.decentralized = envs[0].decentralized
        self.horizon = envs[0].horizon
        
    def reset(self) -> Tuple[State, bool, bool]:
        """
        Reset all sub-environments and return the batched initial observations.
        """
        obs = State.stack([env.reset()[0] for env in self.envs])
        return obs, np.zeros(self.batch_size, dtype=np.bool_),  np.zeros(self.batch_size, dtype=np.bool_)
    
    def step(self, actions: List[int]) -> Tuple[State, np.ndarray, np.ndarray]:
        """
        Take a list of actions, one for each sub-environment, and perform a step in each.

        Args:
            actions (List[int]): List of actions, one for each sub-environment.

        Returns:
            Tuple containing batched states and termination flags
        """
        new_states, dones, stopped = zip(*map(lambda env, act: env.step(act), self.envs, actions))

        new_states = State.stack(new_states)
        return new_states, np.array(dones), np.array(stopped)
    def parameters(self) -> List[dict[int | float| np.ndarray]]:
        return [env.parameters() for env in self.envs]