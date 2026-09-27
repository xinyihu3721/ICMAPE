from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from icmape.envs.bandit.config import BanditEnvType
from icmape.envs.bandit.base import BanditEnv



class GaussianBanditEnv(BanditEnv):
    def __init__(self, horizon: int, means: NDArray[np.float64], stds: NDArray[np.float64], enable_stopping_action: bool, decentralized: bool, num_players: int, caps: NDArray[np.int32], reward_struct: str = 'per_player'):
        """Creates a gaussian bandit model

        Args:
            horizon (int): maximum episode length
            means (NDArray[np.float64]): average rewards of the arms
            num_players (int): number of players in the environment
            caps (NDArray[np.int32]): max capacity of each arm
            reward_struct (str): reward structure
        """
        assert len(stds) == len(means), "Size of the standard deviation vector is not the same as the number of arms"

       
        super().__init__(horizon=horizon, means=means, enable_stopping_action=enable_stopping_action, decentralized=decentralized, num_players=num_players, caps=caps, reward_struct=reward_struct)
        self.std = stds
        assert np.any(self.std < 0) == False, 'Standard deviation needs to be non-negative'
        self.env_type = BanditEnvType.GAUSSIAN_BANDIT
    
    def _step(self, action: int) -> float:
        return self.means[action] + self.std[action] * np.random.normal()
    
    def _step_with_caps(self, action_counts: np.ndarray, actions: np.ndarray) -> float:
        # action: vector of action
        # action_counts: number of players pulling each arm, shape [num_arms]
        # For each player: if their selected arm's count > cap, reward 0; else sample gaussian(means[arm], std[arm])
        caps_exceeded = action_counts[actions] > self.caps[actions]
        random_draws = np.random.normal(self.means[actions], self.std[actions])
        individual_rewards = np.where(caps_exceeded, 0, random_draws)
        reward = np.sum(individual_rewards)
        
        return individual_rewards, reward
    def parameters(self) -> dict[int | np.ndarray]:
        return {'horizon': self.horizon, 'means': self.means, 'stds': self.std, 'enable_stopping_action': self.enable_stopping_action, 'decentralized': self.decentralized, 'num_players': self.num_players, 'caps': self.caps, 'reward_struct': self.reward_struct}
    
