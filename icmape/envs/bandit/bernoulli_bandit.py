from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from icmape.envs.bandit.config import BanditEnvType
from icmape.envs.bandit.base import BanditEnv

class BernoulliBanditEnv(BanditEnv):
    def __init__(self, horizon: int, means: NDArray[np.float64], enable_stopping_action: bool, decentralized: bool, num_players: int,caps: NDArray[np.int32], reward_struct: str = 'per_player'):
        super().__init__(horizon=horizon, means=means, enable_stopping_action=enable_stopping_action, decentralized=decentralized, num_players=num_players, caps=caps, reward_struct=reward_struct)
        self.means = means
        self.env_type = BanditEnvType.BERNOULLI_BANDIT
        self.horizon = horizon

    def _step(self, action: int) -> float:
        return  np.random.rand() < self.means[action]
    
    def _step_with_caps(self, action_counts: np.ndarray, actions: np.ndarray) -> float:
        # actions: vector of arm indices, each player picks one arm (len(actions) == num_players)
        # action_counts: number of players pulling each arm, shape [num_arms]
        # For each player: if their selected arm's count > cap, reward 0; else sample bernoulli(means[arm])

        # Determine for each player if too many are pulling that arm
        caps_exceeded = action_counts[actions] > self.caps[actions]
        # Sample Bernoulli trials for each player on their chosen arm
        random_draws = np.random.binomial(1, self.means[actions])
        individual_rewards = np.where(caps_exceeded, 0, random_draws)
        reward = np.sum(individual_rewards)
        return individual_rewards, reward
    
    def parameters(self) -> dict[int | np.ndarray]:
        return {'horizon': self.horizon, 'means': self.means, 'enable_stopping_action': self.enable_stopping_action, 'decentralized': self.decentralized, 'num_players': self.num_players, 'caps': self.caps, 'reward_struct': self.reward_struct}
    

