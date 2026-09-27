from enum import Enum
from dataclasses import dataclass

class BanditEnvType(str, Enum):
    GAUSSIAN_BANDIT = 'GaussianBandit'
    BERNOULLI_BANDIT = 'BernoulliBandit'


@dataclass
class BernoulliBandit:
    type: BanditEnvType = BanditEnvType.BERNOULLI_BANDIT

@dataclass
class GaussianBandit:
    min_value: float
    max_value: float
    std: float
    num_magic_actions: int
    std_magic_action: float
    minimum_gap: float
    type: BanditEnvType = BanditEnvType.GAUSSIAN_BANDIT  

@dataclass
class BanditEnvConfig:
    bandit_env: GaussianBandit | BernoulliBandit
    num_actions: int
    horizon: int
    enable_stopping_action: bool
    decentralized: bool
    num_players: int
    num_rewards: int = 1 
