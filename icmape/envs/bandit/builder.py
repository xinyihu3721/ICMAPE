import numpy as np
from icmape.envs.bandit.base import BanditEnv, VecBanditEnv
from icmape.envs.bandit.config import BanditEnvType, BanditEnvConfig
from icmape.envs.bandit.gaussian_bandit import GaussianBanditEnv
from icmape.envs.bandit.bernoulli_bandit import BernoulliBanditEnv
from icmape.envs.bandit.utils import embed_magic_actions, minimum_gap_sampling, beta_sampling

def create_bandit_env(type: BanditEnvType, **kwargs) -> BanditEnv:
    """Sample a random environment according to the configuration

    Args:
        type (BanditEnvType): bandit env type
        params (dict): parameters of the environment

    Raises:
        Exception: If the type of environment is not found

    Returns:
        BanditEnv: The random bandit environment
    """
    if type == BanditEnvType.GAUSSIAN_BANDIT:
        return GaussianBanditEnv(**kwargs)
    elif type == BanditEnvType.BERNOULLI_BANDIT:
        return BernoulliBanditEnv(**kwargs)
    raise Exception(f'Environment type {type} not implemented!')

def sample_env(cfg: BanditEnvConfig) -> BanditEnv:
    """Sample a random environment according to the configuration

    Args:
        cfg (BanditEnvConfig): bandit env configuration

    Raises:
        Exception: If the type of environment is not found

    Returns:
        BanditEnv: The random bandit environment
    """
    if cfg.bandit_env.type == BanditEnvType.GAUSSIAN_BANDIT:
        if 'minimum_gap' in cfg.bandit_env:
            minimum_gap = cfg.bandit_env.minimum_gap
        else:
            minimum_gap = 0
        means = minimum_gap_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, minimum_gap)
        stds = cfg.bandit_env.std * np.ones(cfg.num_actions)
        if cfg.capacity == 'unit':
            caps = np.ones(cfg.num_actions)
        else:
            caps = np.random.randint(1, cfg.num_players + 1, size=cfg.num_actions) 

        if cfg.bandit_env.num_magic_actions > 0:
            means, stds = embed_magic_actions(np.array(means), np.array(stds), cfg.bandit_env.num_magic_actions, cfg.bandit_env.std_magic_action)
        return GaussianBanditEnv(means=means,
                                 caps=caps,
                                 horizon=cfg.horizon,
                                 stds=stds,
                                 enable_stopping_action=cfg.enable_stopping_action,
                                 decentralized=cfg.decentralized,
                                 num_players=cfg.num_players)
    elif cfg.bandit_env.type == BanditEnvType.BERNOULLI_BANDIT:
        means = np.random.rand(cfg.num_actions)
        if cfg.capacity == 'unit':
            caps = np.ones(cfg.num_actions)
        else:
            caps = np.random.randint(1, cfg.num_players + 1, size=cfg.num_actions) 
            
        if 'minimum_gap' in cfg.bandit_env:
            minimum_gap = cfg.bandit_env.minimum_gap
        else:
            minimum_gap = 0
        means = minimum_gap_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, minimum_gap)
        return BernoulliBanditEnv(horizon=cfg.horizon, means=means, enable_stopping_action=cfg.enable_stopping_action, decentralized=cfg.decentralized, num_players=cfg.num_players, caps=caps)
    raise Exception(f'Environment type {cfg.bandit_env.type} not implemented!')


def sample_env_beta(cfg: BanditEnvConfig, alpha: float, beta_param: float) -> BanditEnv:
    """Like sample_env but draws arm means from Beta(alpha, beta_param) scaled to [min_value, max_value].
    Capacities are sampled the same way as in sample_env. Only Gaussian and Bernoulli types supported.
    """
    if cfg.bandit_env.type == BanditEnvType.GAUSSIAN_BANDIT:
        means = beta_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, alpha, beta_param)
        stds  = cfg.bandit_env.std * np.ones(cfg.num_actions)
        caps  = np.ones(cfg.num_actions) if cfg.capacity == 'unit' else np.random.randint(1, cfg.num_players + 1, size=cfg.num_actions)
        if cfg.bandit_env.num_magic_actions > 0:
            means, stds = embed_magic_actions(np.array(means), np.array(stds), cfg.bandit_env.num_magic_actions, cfg.bandit_env.std_magic_action)
        return GaussianBanditEnv(means=means, caps=caps, horizon=cfg.horizon, stds=stds,
                                 enable_stopping_action=cfg.enable_stopping_action,
                                 decentralized=cfg.decentralized, num_players=cfg.num_players)
    elif cfg.bandit_env.type == BanditEnvType.BERNOULLI_BANDIT:
        means = beta_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, alpha, beta_param)
        caps  = np.ones(cfg.num_actions) if cfg.capacity == 'unit' else np.random.randint(1, cfg.num_players + 1, size=cfg.num_actions)
        return BernoulliBanditEnv(horizon=cfg.horizon, means=means, enable_stopping_action=cfg.enable_stopping_action,
                                  decentralized=cfg.decentralized, num_players=cfg.num_players, caps=caps)
    raise Exception(f'sample_env_beta: unsupported env type {cfg.bandit_env.type}')


def sample_env_with_cap_theta(cfg: BanditEnvConfig, theta: np.ndarray) -> BanditEnv:
    """Sample env where arm capacities are drawn from Categorical(theta).
    theta is a pre-computed probability vector of length num_players.
    Arm means are sampled the same way as in sample_env.
    """
    caps = (np.random.choice(len(theta), size=cfg.num_actions, p=theta) + 1).astype(np.int32)
    if cfg.bandit_env.type == BanditEnvType.GAUSSIAN_BANDIT:
        minimum_gap = cfg.bandit_env.minimum_gap if 'minimum_gap' in cfg.bandit_env else 0
        means = minimum_gap_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, minimum_gap)
        stds  = cfg.bandit_env.std * np.ones(cfg.num_actions)
        if cfg.bandit_env.num_magic_actions > 0:
            means, stds = embed_magic_actions(np.array(means), np.array(stds), cfg.bandit_env.num_magic_actions, cfg.bandit_env.std_magic_action)
        return GaussianBanditEnv(means=means, caps=caps, horizon=cfg.horizon, stds=stds,
                                 enable_stopping_action=cfg.enable_stopping_action,
                                 decentralized=cfg.decentralized, num_players=cfg.num_players)
    elif cfg.bandit_env.type == BanditEnvType.BERNOULLI_BANDIT:
        minimum_gap = cfg.bandit_env.minimum_gap if 'minimum_gap' in cfg.bandit_env else 0
        means = minimum_gap_sampling(cfg.num_actions, cfg.bandit_env.min_value, cfg.bandit_env.max_value, minimum_gap)
        return BernoulliBanditEnv(horizon=cfg.horizon, means=means, enable_stopping_action=cfg.enable_stopping_action,
                                  decentralized=cfg.decentralized, num_players=cfg.num_players, caps=caps)
    raise Exception(f'sample_env_with_cap_theta: unsupported env type {cfg.bandit_env.type}')




def sample_envs(cfg: BanditEnvConfig, B: int) -> VecBanditEnv:
    """Sample B random environments according to the configuration

    Args:
        cfg (BanditEnvConfig): bandit env configuration
        B (int): number of environemnts

    Returns:
        BatchedBanditEnv: The random bandit environments
    """
    return VecBanditEnv([sample_env(cfg) for _ in range(B)])