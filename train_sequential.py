import torch
import numpy as np
import hydra
import os
import wandb
from datetime import datetime
from omegaconf import DictConfig
from icmape.agents.bandit.sequential.multicpe_sac import MULTICPE_SAC_CentralQ
from icmape.agents.bandit.sequential.multicpe_qlearning import MULTICPE_QLEARNING
from icmape.agents.bandit.sequential.multicpe_td3 import MULTICPE_TD3
from icmape.dataclasses import Config
from icmape.agents.bandit.config import AlgorithmType


def run_train(cfg: DictConfig) -> None:
    cfg: Config = Config(**cfg)
    date = datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
    print(f'Configuration {cfg}', flush=True)

    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    env = cfg.environment
    env_type_tag = getattr(getattr(env, "bandit_env", None), "type", None) or "env"

    wandb_run_name = (
        f'{cfg.agent.type}_{cfg.agent.architecture_type} - '
        f'({cfg.environment.horizon}/{cfg.environment.num_actions}/{cfg.environment.num_players}/'
        f'{cfg.experiment.seed})'
    )
    wandb_tags = [
        cfg.agent.type,
        f'{env_type_tag}',
        f'H={cfg.environment.horizon}',
        f'K={cfg.environment.num_actions}',
        f'P={cfg.environment.num_players}',
        cfg.agent.architecture_type,
    ]
    if getattr(cfg.agent, "uniform_action_only", False):
        wandb_run_name = f"UNIFORM_{wandb_run_name}"
        wandb_tags.append("uniform_action_only")
    elif getattr(getattr(cfg.agent, "critic_network", None), "VDN_mode", False):
        wandb_run_name = f"VDN0_{wandb_run_name}"
        wandb_tags.append("VDN0_mode")

    wandb.init(
        project=cfg.experiment.name,
        notes=f'[{date}] - {cfg.experiment.description}',
        name=wandb_run_name,
        tags=wandb_tags,
        group=getattr(cfg.experiment, 'wandb_group', None),
        config=cfg
    )

    # Set algo tag to distinguish algorithm families and ICPE variants
    agent_type = cfg.agent.type
    if agent_type == AlgorithmType.MULTICPE_TD3:
        algo = "TD3"
    elif agent_type in (
        AlgorithmType.MULTICPE_SAC_CENTRALQ,
    ):
        algo = "SAC"
    elif agent_type in (AlgorithmType.MULTICPE_QLEARNING):
        if getattr(cfg.agent, "uniform_action_only", False):
            algo = "Uniform"
        elif getattr(getattr(cfg.agent, "critic_network", None), "VDN_mode", False):
            algo = "VDN"
        else:
            algo = "INDEP"
    else:
        algo = str(agent_type)
    wandb.config.update({"algo": algo})

    # Check if CUDA should be disabled via environment variable
    if os.environ.get('FORCE_CPU', 'false').lower() == 'true':
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}', flush=True)

    if cfg.agent.type == AlgorithmType.MULTICPE_QLEARNING:
        agent = MULTICPE_QLEARNING(cfg, device)
    elif cfg.agent.type == AlgorithmType.MULTICPE_SAC_CENTRALQ:
        agent = MULTICPE_SAC_CentralQ(cfg, device)
    elif cfg.agent.type == AlgorithmType.MULTICPE_TD3:
        agent = MULTICPE_TD3(cfg, device)
    else:
        raise Exception(f'Agent {cfg.agent.type} not implemented')

    # Save path: ICPE + uniform_action_only → fixed experiment tag (no shared critic / uniform sampling).
    exp_name_for_models = cfg.experiment.name
    if cfg.agent.type == AlgorithmType.MULTICPE_QLEARNING and getattr(
        cfg.agent, "uniform_action_only", False
    ):
        exp_name_for_models = "UNIFORM_ACTION"

    env_name = (
        f'{exp_name_for_models}_H{cfg.environment.horizon}'
        f'_K{cfg.environment.num_actions}_P{cfg.environment.num_players}'
    )
    PATH = f'/models/bandits/{env_name}_{algo}'
    # Create directory if it doesn't exist, or use existing one if it does
    os.makedirs(PATH, exist_ok=True)
    agent.train(PATH)
    
    agent.save(PATH)

    print("Training complete.", flush=True)

    wandb.finish()


@hydra.main(config_path="configs", config_name="config_sequential.yaml")
def train(cfg: DictConfig):
    """Bandit default root is ``config_sequential.yaml``. Robot: ``--config-name=config_robot`` (see ``train_robot.py``)."""
    run_train(cfg)


if __name__ == '__main__':
    train()
