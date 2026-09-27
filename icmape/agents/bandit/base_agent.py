from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import wandb
from abc import abstractmethod
from typing import Tuple, List, Dict
from icmape.dataclasses import Config, BaseAgentConfig
from icmape.envs.bandit.base import VecBanditEnv
from icmape.envs.bandit.builder import create_bandit_env, sample_env
from icmape.agents.bandit.history import HistoryData
from icmape.agents.bandit.utils import mle, stopping_rule

class BaseAgent(object):
    cfg: Config
    agent_cfg: BaseAgentConfig
    device: torch.device
    current_step: int

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.agent_cfg = cfg.agent
        self.device = device
        self.current_step = 0

    @abstractmethod
    def sample_trajectories(self, B: int, N: int) -> HistoryData:
        """Sample trajectories according to the current exploratory policy

        Args:
            B (int): number of environments to sample
            N (int): number of trajectories per environment

        Returns:
            HistoryData: Sampled Trajectories
        """
        raise Exception('Not implemented')

    @abstractmethod
    def train(self):
        raise Exception('Not Implemented')

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
        K = traj.exp_distribution.shape[-1]
        H = self.cfg.environment.horizon
        idenv = np.random.choice(B)

        eval_dict = {
            x: [float(y) for y in np.round(df_exp.loc[(idenv, x)].to_numpy(),3)] for x in [0, H//4, H//2, 3*H//4,H-1]
        }

        print(f'----------------------------------')
        print(f'[Eval] Agent: {self.agent_cfg.type} - Env: {self.cfg.environment.bandit_env.type} - Horizon: {self.cfg.environment.horizon}')
        print(f'[Eval] Env means: {traj.means[idenv]}')
        print(f'[Eval] MLE: { mle(traj.exp_actions, traj.rew_exp_actions, K)[idenv].mean(0)[-1]}')

   
        print(df_exp.loc[idenv].round(2))

        wandb.log({f"eval/exploration_dist_{k}": wandb.Histogram(v) for k,v in eval_dict.items()}, 
                  step=self.current_step)
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
    def _log_statistics(self, step: int, training_statistics: Dict[str, List[float]], mle_rewards: torch.Tensor,
                        log_probs_e_network: torch.Tensor, true_optimal: torch.Tensor,
                        qvalues: torch.Tensor, exploration_probs: torch.Tensor = None):
            """Logs statistics to Wandb

            Args:
                step (int): current training step
                training_statistics (Dict[str, List[float]]): A dictonary containing training statistics
                mle_rewards (torch.Tensor): The estimated maximum likelihood reward vector
                log_probs_e_network (torch.Tensor): log probabilities of the estimated optimal arm from the e network
                true_optimal (torch.Tensor): vector with the true optimal arms
                qvalues (torch.Tensor): qvalues of the exploration policy
                exploration_probs (Optional[torch.Tensor]): exploration probabilities of the true policy. Defaults to None
            """
            estm_rewards_argmax = (log_probs_e_network.argmax(-1) == true_optimal).float()
            estm_rewards_probs = log_probs_e_network.exp().gather(-1, true_optimal.unsqueeze(-1)).flatten()
    
            final_action_correct = mle_rewards
            mu = np.round(final_action_correct.mean().item(), 3)
            std = np.round(final_action_correct.std().item(), 3)
            estm_rewards_argmax = estm_rewards_argmax.detach().cpu().numpy()
            estm_rewards_probs = estm_rewards_probs.detach().cpu().numpy()

            print(f"Step {self.current_step + 1} completed. Final Action Correct: {mu}/{std} ")

            values = {
                "values/avg_qvalues": qvalues.detach().cpu().numpy().mean(),
                "values/std_avg_qvalues": qvalues.detach().cpu().numpy().mean(-1).std(),
                "values/max_qvalues": qvalues.max(-1)[0].mean().detach().cpu().numpy(),
                "values/min_qvalues": qvalues.min(-1)[0].mean().detach().cpu().numpy(),
            }

            performance = {
                "performance/mle_rewards": mle_rewards.mean().item(),
                "performance/est_rewards_argmax": estm_rewards_argmax.mean(),
                "performance/est_rewards_probs": estm_rewards_probs.mean(),
            }
            training_stats = {
                f"loss/{x}": np.mean(y) for x,y in training_statistics.items()
            }

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
            wandb.log({
                **performance, **training_stats, **values, **policy
            }, step=step)


    def should_stop(self, state: torch.Tensor, h: int, stopped: torch.Tensor):
        num_actions = self.cfg.environment.num_actions
        _mle = mle(state.actions, state.rewards, num_actions)[:,h]
        N = torch.nn.functional.one_hot(state.actions, num_actions).cumsum(1)[:,h]
        delta = 1-self.agent_cfg.confidence_level
        sigma = self.cfg.environment.bandit_env.std

        ret = np.ones(_mle.shape[0], dtype=np.bool_)
        idxs_not_stopped = stopped==False
        ret[idxs_not_stopped.detach().cpu().numpy()] = stopping_rule(_mle[idxs_not_stopped], N[idxs_not_stopped], delta, h, sigma, 0.4)
        return ret