from dataclasses import dataclass
from enum import Enum
from icmape.models.sequence_net.utils import TransformerParameters, LSTMParameters
from typing import List, Union, Optional

CriticParameters = Union[TransformerParameters, LSTMParameters]
EParameters = Union[TransformerParameters, LSTMParameters]
ActorParameters = Union[TransformerParameters, LSTMParameters]

class AlgorithmType(str, Enum):
    UNIFORM = 'Uniform'
    MULTICPE_QLEARNING = 'MULTICPE_QLEARNING'
    MULTICPE_SAC_CENTRALQ = 'MULTICPE_SAC_CENTRALQ'
    ROBOT_MULTICPE_SAC = 'ROBOT_MULTICPE_SAC'
    ROBOT_MULTICPE_TD3 = 'ROBOT_MULTICPE_TD3'
    ROBOT_MULTICPE_QLEARNING = 'ROBOT_MULTICPE_QLEARNING'
    MULTICPE_TD3 = 'MULTICPE_TD3'

@dataclass
class BaseAgentConfig:
    type: str
    num_env_instances: int
    num_trajectories: int
    confidence_level: float

@dataclass
class BaseOffPolicyAgentConfig(BaseAgentConfig):
    learning_rate: float
    final_learning_rate: float
    grad_clipping: float
    buffer_size: int
    gamma: float                        # discount factor for rewards
    polyak_update_q_network: float
    frequency_update_i_network: int
    batch_size: int                     # batch size
    replay_ratio: int
    inference_network_bootstrap_num_batches: int
    initial_buffer_bootstrap_num_batches: int
    critic_network: CriticParameters
    inference_network: EParameters
    cost_learning_rate: float
    architecture_type: str = "lstm"  # "transformer" or "lstm"
    device_ids: Optional[List[int]] = None  # e.g. [0, 1] -> critic/actor on cuda:0, inference on cuda:1
    exploration_epsilon_start: float = 1.0
    exploration_epsilon_end: float = 0.0
    exploration_epsilon_decay_steps: Optional[int] = None  # if None, decay over full training
    uniform_action_only: bool = False  # if True: no RL (Q) training; has_training_started stays False -> uniform exploration

@dataclass
class ICPEConfig(BaseOffPolicyAgentConfig):
    pass

@dataclass
class MULTICPE_SAC_Config(BaseOffPolicyAgentConfig):
    pass


