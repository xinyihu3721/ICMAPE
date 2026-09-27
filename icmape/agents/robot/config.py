"""Robot sequential agent config types (Hydra ``configs/agent/robot``).

These mirror the bandit off-policy / MULTICPE SAC dataclasses so YAML fields stay compatible,
while giving robot code a dedicated import path.
"""

from __future__ import annotations

from dataclasses import dataclass

from icmape.agents.bandit.config import (
    AlgorithmType,
    BaseAgentConfig,
    BaseOffPolicyAgentConfig,
    MULTICPE_SAC_Config,
)

__all__ = [
    "AlgorithmType",
    "BaseAgentConfig",
    "RobotBaseOffPolicyAgentConfig",
    "RobotMULTICPE_SAC_Config",
    "RobotMULTICPE_QLEARNING_Config",
]


@dataclass
class RobotBaseOffPolicyAgentConfig(BaseOffPolicyAgentConfig):
    """Off-policy agent block for robot sequential training."""

    # ``icpe.models.robot_net.builder`` uses ``cnn_lstm`` for grid CNN + location LSTM hybrids.
    architecture_type: str = "cnn_lstm"


@dataclass
class RobotMULTICPE_SAC_Config(MULTICPE_SAC_Config):
    """MULTICPE SAC agent config for ``AlgorithmType.ROBOT_MULTICPE_SAC``."""

    architecture_type: str = "cnn_lstm"


@dataclass
class RobotMULTICPE_QLEARNING_Config(RobotBaseOffPolicyAgentConfig):
    """MULTICPE Q-learning agent config for ``AlgorithmType.ROBOT_MULTICPE_QLEARNING``.

    No actor network -- Q-values drive action selection via epsilon-greedy.
    VDN_mode is a field on ``critic_network`` (same as bandit ICPE).
    """

    architecture_type: str = "cnn_lstm"
