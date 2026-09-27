from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from icmape.models.robot_net.actor_lstm import ActorLSTM
from icmape.models.robot_net.actor_lstm_td3 import ActorLSTMTD3
from icmape.models.robot_net.critic_sac_lstm import CriticLSTM as CriticLSTMSAC
from icmape.models.robot_net.qlearning_net import QNetworkLSTM
from icmape.models.robot_net.inference_cnn import InferenceCNN
from icmape.models.robot_net.utils import to_lstm_parameters
from icmape.models.sequence_net.builder import (
    create_actor_network as _create_actor_network,
    create_critic_network_sac as _create_critic_network_sac,
    create_inference_network as _create_inference_network,
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        stddev = 1 / np.sqrt(m.weight.shape[1])
        torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2 * stddev, b=2 * stddev)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_critic_network_sac(
    state_dim: int,
    output_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config,
    grid_size: int = 50,
    bins: int = 1,
    n_critic: int = 2,
    architecture_type: str = "transformer",
    belief_channels: int = 1,
):
    if architecture_type  == "cnn_lstm":
        params = to_lstm_parameters(config)
        network = CriticLSTMSAC(
            state_dim=state_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            num_players=num_players,
            horizon=horizon,
            grid_size=grid_size,
            config=params,
            bins=bins,
            n_critic=n_critic,
            belief_channels=belief_channels,
        ).apply(init_weights)
        num_params = count_parameters(network)
        print(f"Robot Critic Network (cnn): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
        return network

    # keep non-LSTM architectures compatible with existing sequence-net impls
    return _create_critic_network_sac(
        state_dim=state_dim,
        output_dim=output_dim,
        num_actions=num_actions,
        num_players=num_players,
        horizon=horizon,
        config=config,
        bins=bins,
        n_critic=n_critic,
        architecture_type=architecture_type,
    )


def create_inference_network(
    config,
    grid_size: int = 50,
):
    # For robot wildfire inference we use a lightweight CNN that predicts a
    # per-cell probability map. It is compatible with the existing caller
    # interface (it ignores the extra timestep arg).
    hidden_channels = int(config.hidden_channels)
    in_channels = int(config.in_channels)

    network = InferenceCNN(
        grid_size=grid_size,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
    )
    num_params = count_parameters(network)
    print(
        f"Robot Inference Network (cnn): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)"
    )
    return network


def create_critic_network_qlearning(
    state_dim: int,
    output_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config,
    grid_size: int = 50,
    bins: int = 1,
    architecture_type: str = "cnn_lstm",
    belief_channels: int = 1,
):
    """Build a per-player Q-network for the robot Q-learning agent."""
    if architecture_type == "cnn_lstm":
        params = to_lstm_parameters(config)
        network = QNetworkLSTM(
            state_dim=state_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            num_players=num_players,
            horizon=horizon,
            grid_size=grid_size,
            config=params,
            bins=bins,
            belief_channels=belief_channels,
        ).apply(init_weights)
        num_params = count_parameters(network)
        print(
            f"Robot Q-Network (cnn_lstm): {num_params:,} parameters "
            f"({num_params / 1_000_000:.2f}M)"
        )
        return network
    raise ValueError(
        f"create_critic_network_qlearning: unsupported architecture_type={architecture_type!r}; "
        "only 'cnn_lstm' is supported for the robot Q-learning setting."
    )


def create_actor_network_td3(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config,
    grid_size: int = 50,
    shared_critic: bool = False,
    architecture_type: str = "cnn_lstm",
    belief_channels: int = 1,
):
    """Build a Gumbel-Softmax actor for robot TD3 (``cnn_lstm`` only)."""
    if architecture_type == "cnn_lstm":
        params = to_lstm_parameters(config)
        network = ActorLSTMTD3(
            state_dim=state_dim,
            num_actions=num_actions,
            num_players=num_players,
            horizon=horizon,
            grid_size=grid_size,
            config=params,
            shared_critic=shared_critic,
            belief_channels=belief_channels,
        ).apply(init_weights)
        num_params = count_parameters(network)
        print(
            f"Robot TD3 Actor Network (cnn_lstm): {num_params:,} parameters "
            f"({num_params / 1_000_000:.2f}M)"
        )
        return network
    raise ValueError(
        f"create_actor_network_td3: unsupported architecture_type={architecture_type!r}; "
        "only 'cnn_lstm' is supported for the robot setting."
    )


def create_actor_network(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config,
    grid_size: int = 50,
    shared_critic: bool = False,
    architecture_type: str = "transformer",
    belief_channels: int = 1,
):
    if architecture_type == "cnn_lstm":
        params = to_lstm_parameters(config)
        network = ActorLSTM(
            state_dim=state_dim,
            num_actions=num_actions,
            num_players=num_players,
            horizon=horizon,
            grid_size=grid_size,
            config=params,
            shared_critic=shared_critic,
            belief_channels=belief_channels,
        ).apply(init_weights)
        num_params = count_parameters(network)
        print(f"Robot Actor Network (cnn): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
        return network

    # keep non-LSTM architectures compatible with existing sequence-net impls
    return _create_actor_network(
        state_dim=state_dim,
        num_actions=num_actions,
        num_players=num_players,
        horizon=horizon,
        config=config,
        shared_critic=shared_critic,
        architecture_type=architecture_type,
    )
