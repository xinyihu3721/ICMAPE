from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Union
from icmape.models.sequence_net.actor_lstm import ActorLSTM
from icmape.models.sequence_net.critic_sac_lstm_100 import CriticLSTM as CriticLSTMSAC
from icmape.models.sequence_net.inference_lstm import InferenceLSTM
from icmape.models.sequence_net.utils import (
    TransformerParameters,
    LSTMParameters,
    QMixerParameters,
    to_transformer_parameters,
    to_lstm_parameters,
)
from icmape.models.sequence_net.critic_lstm import CriticLSTM as CriticLSTM
from icmape.models.sequence_net.critic_td3_lstm import CriticLSTM as CriticLSTM_TD3
from icmape.models.sequence_net.actor_lstm_ddpg import ActorLSTM as ActorLSTM_DDPG
from icmape.models.sequence_net.q_mixer import QMixer


def init_weights(m):
    if isinstance(m, nn.Linear):
        stddev = 1 / np.sqrt(m.weight.shape[1])
        torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_critic_network(
    state_dim: int,
    output_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters,
    bins: int = 1,
    architecture_type: str = "transformer"
) -> CriticLSTM:
    # Convert config and create corresponding network
    if architecture_type == "lstm":
        params = to_lstm_parameters(config)
        network = CriticLSTM(state_dim, output_dim, num_actions, num_players, horizon, params, bins).apply(init_weights)
    elif architecture_type == "transformer":
        params = to_transformer_parameters(config)
        network = CriticTransformer(state_dim, output_dim, num_actions, num_players, horizon, params, bins).apply(init_weights)
    else:
        raise ValueError(f"Invalid architecture type: {architecture_type}")
    num_params = count_parameters(network)
    print(f"Critic Network ({architecture_type}): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
    return network


def create_critic_network_sac(
    state_dim: int,
    output_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters,
    bins: int = 1,
    n_critic: int = 2,
    architecture_type: str = "transformer"
) -> CriticLSTMSAC:
    # Convert config and create corresponding network
    if architecture_type == "lstm":
        params = to_lstm_parameters(config)
        network = CriticLSTMSAC(state_dim, output_dim, num_actions, num_players, horizon, params, bins, n_critic=n_critic).apply(init_weights)
    elif architecture_type == "transformer":
        params = to_transformer_parameters(config)
        network = CriticTransformerSAC(state_dim, output_dim, num_actions, num_players, horizon, params, bins, n_critic=n_critic).apply(init_weights)
    else:
        raise ValueError(f"Invalid architecture type: {architecture_type}")
    num_params = count_parameters(network)
    print(f"Critic Network ({architecture_type}): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
    return network


def create_critic_network_td3(
    state_dim: int,
    output_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters,
    bins: int = 1,
    n_critic: int = 2,
    architecture_type: str = "transformer"
) -> CriticLSTM_TD3:
    # Convert config and create corresponding network
    if architecture_type == "lstm":
        params = to_lstm_parameters(config)
        network = CriticLSTM_TD3(state_dim, output_dim, num_actions, num_players, horizon, params, bins, n_critic=n_critic).apply(init_weights)
    else:
        raise ValueError(f"Invalid architecture type: {architecture_type}")
    num_params = count_parameters(network)
    print(f"Critic Network ({architecture_type}): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
    return network

def create_inference_network(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters,
    architecture_type: str = "transformer"
) -> InferenceLSTM:
    # Convert config and create corresponding network
    if architecture_type == "lstm":
        params = to_lstm_parameters(config)
        network = InferenceLSTM(state_dim, num_actions, num_players, horizon, params).apply(init_weights)
    elif architecture_type == "transformer":
        params = to_transformer_parameters(config)
        network = InferenceTransformer(state_dim, num_actions, num_players, horizon, params).apply(init_weights)
    else:
        raise ValueError(f"Invalid architecture type: {architecture_type}")
    
    num_params = count_parameters(network)
    print(f"Inference Network ({architecture_type}): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
    return network


def create_actor_network(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters,
    shared_critic: bool = False,
    architecture_type: str = "transformer"
) -> ActorLSTM:
    # Convert config and create corresponding network
    if architecture_type == "lstm":
        params = to_lstm_parameters(config)
        network = ActorLSTM(state_dim, num_actions, num_players, horizon, params, shared_critic).apply(init_weights)
    elif architecture_type == "transformer":
        params = to_transformer_parameters(config)
        network = ActorTransformer(state_dim, num_actions, num_players, horizon, params, shared_critic).apply(init_weights)
    else:
        raise ValueError(f"Invalid architecture type: {architecture_type}")
    
    # Print parameter count
    num_params = count_parameters(network)
    print(f"Actor Network ({architecture_type}): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
    return network


def create_actor_network_td3(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters,
    shared_critic: bool = False,
    architecture_type: str = "lstm",
) -> ActorLSTM_DDPG:
    """TD3 / DDPG-style actor: LSTM trunk + linear logits -> ``(probs, log_probs)`` like ``ActorLSTM``."""
    if architecture_type != "lstm":
        raise ValueError(f"create_actor_network_td3 only supports lstm, got {architecture_type!r}")
    params = to_lstm_parameters(config)
    network = ActorLSTM_DDPG(
        state_dim, num_actions, num_players, horizon, params, shared_critic
    ).apply(init_weights)
    num_params = count_parameters(network)
    print(
        f"Actor Network (TD3 DDPG lstm): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)"
    )
    return network


def create_separate_actor_network(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: TransformerParameters | LSTMParameters | MambaParameters,
    shared_critic: bool = False,
    architecture_type: str = "lstm"
) -> MultiActorLSTM:
    # Convert config and create corresponding network
    if architecture_type == "lstm":
        params = to_lstm_parameters(config)
        network = MultiActorLSTM(num_players, state_dim, num_actions, horizon, params, shared_critic).apply(init_weights)
    else:
        raise ValueError(f"Invalid architecture type: {architecture_type}")
    # Print parameter count
    num_params = count_parameters(network)
    print(f"Actor Network ({architecture_type}): {num_params:,} parameters ({num_params / 1_000_000:.2f}M)")
    return network



def create_q_mixer(
    state_dim: int,
    num_actions: int,
    num_players: int,
    horizon: int,
    config: QMixerParameters,
    mixer_type: str = "Qmix"
) -> QMixer | QattenMixer:
    if mixer_type == "Qmix":
        network = QMixer(state_dim, num_actions, num_players, config)
    elif mixer_type == "Qatten":
        network = QattenMixer(state_dim, num_actions, num_players, horizon)
    else:
        raise ValueError(f"Invalid type: {type}")
    #network = QattenMixer(state_dim, num_actions, num_players, horizon)
    network.apply(init_weights)
    return network
