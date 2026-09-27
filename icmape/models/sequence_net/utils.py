from __future__ import annotations
import torch
from torch import nn
from dataclasses import dataclass
from typing import NamedTuple, Optional


class TransformerState(NamedTuple):
    state: torch.Tensor
    timestep: torch.Tensor



@dataclass
class QMixerParameters:
    hypernet_layers: int
    hypernet_embed: int

@dataclass
class TransformerParameters:
    embed_dim: int
    num_heads: int
    num_layers: int
    attn_dropout: float
    resid_dropout: float
    embed_dropout: float
    activation_type: str = "gelu"

@dataclass
class LSTMParameters:
    embed_dim: int
    num_layers: int
    dropout: float
    activation_type: str = "gelu"
    gumbel_temperature: float = 1.0  # Gumbel-Softmax tau for DDPG/TD3 actor (actor_lstm_ddpg)




class ProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1e-9
        return x / x.norm(2, dim=-1, keepdim=True)

class GradientBottleneck(nn.Module):
    def __init__(self, alpha: float = 1):
        assert alpha > 0, 'alpha needs to be strictly positive'
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (x - self.alpha * x).detach()

class BaseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
 
    def hard_update(self, source: BaseTransformer) -> None:
        """ Hard update of the network from a source network """
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: BaseTransformer, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        return self.polyak_update(source, tau)

    def polyak_update(self, source: BaseTransformer, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def create_gpt2_config(
    horizon: int,
    params: TransformerParameters
) -> GPT2Config:
    """Helper function to create a GPT2Config with standard parameters"""
    return GPT2Config(
        n_positions=horizon,
        n_embd=params.embed_dim,
        n_layer=params.num_layers,
        n_head=params.num_heads,
        resid_pdrop=params.resid_dropout,
        embd_pdrop=params.embed_dropout,
        attn_pdrop=params.attn_dropout,
        use_cache=False,
        activation_function=params.activation_type,
    )

def create_lstm_config(
    params: LSTMParameters
) -> dict:
    """Helper function to create LSTM configuration dictionary with standard parameters
    
    Note: activation_type from params is used in embedding layers (e.g., F.gelu) but not
    in the LSTM layer itself, as PyTorch's nn.LSTM uses fixed activations (tanh/sigmoid).
    """
    return {
        'input_size': params.embed_dim,
        'hidden_size': params.embed_dim,
        'num_layers': params.num_layers,
        'dropout': params.dropout,
        'batch_first': True
    }



def to_transformer_parameters(config) -> TransformerParameters:
    """Convert config (dict/OmegaConf/TransformerParameters) to TransformerParameters.
    
    If config is already a TransformerParameters instance, returns it as-is.
    Otherwise, extracts relevant fields from dict/OmegaConf and creates TransformerParameters.
    """
    if isinstance(config, TransformerParameters):
        return config
    
    # Handle dict or OmegaConf (OmegaConf supports attribute access, dict uses getitem)
    if hasattr(config, 'embed_dim'):
        # OmegaConf or object with attributes
        return TransformerParameters(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            embed_dropout=config.embed_dropout,
            activation_type=getattr(config, 'activation_type', 'gelu')
        )
    else:
        # Regular dict
        return TransformerParameters(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            attn_dropout=config['attn_dropout'],
            resid_dropout=config['resid_dropout'],
            embed_dropout=config['embed_dropout'],
            activation_type=config.get('activation_type', 'gelu')
        )


def to_lstm_parameters(config) -> LSTMParameters:
    """Convert config (dict/OmegaConf/LSTMParameters) to LSTMParameters.
    
    If config is already a LSTMParameters instance, returns it as-is.
    Otherwise, extracts relevant fields from dict/OmegaConf and creates LSTMParameters.
    """
    if isinstance(config, LSTMParameters):
        return config
    
    # Handle dict or OmegaConf (OmegaConf supports attribute access, dict uses getitem)
    if hasattr(config, 'embed_dim'):
        # OmegaConf or object with attributes
        # OmegaConf or object with attributes
        # OmegaConf or object with attributes
        # OmegaConf or object with attributes
        return LSTMParameters(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation_type=getattr(config, 'activation_type', 'gelu'),
            gumbel_temperature=float(getattr(config, 'gumbel_temperature', 1.0)),
        )
    else:
        # Regular dict
        return LSTMParameters(
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            activation_type=config.get('activation_type', 'gelu'),
            gumbel_temperature=float(config.get('gumbel_temperature', 1.0)),
        )
   
class BaseLSTM(nn.Module):
    def __init__(self):
        super().__init__()
 
    def hard_update(self, source: BaseLSTM) -> None:
        """ Hard update of the network from a source network """
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: BaseLSTM, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        return self.polyak_update(source, tau)

    def polyak_update(self, source: BaseLSTM, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class BaseTargetUpdate(nn.Module):
    """Base for wrapper modules (e.g. MultiCriticLSTM) that need target network updates.
    Same API as BaseLSTM.hard_update / soft_update but source is any nn.Module."""
    def __init__(self):
        super().__init__()

    def hard_update(self, source: nn.Module) -> None:
        """Hard update: copy parameters from source into this module."""
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: nn.Module, tau: float) -> None:
        """Polyak (soft) update using time constant tau."""
        self.polyak_update(source, tau)

    def polyak_update(self, source: nn.Module, tau: float) -> None:
        """Polyak update: target = tau * source + (1 - tau) * target."""
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)




