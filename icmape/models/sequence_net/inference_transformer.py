from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2Model
from icmape.models.sequence_net.utils import BaseTransformer, TransformerParameters, create_gpt2_config

class InferenceTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: TransformerParameters
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_players = num_players
        self._state_dim = state_dim
        self._cfg = config

        self._gpt2_config = create_gpt2_config(horizon, config)
        self._gpt2 = GPT2Model(self._gpt2_config)

        self._embed = nn.Linear(state_dim, config.embed_dim)

        self._output = nn.Linear(config.embed_dim, num_actions*(num_players))

    def forward(
        self,
        states: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass

        Args:
            actions: [B, T, action_size] one-hot encoded actions
            rewards: [B, T] rewards

        Returns:
            log_probs: [B, T, action_size] log probabilities
        """
    
        h = F.gelu(self._embed(states))

        h = self._gpt2(inputs_embeds=h)
        h = h.last_hidden_state
        h = h[:, -1, :]

        out = self._output(h)

        #log_probs = F.log_softmax(out, dim=-1)

        # the reshaped out has shape [batch_size, base_num_actions, num_players]
        
        log_probs = F.log_softmax(out.view(-1, self._num_actions, self._num_players), dim=-1)
       

        return log_probs