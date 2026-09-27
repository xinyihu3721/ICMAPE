from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
from transformers import GPT2Model
from icmape.models.sequence_net.utils import BaseTransformer, TransformerParameters, create_gpt2_config

class ActorTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: TransformerParameters,
        shared_critic: bool
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_players = num_players
        self._state_dim = state_dim
        self._cfg = config
        self.shared_critic = shared_critic
        self._gpt2_config = create_gpt2_config(horizon, config)
        self._gpt2 = GPT2Model(self._gpt2_config)

        self._embed = nn.Linear(state_dim, config.embed_dim)

        output_size = num_actions + 1
        self._output = nn.Linear(config.embed_dim, output_size)


    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass
        
        Args:
            state: [B, T, num_players+num_actions+reward] input states player_id+action_id+reward

        Returns:
            - probabilities         [B, T, num_actions]
            - log probabilities     [B, T, num_actions]
        """
        h = F.gelu(self._embed(states))

        h = self._gpt2(inputs_embeds=h)
        h = h.last_hidden_state

        h = h[:, -1, :]

   

        out = self._output(h)
        
        log_probs = F.log_softmax(out, dim=-1)
        
        return log_probs.exp(), log_probs