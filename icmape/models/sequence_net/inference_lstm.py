from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from icmape.models.sequence_net.utils import BaseLSTM, LSTMParameters
from icmape.models.sequence_net.lstm_utils import LSTMModel

class InferenceLSTM(BaseLSTM):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_players = num_players
        self._state_dim = state_dim
        self._cfg = config
        self._horizon = horizon

        self._embed = nn.Linear(state_dim, config.embed_dim)
        self._lstm = LSTMModel(
            emb_size=config.embed_dim,
            lstm_hid_size=config.embed_dim,
            lstm_layers=config.num_layers
        )

        self._output = nn.Linear(config.embed_dim, num_actions*(num_players))
        
    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            states: [B, T, state_dim] input states
            timestep: [B] input timesteps
        Returns:
            log_probs: [B, num_actions, num_players] log probabilities
        """

        x = F.gelu(self._embed(states))

        # Create mask for valid timesteps based on timestep parameter
        batch_size = states.shape[0]
        seq_len = states.shape[1]
        # Clamp timestep to ensure at least one valid timestep (avoid NaN in attention pooling)
        
        timestep_clamped = timestep.clamp(min=1, max=seq_len)
        # Create mask: True for valid timesteps (up to timestep), False for padding
        mask = torch.arange(seq_len, device=states.device).unsqueeze(0).expand(batch_size, seq_len)
        mask = mask < timestep_clamped.unsqueeze(1)
       
        z = self._lstm(x)  # [B, T, embed_dim]
        h_last = self._lstm.attn_pool(z, mask=mask)  # [B, embed_dim]

        out = self._output(h_last)

        # the reshaped out has shape [batch_size, base_num_actions, num_players]
        log_probs = F.log_softmax(out.view(-1, self._num_actions, self._num_players), dim=-1)
       
        return log_probs



