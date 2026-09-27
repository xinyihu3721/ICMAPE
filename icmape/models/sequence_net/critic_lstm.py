"""DQN-style single-head LSTM critic (same interface as CriticTransformer)."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from icmape.models.sequence_net.utils import BaseLSTM, LSTMParameters
from icmape.models.sequence_net.lstm_utils import LSTMModel


class CriticLSTM(BaseLSTM):
    """Single-head LSTM critic for DQN-style agents. Same API as CriticTransformer."""

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters,
        bins: int = 1,
    ):
        super().__init__()
        self._horizon = horizon
        self._num_actions = num_actions
        self._num_players = num_players
        self._cfg = config
        self.bins = bins
        self._state_dim = state_dim

        self._embed = nn.Linear(state_dim, config.embed_dim)
        self._lstm = LSTMModel(
            emb_size=config.embed_dim,
            lstm_hid_size=config.embed_dim,
            lstm_layers=config.num_layers,
        )
        self._output = nn.Linear(config.embed_dim, output_dim)

    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            states: [B, H, state_dim] input states (sequence of state-action pairs)
            timestep: Optional [B] or [B*P] timestep (for API compatibility).

        Returns:
            values: [B, num_actions] Q-value estimates for each action
        """
        batch_size = states.shape[0]
        seq_len = states.shape[1]
        x = F.gelu(self._embed(states))

        if timestep is not None:
            timestep_clamped = timestep.clamp(min=1, max=seq_len)
            mask = torch.arange(seq_len, device=states.device).unsqueeze(0).expand(batch_size, seq_len)
            mask = mask < timestep_clamped.unsqueeze(1)
        else:
            mask = None

        z = self._lstm(x)
        z_pooled = self._lstm.attn_pool(z, mask=mask)
        out = self._output(z_pooled)
        return out
