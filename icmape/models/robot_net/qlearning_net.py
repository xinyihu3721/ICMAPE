from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from icmape.models.robot_net.utils import BaseLSTM, CNN50Encoder, LSTMParameters


class QNetworkLSTM(BaseLSTM):
    """Decentralised Q-network for the robot wildfire environment.

    Differences from ``CriticLSTM`` (SAC):
    - Each player has its **own** CNN that sees only its local state
      ``states[:, p, :, :]`` (single channel) -- fully decentralised.
    - The location LSTM is **shared** across all players.
    - No others' actions: each player conditions only on its own local
      information (local state + location history + player ID).
    - Single output head (no min-over-heads); returns a one-element list for
      API compatibility with the rest of the codebase.
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters,
        grid_size: int = 50,
        bins: int = 1,
        belief_channels: int = 1,
    ):
        super().__init__()
        self._horizon = horizon
        self._num_actions = num_actions
        self._num_players = num_players
        self._cfg = config
        self._embed_dim = config.embed_dim
        self._belief_channels = int(belief_channels)

        # One CNN per player -- each sees only its own local state channel(s).
        self._player_cnns = nn.ModuleList([
            CNN50Encoder(
                in_channels=self._belief_channels,
                out_dim=self._embed_dim,
            )
            for _ in range(num_players)
        ])

        # Shared location LSTM across all players.
        self._loc_lstm = nn.LSTM(
            input_size=num_players * 2,
            hidden_size=self._embed_dim,
            num_layers=max(1, config.num_layers),
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.register_buffer(
            "player_id_encode",
            F.one_hot(torch.arange(num_players), num_classes=num_players).float(),
        )

        # fuse: z_grid (E) + z_loc (E) + player_id (P)
        fuse_in_dim = self._embed_dim * 2 + num_players
        self._fuse_layer = nn.Sequential(
            nn.Linear(fuse_in_dim, self._embed_dim),
            nn.ReLU(),
        )
        self._output_layer = nn.Linear(self._embed_dim, output_dim)

    def _encode_location_history(
        self,
        hist: torch.Tensor,
        timestep: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Shared LSTM over all players' location histories.

        hist:     ``[B, P, T, 2]``
        timestep: ``[B]``
        returns:  ``[B, E]``
        """
        B, P, T, _ = hist.shape
        seq = hist.permute(0, 2, 1, 3).reshape(B, T, P * 2).float()
        output, _ = self._loc_lstm(seq)                        # [B, T, E]
        idx = timestep.long().clamp(max=T - 1)
        return output[torch.arange(B, device=device), idx]    # [B, E]

    def forward(
        self,
        states: torch.Tensor,
        location_history: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            states:           ``[B, P, G, G]``
            location_history: ``[B, P, T, 2]``
            timestep:         ``[B]``

        Returns:
            One-element list containing ``[B, P, num_actions]`` Q-values.
        """
        B, P, G, _ = states.shape
        device = states.device
        states_f = states.float()

        # Per-player CNN: each player sees only its own local state.
        z_grid = torch.stack(
            [F.gelu(cnn(states_f[:, p : p + self._belief_channels, :, :]))
             for p, cnn in enumerate(self._player_cnns)],
            dim=1,
        )  # [B, P, E]

        # Shared location LSTM, broadcast to all players.
        z_loc = self._encode_location_history(
            location_history, timestep, device
        ).unsqueeze(1).expand(B, P, self._embed_dim)  # [B, P, E]

        # Player ID one-hots.
        player_ids = self.player_id_encode.unsqueeze(0).expand(B, -1, -1)  # [B, P, P]

        fused = torch.cat([z_grid, z_loc, player_ids], dim=-1)  # [B, P, fuse_in_dim]
        q_values = self._output_layer(self._fuse_layer(fused))  # [B, P, num_actions]

        return q_values
