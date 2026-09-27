from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from icmape.models.robot_net.utils import BaseLSTM, CNN50Encoder, LSTMParameters


class ActorLSTM(BaseLSTM):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters,
        grid_size: int = 15,
        shared_critic: bool = False,
        belief_channels: int = 1,
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_players = num_players
        self._state_dim = state_dim
        self._horizon = horizon
        self._grid_size = int(grid_size)
        self._belief_channels = int(belief_channels)
        self._cfg = config
        self.shared_critic = shared_critic
        self._embed_dim = config.embed_dim

        # one CNN per player
        self._cnn = nn.ModuleList(
            [
                CNN50Encoder(
                    in_channels=self._belief_channels,
                    out_dim=self._embed_dim,
                )
                for _ in range(self._num_players)
            ]
        )

        # one LSTM per player, all read the same global location history
        self._loc_lstm = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=num_players * 2,
                    hidden_size=self._embed_dim,
                    num_layers=max(1, config.num_layers),
                    batch_first=True,
                    dropout=config.dropout if config.num_layers > 1 else 0.0,
                )
                for _ in range(self._num_players)
            ]
        )

        # one fusion head per player
        self._fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._embed_dim * 2, self._embed_dim),
                    nn.GELU(),
                )
                for _ in range(self._num_players)
            ]
        )

        # one output head per player
        self._output = nn.ModuleList(
            [
                nn.Linear(self._embed_dim, num_actions + 1)
                for _ in range(self._num_players)
            ]
        )

    def _encode_location_history(
        self,
        hist: torch.Tensor,
        timestep: torch.Tensor,
        loc_lstm: nn.LSTM,
        device: torch.device,
    ) -> torch.Tensor:
        """
        hist: [B, P, T, 2]
        timestep: [B]
        return: [B, E]
        """
        B, P, T, _ = hist.shape
        seq = hist.permute(0, 2, 1, 3).reshape(B, T, P * 2).float()
        output, _ = loc_lstm(seq)  # [B, T, E]

        idx = timestep.long().clamp(max=T - 1)
        emb = output[torch.arange(B, device=device), idx]  # [B, E]
        return emb

    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        location_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        states: [B, P, G, G]
        timestep: [B]
        location_history: [B, P, T, 2]

        returns:
            probs: [B, P, A+1]
            log_probs: [B, P, A+1]
        """
        output_list = []

        for i, (cnn, loc_lstm, fuse, output) in enumerate(
            zip(self._cnn, self._loc_lstm, self._fuse, self._output)
        ):
            x_i = states[:, i : i + 1, :, :].float()      # [B,1,G,G]
            z_grid = F.gelu(cnn(x_i))                     # [B,E]

            z_loc = self._encode_location_history(
                location_history,
                timestep,
                loc_lstm,
                states.device,
            )                                            # [B,E]

            fused = fuse(torch.cat([z_grid, z_loc], dim=-1))   # [B,E]
            logits = output(fused)                             # [B,A+1]
            log_probs = F.log_softmax(logits, dim=-1)          # [B,A+1]
            output_list.append(log_probs)

        log_probs = torch.stack(output_list, dim=1)            # [B,P,A+1]
        probs = log_probs.exp()

        return probs, log_probs