from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from icmape.models.robot_net.utils import BaseLSTM, CNN50Encoder, LSTMParameters


class CriticLSTM(BaseLSTM):
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
        n_critic: int = 2,
        belief_channels: int = 1,
    ):
        super().__init__()
        self._horizon = horizon
        self._num_actions = num_actions
        self._num_players = num_players
        self._cfg = config
        self.bins = bins
        self._state_dim = state_dim
        self._action_dim = num_actions
        self._grid_size = int(grid_size)
        self._belief_channels = int(belief_channels)
        self._embed_dim = config.embed_dim
        self._joint_action_feat_dim = self._num_players * self._num_actions

        # Separate multi-channel CNN per critic -- fully independent like bandit CriticLSTM.
        # Each CNN takes [B, P, G, G] (P channels) and produces [B, E].
        self._cnns = nn.ModuleList([
            CNN50Encoder(
                in_channels=num_players * self._belief_channels,
                out_dim=self._embed_dim,
            )
            for _ in range(n_critic)
        ])

        self.register_buffer(
            "player_id_encode",
            F.one_hot(torch.arange(num_players), num_classes=num_players).float(),
        )

        # Separate location LSTM per critic -- mirrors bandit CriticLSTM.
        self._loc_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=num_players * 2,
                hidden_size=self._embed_dim,
                num_layers=max(1, config.num_layers),
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            for _ in range(n_critic)
        ])

        # one fuse block per critic head
        fuse_in_dim = self._embed_dim * 2 + num_players + self._joint_action_feat_dim
        self._fuse_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fuse_in_dim, self._embed_dim),
                    nn.ReLU(),
                )
                for _ in range(n_critic)
            ]
        )

        self.critics_output = nn.ModuleList(
            [nn.Linear(self._embed_dim, output_dim) for _ in range(n_critic)]
        )

    def action_onehot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.long(), num_classes=self._num_actions)

    def action_masking(self, action: torch.Tensor) -> torch.Tensor:
        """
        Match bandit SAC critic: keep P*A slots per player, zero self slot.
        Input: action [B, P]
        Output: [B, P, P*A]
        """
        B, P = action.shape
        A = self._num_actions
        action_onehot = self.action_onehot(action).to(dtype=torch.float32)  # [B,P,A]
        stacked = action_onehot.unsqueeze(1).expand(B, P, P, A)  # [B,P,P,A]
        player_eye = (1 - torch.eye(P, dtype=action_onehot.dtype, device=action_onehot.device))
        masked = stacked * player_eye.unsqueeze(0).unsqueeze(-1)
        return masked.reshape(B, P, P * A)

    def _encode_location_history(
        self,
        hist: torch.Tensor,
        timestep: torch.Tensor,
        device: torch.device,
        loc_lstm: nn.LSTM,
    ) -> torch.Tensor:
        """
        hist:     [B, P, T, 2]
        timestep: [B]
        returns:  [B, E]
        """
        B, P, T, _ = hist.shape
        seq = hist.permute(0, 2, 1, 3).reshape(B, T, P * 2).float()
        output, _ = loc_lstm(seq)                                    # [B, T, E]
        idx = timestep.long().clamp(max=T - 1)
        return output[torch.arange(B, device=device), idx]          # [B, E]

    def forward(
        self,
        states: torch.Tensor,
        location_history: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        player: Optional[torch.Tensor] = None,
        others_actions: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        del player
        B, P, G, _ = states.shape
        device = states.device

        states_f = states.float()  # cast once

        # Shared: action features and player IDs (identical for both critics)
        act = action if action is not None else others_actions
        if act is not None:
            others_feat = self.action_masking(act).to(device=device, dtype=states_f.dtype)
        else:
            others_feat = torch.zeros(
                (B, P, self._joint_action_feat_dim), device=device, dtype=states_f.dtype
            )
        player_ids = self.player_id_encode.unsqueeze(0).expand(B, -1, -1)  # [B, P, P]

        # Per-critic: fully independent CNN + LSTM + fuse + output (mirrors bandit CriticLSTM)
        q_values: list[torch.Tensor] = []
        for cnn, loc_lstm, fuse_layer, output_layer in zip(
            self._cnns, self._loc_lstms, self._fuse_layers, self.critics_output
        ):
            z_grid = F.gelu(cnn(states_f)).unsqueeze(1).expand(B, P, self._embed_dim)  # [B, P, E]
            z_loc = self._encode_location_history(
                location_history, timestep, device, loc_lstm
            ).unsqueeze(1).expand(B, P, self._embed_dim)                               # [B, P, E]

            fused_input = torch.cat([z_grid, z_loc, player_ids, others_feat], dim=-1)
            q_values.append(output_layer(fuse_layer(fused_input)))                     # [B, P, output_dim]

        return q_values