from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from icmape.models.robot_net.utils import BaseLSTM, CNN50Encoder, LSTMParameters


class ActorLSTMTD3(BaseLSTM):
    """Per-player CNN + shared location-history LSTM actor for robot TD3.

    Mirrors ``ActorLSTM`` in architecture but forward returns
    ``(gumbel_actions, policy_probs, logits)`` -- matching the bandit
    ``actor_lstm_ddpg.ActorLSTM`` interface so TD3 training can use
    Gumbel-Softmax (``hard=True``) for a differentiable actor gradient.

    Args:
        state_dim:       legacy; not used by the CNN-LSTM path.
        num_actions:     base action count (without stop).
        num_players:     P.
        horizon:         T (used only as a size hint).
        config:          ``LSTMParameters``; must expose ``gumbel_temperature``
                         and ``gumbel_tau_min`` (both float).
        grid_size:       spatial side length of belief maps.
        shared_critic:   unused; kept for API parity.
        belief_channels: channels per player belief map (usually 1).
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters,
        grid_size: int = 50,
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

        # Gumbel temperature -- can be updated externally for annealing.
        self._gumbel_tau: float = float(getattr(config, "gumbel_temperature", 1.0))

        # One CNN per player (processes that player's own belief map).
        self._cnn = nn.ModuleList(
            [
                CNN50Encoder(
                    in_channels=self._belief_channels,
                    out_dim=self._embed_dim,
                )
                for _ in range(self._num_players)
            ]
        )

        # One location LSTM per player (reads shared P*2 location history).
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

        # One fusion head per player.
        self._fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._embed_dim * 2, self._embed_dim),
                    nn.GELU(),
                )
                for _ in range(self._num_players)
            ]
        )

        # One output head per player -> num_actions + 1 (includes stop action).
        self._output = nn.ModuleList(
            [
                nn.Linear(self._embed_dim, num_actions + 1)
                for _ in range(self._num_players)
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_location_history(
        self,
        hist: torch.Tensor,
        timestep: torch.Tensor,
        loc_lstm: nn.LSTM,
        device: torch.device,
    ) -> torch.Tensor:
        """
        hist:     [B, P, T, 2]
        timestep: [B]
        returns:  [B, E]
        """
        B, P, T, _ = hist.shape
        seq = hist.permute(0, 2, 1, 3).reshape(B, T, P * 2).float()
        output, _ = loc_lstm(seq)  # [B, T, E]
        idx = timestep.long().clamp(max=T - 1)
        return output[torch.arange(B, device=device), idx]  # [B, E]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        location_history: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            states:           [B, P, G, G]  per-player belief maps.
            timestep:         [B]            current step indices.
            location_history: [B, P, T, 2]  position history.
            hard:             if True, return hard (one-hot) Gumbel samples --
                              use during actor update for straight-through gradient.

        Returns:
            gumbel_actions: [B, P, A+1]  Gumbel-Softmax samples (differentiable).
            policy_probs:   [B, P, A+1]  softmax probabilities.
            logits:         [B, P, A+1]  raw pre-softmax logits.
        """
        device = states.device
        logits_list = []

        for i, (cnn, loc_lstm, fuse, output) in enumerate(
            zip(self._cnn, self._loc_lstm, self._fuse, self._output)
        ):
            x_i = states[:, i : i + 1, :, :].float()  # [B,1,G,G]
            z_grid = F.gelu(cnn(x_i))                  # [B,E]

            z_loc = self._encode_location_history(
                location_history, timestep, loc_lstm, device
            )                                           # [B,E]

            fused = fuse(torch.cat([z_grid, z_loc], dim=-1))  # [B,E]
            logits_i = output(fused)                           # [B,A+1]
            logits_list.append(logits_i)

        logits = torch.stack(logits_list, dim=1)  # [B,P,A+1]
        policy_probs = F.softmax(logits, dim=-1)  # [B,P,A+1]

        # Gumbel-Softmax: flatten P into batch for gumbel_softmax, then reshape.
        B, P, A1 = logits.shape
        gumbel_actions = F.gumbel_softmax(
            logits.reshape(B * P, A1),
            tau=self._gumbel_tau,
            hard=hard,
            dim=-1,
        ).reshape(B, P, A1)  # [B,P,A+1]

        return gumbel_actions, policy_probs, logits
