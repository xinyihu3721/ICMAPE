from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from icmape.models.sequence_net.utils import BaseLSTM, LSTMParameters #, create_lstm_config
from icmape.models.sequence_net.lstm_utils import LSTMModel



class CriticLSTM(BaseLSTM):
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters,
        bins: int = 1,
        ent_coef: Union[str, float] = 1,
        target_entropy: Union[str, float] = 1,
        target_update_interval: int = 1,
        n_critic: int = 2,
    ):
        super().__init__()
        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self._horizon = horizon
        self._num_actions = num_actions
        self._num_players = num_players
        self._cfg = config
        self.bins = bins
        self._state_dim = state_dim
        self._action_dim = num_actions

        self._embed = nn.Linear(state_dim, config.embed_dim)
        player_mask = (1 - torch.eye(self._num_players, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("_player_eye", player_mask, persistent=False)

        self._lstm_models = nn.ModuleList(
            [
                LSTMModel(
                    emb_size=config.embed_dim,
                    lstm_hid_size=config.embed_dim,
                    lstm_layers=config.num_layers,
                )
                for _ in range(n_critic)
            ]
        )

        self._output_dim = output_dim

        lstm_output_dim = config.embed_dim
        action_feat_dim = self._num_players * self._num_actions
        critic_in_dim = lstm_output_dim + action_feat_dim
        mlp_hidden = config.embed_dim
        self.critics_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(critic_in_dim, mlp_hidden),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden, self._output_dim),
                )
                for _ in range(n_critic)
            ]
        )

    def action_onehot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.long(), num_classes=self._num_actions)

    def action_masking(self, action: torch.Tensor) -> torch.Tensor:
        """Same as critic_td3_lstm.CriticLSTM.action_masking."""
        B, P = action.shape
        A = self._num_actions
        action_onehot = self.action_onehot(action)
        stacked = action_onehot.unsqueeze(1).expand(B, P, P, A)
        mask = self._player_eye.unsqueeze(-1)
        masked_action = stacked * mask
        return masked_action.reshape(B * P, P * A)

    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        player: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = states.shape[0]
        seq_len = states.shape[1]
        x = F.gelu(self._embed(states))

        timestep_clamped = timestep.clamp(min=1, max=seq_len)
        mask = torch.arange(seq_len, device=states.device).unsqueeze(0).expand(batch_size, seq_len)
        mask = mask < timestep_clamped.unsqueeze(1)

        q_values = []
        if action is None:
            action_masked = torch.zeros(
                batch_size,
                self._num_players * self._num_actions,
                device=states.device,
                dtype=x.dtype,
            )
        else:
            action_masked = self.action_masking(action)

        for _lstm, head in zip(self._lstm_models, self.critics_head):
            z = _lstm(x)
            z_pooled = _lstm.attn_pool(z, mask=mask)
            z_concat = torch.cat([z_pooled, action_masked], dim=-1)
            q_values.append(head(z_concat))

        return q_values




