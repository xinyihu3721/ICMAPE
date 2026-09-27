from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional
from icmape.models.sequence_net.utils import BaseLSTM, LSTMParameters
from icmape.models.sequence_net.lstm_utils import LSTMModel

class ActorLSTM(BaseLSTM):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        horizon: int,
        config: LSTMParameters,
        shared_critic: bool
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_players = num_players
        self._state_dim = state_dim
        self._horizon = horizon
        self._cfg = config
        self.shared_critic = shared_critic
        self._gumbel_tau = config.gumbel_temperature
        # self._lstm_config = create_lstm_config(config)
        # self._lstm = nn.LSTM(**self._lstm_config)  # Old LSTM implementation
        self._lstm = LSTMModel(
            emb_size=config.embed_dim,
            lstm_hid_size=config.embed_dim,
            lstm_layers=config.num_layers
        )

        self._embed = nn.Linear(state_dim, config.embed_dim)
  
        output_size = num_actions + 1
        self._output = nn.Linear(config.embed_dim, output_size)
        


    def forward(self, states: torch.Tensor, timestep: Optional[torch.Tensor] = None,hard: bool = False):
        h = F.gelu(self._embed(states))          # [B,T,E]
        B, T = states.shape[0], states.shape[1]

        if timestep is None:
            mask = torch.ones(B, T, device=states.device, dtype=torch.bool)
        else:
            timestep_clamped = timestep.long().clamp(min=1, max=T)  # [B]
            ar = torch.arange(T, device=states.device).unsqueeze(0) # [1,T]
            mask = ar < timestep_clamped.unsqueeze(1)               # [B,T]

        z = self._lstm(h)                        # [B,T,E] (your LSTMModel returns activated outputs)
        z_pooled = self._lstm.attn_pool(z, mask=mask)  # [B,E]
        logits = self._output(z_pooled)          # [B,A+1]
        policy_probs = F.softmax(logits, dim=-1)
        gumbel_actions = F.gumbel_softmax(
            logits, tau=self._gumbel_tau,hard=hard,dim=-1
        )
        #log_pi = (policy_probs.clamp_min(1e-8)).log()
        return gumbel_actions, policy_probs, logits


    def step(
        self,
        x_t: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        feat_hist: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_emb = F.gelu(self._embed(x_t))
        feat_hist, (h_n, c_n) = self._lstm.step(
            x_t=x_emb,
            state=hidden_state,
            feat_hist=feat_hist,
            timesteps=timesteps,
        )
        T = feat_hist.size(1)
        t_clamped = timesteps.long().clamp(min=1, max=T)
        ar = torch.arange(T, device=x_t.device).unsqueeze(0)
        mask = ar < t_clamped.unsqueeze(1)
        pooled = self._lstm.attn_pool(feat_hist, mask=mask)
        logits = self._output(pooled)
        policy_probs = F.gumbel_softmax(
            logits, tau=self._gumbel_tau, hard=False, dim=-1
        )
        log_pi = (policy_probs.clamp_min(1e-8)).log()
        return policy_probs, log_pi, feat_hist, (h_n, c_n)
