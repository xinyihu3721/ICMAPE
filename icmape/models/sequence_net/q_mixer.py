"""
Network to combine per-agent Q values into a single Q_total (QMix-style).
Input: agent_qs [B, P], states [B, state_dim].
Output: Q_total [B, 1].
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from icmape.models.sequence_net.utils import QMixerParameters


class QMixer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        num_players: int,
        config: QMixerParameters,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = num_players
        self.embed_dim = config.hypernet_embed

        if config.hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif config.hypernet_layers == 2:
            hypernet_embed = config.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise ValueError(f"hypernet_layers must be 1 or 2, got {config.hypernet_layers}")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def hard_update(self, source: nn.Module) -> None:
        """Hard update: copy parameters from source."""
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: nn.Module, tau: float) -> None:
        """Polyak (soft) update: self = tau * source + (1 - tau) * self."""
        for self_p, src_p in zip(self.parameters(), source.parameters()):
            self_p.data.copy_(tau * src_p.data + (1.0 - tau) * self_p.data)

    def forward(self, agent_qs, states, ind_states=None): 
        bs = agent_qs.size(0)
        states = states.reshape(bs, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot