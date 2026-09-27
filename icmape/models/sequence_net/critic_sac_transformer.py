from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union
from dataclasses import dataclass
from transformers import GPT2Model
from icmape.models.sequence_net.utils import BaseTransformer, TransformerParameters, create_gpt2_config


class CriticTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        num_actions: int,
        num_players: int, 
        horizon: int,
        config: TransformerParameters,
        bins: int = 1,
        ent_coef: Union[str, float] = 1,#"auto",
        target_entropy: Union[str, float] = 1,#"auto",
        target_update_interval: int = 1,
        n_critic: int = 2,
    ):
        super().__init__()
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        
        self._num_actions = num_actions
        self._num_players = num_players
        self._cfg = config
        self.bins = bins
        self._state_dim = state_dim 
        self._action_dim = num_actions #* num_players

        self._gpt2_config = create_gpt2_config(horizon, config)
        self._gpt2 = GPT2Model(self._gpt2_config)

        self._embed = nn.Linear(state_dim, config.embed_dim)

        self._output_dim = output_dim  

        transformer_output_dim = config.embed_dim  # Only state, no action input needed
        hidden_dim = transformer_output_dim // 2
        self.critics_layer1 = nn.ModuleList([
            nn.Linear(transformer_output_dim, hidden_dim) for _ in range(n_critic)
        ])
        self.critics_layer2 = nn.ModuleList([
            nn.Linear(hidden_dim, self._output_dim) for _ in range(n_critic)  # Output num_actions Q-values
        ])
        # Add dropout layers for each critic
        self.critics_dropout = nn.ModuleList([
            nn.Dropout(config.resid_dropout) for _ in range(n_critic)
        ])

        self.activation = nn.ReLU()
       
    
 

    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None, 
        player: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            states: [B, H, state_dim] input states (state sequence)
            action: Optional, not used (kept for compatibility)
            player: Optional, not used (kept for compatibility)
        
        Returns:
            qvalues: List of [B, num_actions] Q-values, one for each critic network
        """
        
        batch_size = states.shape[0]
        h = F.gelu(self._embed(states))
        

        h = self._gpt2(inputs_embeds=h)
    
        h = h.last_hidden_state

        h = h[:, -1, :]  # [B, embed_dim]

        # No need to concatenate action - output Q-values for all actions
        # state_action_cat = torch.cat([h, action], dim=-1)
        q_values = []
 
        for l1, l2, dropout in zip(self.critics_layer1, self.critics_layer2, self.critics_dropout):
            q = self.activation(l1(h))  # Use only state embedding
            q = dropout(q)  
            q = l2(q)  
            # q_values.append(q)
            # q_reshaped = q.contiguous().view(batch_size * self._num_players, self._num_actions)
            q_values.append(q)

        # Return list of Q-values from each critic (for taking min across critics)
        # Each element in list is [B, num_actions]
        return q_values
