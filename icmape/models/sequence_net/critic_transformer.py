# from __future__ import annotations
# import torch
# from torch import nn
# import torch.nn.functional as F
# from dataclasses import dataclass
# from transformers import GPT2Model
# from icmape.models.sequence_net.utils import BaseTransformer, TransformerParameters, create_gpt2_config


# class CriticTransformer(BaseTransformer):
#     def __init__(
#         self,
#         state_dim: int,
#         num_actions: int,
#         num_players: int, 
#         horizon: int,
#         config: TransformerParameters,
#         bins: int = 1,
#         ent_coef: Union[str, float] = 1,#"auto",
#         target_entropy: Union[str, float] = 1,#"auto",
#         target_update_interval: int = 1,
#         n_critic: int = 2,
#     ):
#         super().__init__()
#         self.target_entropy = target_entropy
#         self.log_ent_coef = None  # type: Optional[th.Tensor]
#         self.ent_coef = ent_coef
#         self.target_update_interval = target_update_interval
        
#         self._num_actions = num_actions
#         self._num_players = num_players
#         self._cfg = config
#         self.bins = bins
#         self._state_dim = state_dim 
#         self._action_dim = num_actions #* num_players

#         self._gpt2_config = create_gpt2_config(horizon, config)
#         self._gpt2 = GPT2Model(self._gpt2_config)

#         self._embed = nn.Linear(state_dim, config.embed_dim)
#         self._output_dim = 1


#         # self._output = nn.Linear(config.embed_dim, num_actions)
#         # self._player_heads = nn.ModuleList(
#         #     [nn.Linear(config.embed_dim, 1) for _ in range(num_players)]
#         input_dim = config.embed_dim + self._action_dim
#         hidden_dim = input_dim // 2
#         self.critics_layer1 = nn.ModuleList([
#             nn.Linear(input_dim, hidden_dim) for _ in range(n_critic)
#         ])
#         self.critics_layer2 = nn.ModuleList([
#             nn.Linear(hidden_dim, self._output_dim) for _ in range(n_critic)
#         ])

#         self.activation = nn.ReLU()
       
    
 

#     def forward(
#         self,
#         states: torch.Tensor,
#         action: torch.Tensor,
#         player: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """Forward pass

#         Args:
#             states: [B, H, state_dim] input states (state-action pairs: state sequence)
#             action: [B, P, num_actions] input one hot encoded actions for all players
#             player: 1D torch.Tensor of player indices, e.g., torch.tensor([0]) or torch.tensor([0, 1, 2])
        
#         Returns:
#             qvalues: [B, len(players)] same Q-values for each player
#         """
        
        
#         h = F.gelu(self._embed(states))
        

#         h = self._gpt2(inputs_embeds=h)
    
#         h = h.last_hidden_state

#         h = h[:, -1, :]  # [B, embed_dim]

#         state_action_cat = torch.cat([h, action], dim=-1)
#         q_values = []
 
#         for l1, l2 in zip(self.critics_layer1, self.critics_layer2):
#             q = self.activation(l1(state_action_cat ))
#             q = l2(q).squeeze(-1) 

#             #q = q.expand(-1, self._num_players)
#             q_values.append(q)

       
        

#         return q_values


from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
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
    ):
        super().__init__()
        self._num_actions = num_actions
        self._num_players = num_players
        self._cfg = config
        self.bins = bins
        self._state_dim = state_dim

        self._gpt2_config = create_gpt2_config(horizon, config)
        self._gpt2 = GPT2Model(self._gpt2_config)

        self._embed = nn.Linear(self._state_dim, config.embed_dim)
        self._output = nn.Linear(config.embed_dim, output_dim)

    def forward(
        self,
        states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            states: [B, H, state_dim] input states (sequence of state-action pairs)
            timestep: Optional [B] or [B*P] timestep (kept for API compatibility).

        Returns:
            values: [B, output_dim] value estimates (e.g. Q-values per action)
        """
        h = F.gelu(self._embed(states))
        h = self._gpt2(inputs_embeds=h)
        h = h.last_hidden_state
        h = h[:, -1, :]  # [B, embed_dim]
        out = self._output(h)
        return out


# from __future__ import annotations
# import torch
# from torch import nn
# import torch.nn.functional as F
# from dataclasses import dataclass
# from transformers import GPT2Model
# from icmape.models.sequence_net.utils import BaseTransformer, TransformerParameters, create_gpt2_config


# class CriticTransformer(BaseTransformer):
#     def __init__(
#         self,
#         state_dim: int,
#         num_actions: int,
#         num_players: int, 
#         horizon: int,
#         config: TransformerParameters,
#         bins: int = 1
#     ):
#         super().__init__()
#         self._num_actions = num_actions
#         self._num_players = num_players
#         self._cfg = config
#         self.bins = bins
#         self._state_dim = state_dim 



#         self._embed_player = nn.Embedding(num_players, num_players)
#         self._embed_action = nn.Embedding(num_actions-1, num_actions-1)

#         self._gpt2_config = create_gpt2_config(horizon, config)
#         self._gpt2 = GPT2Model(self._gpt2_config)

#         self._embed = nn.Linear(self._state_dim, config.embed_dim)


#         self._output = nn.Linear(config.embed_dim, num_actions)

#     def forward(
#         self,
#         states: torch.Tensor
#     ) -> torch.Tensor:
#         """Forward pass

#         Args:
#             actions: [B, T, action_size] one-hot encoded actions
#             rewards: [B, T] rewards

#         Returns:
#             values: [B, T, action_size * bins] value estimates
# """
        
#         player_id = states[...,0].long()
#         action = states[...,1].long()
#         reward = states[...,2].unsqueeze(-1)

#         player_embed = self._embed_player(player_id)
#         action_embed = self._embed_action(action)
        
#         states_embed = torch.cat([player_embed, action_embed, reward], dim=-1)
    

#         h = F.gelu(self._embed(states_embed))

#         h = self._gpt2(inputs_embeds=h)
#         h = h.last_hidden_state

#         h = h[:, -1, :]

#         out = self._output(h)

#         return out