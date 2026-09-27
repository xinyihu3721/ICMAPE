from __future__ import annotations
import torch
from torch import nn
from typing import Optional, Tuple


class LSTMModel(nn.Module):
    def __init__(self,  emb_size: int, lstm_hid_size: int, lstm_layers: int, attn_pool: str = "conditioned", query_dim: int = 5):
        super().__init__()
        self.activation = nn.SiLU()
        self.lstm = nn.LSTM(emb_size, lstm_hid_size, num_layers=lstm_layers, batch_first=True)
        self.c_0 = nn.Parameter(torch.randn(lstm_layers, 1, lstm_hid_size))
        self.h_0 = nn.Parameter(torch.randn(lstm_layers, 1, lstm_hid_size))
        if attn_pool == "conditioned":
            self.attn_pool = ConditionedPool(embed_dim=lstm_hid_size)
        elif attn_pool == "agent":
            self.attn_pool = ConditionedPoolAgent(embed_dim=lstm_hid_size,query_dim=query_dim)
        else:
            raise ValueError(f"Invalid attn_pool: {attn_pool}")
        # self.final_layer = nn.Linear(lstm_hid_size, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # The input has form (batch_size, seg_size, context_size)
        batch_size = x.shape[0]
        seg_size = x.shape[1]
        output, _ = self.lstm.forward(x, (self.h_0.repeat([1,batch_size,1]), self.c_0.repeat([1,batch_size,1])))
        output = self.activation(output)
        return output
        # if return_pooled:
        #     return self.attn_pool(output, time_dim=1, mask=mask)
        # return output

    def init_state(self, batch_size: int, device: torch.device):
        # h_0, c_0 are [L,1,H] parameters -> expand to [L,B,H]
        h0 = self.h_0.to(device).repeat(1, batch_size, 1)
        c0 = self.c_0.to(device).repeat(1, batch_size, 1)
        return h0, c0




    def step(
        self,
        x_t: torch.Tensor,                                  # [B,E]
        feat_hist: torch.Tensor,                            # [B,T,H]
        timesteps: torch.Tensor,                            # [B] 0-based
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (h,c) each [L,B,H]
    ):

       
        B = x_t.size(0)
        device = x_t.device
     

        if state is None:
            h0, c0 = self.init_state(batch_size=B, device=device)   # [L,B,H]
        elif timesteps.max() == 1:
            h0, c0 = self.init_state(batch_size=B, device=device) 
        else:
            h0, c0 = state


        out, (h_n, c_n) = self.lstm(x_t.unsqueeze(1), (h0, c0))     # out [B,1,H]
        
        feat = self.activation(out[:, 0, :])                        # [B,H]

    
        T = feat_hist.size(1)
        t_write = (timesteps.long() - 1).clamp(0, T - 1)

        feat_hist = feat_hist.clone()
        
        feat = feat.to(dtype=feat_hist.dtype)
        
        
        batch_indices = torch.arange(B, device=device, dtype=torch.long)
        feat_hist[batch_indices, t_write, :] = feat

        return feat_hist, (h_n, c_n)



    


class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)
        self.proj = nn.Linear(d * 2, d)
        self.ln = nn.LayerNorm(d)
        self.activation = nn.SiLU()
    def forward(self, h: torch.Tensor, time_dim: int = 1, mask=None) -> torch.Tensor:
        # h: [B,T,d]
        s = self.w.forward(h).squeeze(-1)   # [B, T]
        if mask is not None: s = s.masked_fill(~mask, float("-inf"))
        # #assert mask is not None
        # import pdb; pdb.set_trace()
        # s = s*mask
        a = torch.softmax(s, dim=time_dim).unsqueeze(-1)
        v = (a * self.activation(h)).sum(dim=time_dim)

       
        return v



class Gate(nn.Module):
    def __init__(self, d_m: int, d_v: int):
        super().__init__()
        
        self.proj = nn.Linear(d_m, d_v, bias=False)
        self.activation = nn.SiLU()

    def forward(self, v: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor: 
        return v *  self.activation(self.proj(modulation))
    

class ConditionedPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5
        self.query_in = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.query_out = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.gate = Gate(embed_dim, embed_dim)


    def forward(self, h: torch.Tensor , mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # h: [B,T,d_h], mask: [B,T] True for valid
        query_out = self.query_out.expand(h.shape[0], -1, -1)
        query_in = self.query_in.expand(h.shape[0], -1, -1)
        h = h
        K = self.Wk(h)                                  # [B,T,d_k]
        V = h                                           # [B,T,d_v]
        scores = (query_in * K).sum(-1) / self.scale    # [B,T]
        tau = self.log_tau.clamp(min=-3, max=3).exp() 
        scores = scores / tau
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            # if any row is all -inf, softmax will return nan, so set them to zero
            all_inf_mask = (mask.sum(dim=1) == 0)
            scores = scores.masked_fill(all_inf_mask.unsqueeze(1), 0.0)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)     # [B,T,1]
        c = (w * V).sum(dim=1)                              # [B,d_v]
        return self.gate(c, query_out.squeeze(1))           # [B,d_v]





class ConditionedPoolAgent(nn.Module):
    def __init__(self, embed_dim: int, query_dim: int):
        super().__init__()
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq = nn.Linear(query_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5
        self.query_in = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.query_out = nn.Parameter(torch.zeros(1, 1, embed_dim))
        

        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.gate = Gate(embed_dim, embed_dim)


    def forward(self, h: torch.Tensor , agent_query:torch.Tensor=None, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # h: [B,T,d_h], mask: [B,T] True for valid
        
        query_out = self.query_out.expand(h.shape[0], -1, -1)
        query_in = self.query_in.expand(h.shape[0], -1, -1)
        # query_in = self.Wq(agent_query.unsqueeze(1))
        h = h
        K = self.Wk(h)                                  # [B,T,d_k]
        V = h                                           # [B,T,d_v]
        scores = (query_in * K).sum(-1) / self.scale    # [B,T]
        tau = self.log_tau.clamp(min=-3, max=3).exp() 
        scores = scores / tau
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            # if any row is all -inf, softmax will return nan, so set them to zero
            all_inf_mask = (mask.sum(dim=1) == 0)
            scores = scores.masked_fill(all_inf_mask.unsqueeze(1), 0.0)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)     # [B,T,1]
        c = (w * V).sum(dim=1)                              # [B,d_v]
        return self.gate(c, query_out.squeeze(1))           # [B,d_v]




# class ConditionedPoolperAgent(nn.Module):
#     def __init__(self, embed_dim: int, num_players: int):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_players = num_players
#         self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.scale = embed_dim ** 0.5
        
#         # Learnable queries for each player
#         self.query_in = nn.Parameter(torch.zeros(1, num_players, embed_dim))
#         self.query_out = nn.Parameter(torch.zeros(1, num_players, embed_dim))
#         self.log_tau = nn.Parameter(torch.tensor(0.0))
#         self.gate = Gate(embed_dim, embed_dim)

#     def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # h shape: [B*P, T, E] -> Reshape to [B, P, T, E]
        
#         B_times_P, T, E = h.shape
#         B = B_times_P // self.num_players
#         h = h.view(B, self.num_players, T, E)
        
#         # Project keys: [B, P, T, E]
#         K = self.Wk(h) 
#         V = h
        
#         # Expand queries to match batch size: [B, P, E]
#         q_in = self.query_in.expand(B, -1, -1)
#         q_out = self.query_out.expand(B, -1, -1)
#         tau = self.log_tau.clamp(min=-3, max=3).exp()
        
#         player_outputs = []
#         for p in range(self.num_players):
#             # Attention scores for player p: [B, T]
#             # query: [B, E], keys: [B, T, E] -> dot product
#             scores = torch.bmm(K[:, p, :, :], q_in[:, p, :].unsqueeze(-1)).squeeze(-1) / self.scale
#             scores = scores / tau
#             if mask is not None:
#                 # mask: [B, P, T]
#                 scores = scores.masked_fill(~mask, float("-inf"))
#                 # Prevent NaNs if a player has no valid timesteps
#                 all_inf_mask = (mask.sum(dim=1) == 0)
#                 scores = scores.masked_fill(all_inf_mask.unsqueeze(1), 0.0)
            
#             w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
#             c = (w * V[:, p, :, :]).sum(dim=1)               # [B, E]
            
#             # Gating with player-specific output query
#             out = self.gate(c, q_out[:, p, :])               # [B, E]
            
#             player_outputs.append(out)

#         # Stack to [B, P, E] torch
#         return torch.stack(player_outputs, dim=1)
