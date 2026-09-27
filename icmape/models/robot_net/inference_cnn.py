from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from icmape.models.robot_net.utils import BaseCNN


class InferenceCNN(BaseCNN):
    """
    Predicts a single target location on a grid.

    Input:
        grid: [B, H, W]
           or [B, 1, H, W]

    Output:
        logits: [B, H, W]
        Raw logits over spatial locations.
    """

    def __init__(
        self,
        grid_size: int = 50,
        hidden_channels: int = 32,
        in_channels: int = 3,
    ):
        super().__init__(grid_size=grid_size, hidden_channels=hidden_channels, in_channels=3)
        self._grid_size = int(grid_size)
        ch = int(hidden_channels)

        self._net = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward compatible with callers that pass an extra timestep arg.

        Returns probabilities in [0,1] (for `binary_cross_entropy`).
        """
    
   
        logits = self._net(x).squeeze(1)
        return logits

    @torch.no_grad()
    def predict_proba(self, grid: torch.Tensor) -> torch.Tensor:
        logits = self.forward(grid)   # [B,H,W]
        B, H, W = logits.shape
        probs = F.softmax(logits.view(B, -1), dim=-1).view(B, H, W)
        return probs

    @torch.no_grad()
    def predict_index(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            row: [B]
            col: [B]
        """
        logits = self.forward(grid)                     # [B,H,W]
        B, H, W = logits.shape
        idx = logits.view(B, -1).argmax(dim=-1)
        row = idx // W
        col = idx % W
        return row, col