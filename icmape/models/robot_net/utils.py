from __future__ import annotations

import torch
from torch import nn

from icmape.models.sequence_net.utils import BaseLSTM, LSTMParameters, to_lstm_parameters


class CNN50Encoder(nn.Module):
    """Small CNN encoder for 50x50 single-channel images."""

    def __init__(self, in_channels: int = 1, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 50 -> 25
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 25 -> 13
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 13 -> 7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 50, 50] or [B, C, 50, 50]
        Returns:
            [B, out_dim]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected 3D/4D input, got shape {tuple(x.shape)}")
        feat = self.features(x.float()).flatten(start_dim=1)
        return self.proj(feat)


class BaseCNN(nn.Module):
    """CNN base with target-network update helpers (hard/soft)."""

    def __init__(
        self,
        grid_size: int = 15,
        hidden_channels: int = 32,
        in_channels: int = 1,
    ):
        super().__init__()
        self._grid_size = int(grid_size)
        self._hidden_channels = int(hidden_channels)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 50 -> 25
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 25 -> 13
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 13 -> 7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def hard_update(self, source: nn.Module) -> None:
        """Hard update: copy parameters from `source` into this module."""
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: nn.Module, tau: float) -> None:
        """Soft update (Polyak): target = tau*source + (1-tau)*target."""
        self.polyak_update(source, tau)

    def polyak_update(self, source: nn.Module, tau: float) -> None:
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    
    



__all__ = [
    "BaseLSTM",
    "LSTMParameters",
    "to_lstm_parameters",
    "CNN50Encoder",
]
