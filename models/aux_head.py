"""Auxiliary segmentation head (used during training only)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AuxHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
