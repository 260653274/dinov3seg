"""Pyramid Pooling Module from PSPNet (Zhao et al., 2017)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(
        self,
        in_channels: int = 384,
        pool_sizes: Sequence[int] = (1, 2, 3, 6),
        reduction_channels: int = 96,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.reduction_channels = reduction_channels
        self.pool_sizes = tuple(pool_sizes)

        self.stages = nn.ModuleList()
        for size in self.pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, reduction_channels, 1, bias=False),
                    nn.BatchNorm2d(reduction_channels),
                    nn.ReLU(inplace=True),
                )
            )

    @property
    def out_channels(self) -> int:
        return self.in_channels + self.reduction_channels * len(self.pool_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pyramids = [x]
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=True)
            pyramids.append(feat)
        return torch.cat(pyramids, dim=1)
