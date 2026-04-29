"""Multi-Scale Feature Alignment (MSFA) adapter — Strategy B.

Fuses features from K transformer blocks of a frozen ViT backbone:

    block i_1  ──┐                ┌── 1×1 conv (Cin → c)  ┐
    block i_2  ──┤  (B, Cin, h, w)│                       ├──► concat (B, K·c, h, w)
       ...     ──┤                │                       │
    block i_K  ──┘                └── 1×1 conv (Cin → c)  ┘
                                                                 │
                                                                 ▼
                                                       optional 2× upsample
                                                                 │
                                                                 ▼
                                                  3×3 conv → BN → ReLU
                                                                 │
                                                                 ▼
                                                          (B, out_channels, h', w')

Captures early/mid/late ViT semantics (cf. DPT, ViT-Adapter), giving the
downstream PPM richer multi-scale context than a single final-block feature.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int = 384,
        num_layers: int = 4,
        per_layer_channels: int = 96,
        out_channels: int = 384,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.per_layer_channels = per_layer_channels
        self.upsample = upsample
        self.fused_channels = num_layers * per_layer_channels

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, per_layer_channels, 1, bias=False),
                nn.BatchNorm2d(per_layer_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_layers)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(self.fused_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} feature maps, got {len(feats)}"
            )
        adapted: List[torch.Tensor] = [a(f) for a, f in zip(self.adapters, feats)]
        x = torch.cat(adapted, dim=1)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.fuse(x)
