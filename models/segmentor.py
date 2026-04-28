"""DINOv3 backbone + PSPNet (PPM) segmentation model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_head import AuxHead
from .backbone import DINOv3Backbone
from .ppm import PPM


class DINOv3PSPNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        backbone_name: str = "dinov3_vits16",
        backbone_weights: Optional[str] = None,
        embed_dim: int = 384,
        aux_layer_idx: int = 6,
        freeze_backbone: bool = True,
        ppm_pool_sizes=(1, 2, 3, 6),
        ppm_reduction: int = 96,
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        aux_hidden: int = 256,
        aux_dropout: float = 0.1,
        use_aux: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux

        self.backbone = DINOv3Backbone(
            model_name=backbone_name,
            weights_path=backbone_weights,
            aux_layer_idx=aux_layer_idx,
            freeze=freeze_backbone,
        )

        self.ppm = PPM(
            in_channels=embed_dim,
            pool_sizes=ppm_pool_sizes,
            reduction_channels=ppm_reduction,
        )

        self.head = nn.Sequential(
            nn.Conv2d(self.ppm.out_channels, head_hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_hidden, num_classes, 1),
        )

        if self.use_aux:
            self.aux_head = AuxHead(
                in_channels=embed_dim,
                num_classes=num_classes,
                hidden_channels=aux_hidden,
                dropout=aux_dropout,
            )
        else:
            self.aux_head = None

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        return_aux = self.training and self.use_aux

        if return_aux:
            feat, feat_aux = self.backbone(x, return_aux=True)
        else:
            feat = self.backbone(x, return_aux=False)
            feat_aux = None

        ppm_out = self.ppm(feat)
        logits = self.head(ppm_out)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)

        if return_aux and feat_aux is not None:
            aux_logits = self.aux_head(feat_aux)
            aux_logits = F.interpolate(aux_logits, size=(H, W), mode="bilinear", align_corners=True)
            return logits, aux_logits

        return logits
