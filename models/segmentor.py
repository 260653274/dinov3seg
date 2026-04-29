"""DINOv3 backbone + PSPNet (PPM) segmentation model.

Two architectural variants share this class, switched by ``msfa_enabled``:

* **Baseline (Strategy 0):** final-block features → PPM → head.
* **Strategy B (multi-scale feature alignment):** features from a list of
  transformer blocks → :class:`FeatureAlignmentAdapter` (1×1 per-block
  conv → concat → optional 2× upsample → 3×3 fuse) → PPM → head.

In both variants the auxiliary head is supervised on the ``aux_layer_idx``-th
block's features (default block 6).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import FeatureAlignmentAdapter
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
        backbone_freeze_until_block: Optional[int] = None,
        ppm_pool_sizes=(1, 2, 3, 6),
        ppm_reduction: int = 96,
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        aux_hidden: int = 256,
        aux_dropout: float = 0.1,
        use_aux: bool = True,
        msfa_enabled: bool = False,
        msfa_layers: Sequence[int] = (3, 6, 9, 11),
        msfa_per_layer_channels: int = 96,
        msfa_out_channels: int = 384,
        msfa_upsample: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        self.aux_layer_idx = int(aux_layer_idx)

        self.backbone = DINOv3Backbone(
            model_name=backbone_name,
            weights_path=backbone_weights,
            aux_layer_idx=aux_layer_idx,
            freeze=freeze_backbone,
            freeze_until_block=backbone_freeze_until_block,
        )

        self.msfa_enabled = bool(msfa_enabled)
        if self.msfa_enabled:
            self.msfa_layers = tuple(int(i) for i in msfa_layers)
            self.msfa = FeatureAlignmentAdapter(
                in_channels=embed_dim,
                num_layers=len(self.msfa_layers),
                per_layer_channels=msfa_per_layer_channels,
                out_channels=msfa_out_channels,
                upsample=msfa_upsample,
            )
            ppm_in = msfa_out_channels
        else:
            self.msfa_layers = ()
            self.msfa = None
            ppm_in = embed_dim

        self.ppm = PPM(
            in_channels=ppm_in,
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

    def _required_layers(self) -> list:
        """Sorted, deduplicated list of transformer block indices we need."""
        depth = self.backbone.depth
        idxs = set()
        if self.msfa_enabled:
            idxs.update(self.msfa_layers)
        else:
            idxs.add(depth - 1)  # final block, post-norm
        if self.training and self.use_aux:
            idxs.add(self.aux_layer_idx)
        # always have final available for non-MSFA path; with MSFA the final layer
        # is typically already in msfa_layers, but keep this guard inexpensive.
        return sorted(idxs)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        depth = self.backbone.depth
        return_aux = self.training and self.use_aux

        layers = self._required_layers()
        feats_list = self.backbone(x, return_layers=layers)
        by_idx = dict(zip(layers, feats_list))

        if self.msfa_enabled:
            backbone_out = self.msfa([by_idx[i] for i in self.msfa_layers])
        else:
            backbone_out = by_idx[depth - 1]

        ppm_out = self.ppm(backbone_out)
        logits = self.head(ppm_out)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)

        if return_aux:
            aux_feat = by_idx[self.aux_layer_idx]
            aux_logits = self.aux_head(aux_feat)
            aux_logits = F.interpolate(aux_logits, size=(H, W), mode="bilinear", align_corners=True)
            return logits, aux_logits

        return logits
