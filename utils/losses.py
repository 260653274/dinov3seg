"""Loss helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEAuxLoss(nn.Module):
    """Cross-entropy main loss + optional aux CE loss with weight `aux_weight`."""

    def __init__(
        self,
        ignore_index: int = 255,
        aux_weight: float = 0.4,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.empty(0),
            persistent=False,
        )

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if self.class_weights.numel() > 0 else None
        return F.cross_entropy(
            logits, target, weight=weight,
            ignore_index=self.ignore_index, reduction="mean",
        )

    def forward(self, logits, target, aux_logits: Optional[torch.Tensor] = None):
        main = self._ce(logits, target)
        if aux_logits is None or self.aux_weight == 0:
            return main, {"loss_main": main.detach(), "loss_aux": torch.zeros_like(main)}
        aux = self._ce(aux_logits, target)
        total = main + self.aux_weight * aux
        return total, {"loss_main": main.detach(), "loss_aux": aux.detach()}
