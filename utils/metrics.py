"""Streaming confusion-matrix-based segmentation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class SegMeter:
    """Accumulates a confusion matrix across batches and reports IoU / mIoU / pAcc."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confmat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred: (B, H, W) int64 class indices.  target: (B, H, W) int64."""
        pred = pred.detach().cpu().numpy().reshape(-1)
        target = target.detach().cpu().numpy().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]
        idx = target * self.num_classes + pred
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confmat += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm = self.confmat.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
            acc = np.where(cm.sum(axis=1) > 0, tp / cm.sum(axis=1), np.nan)
        miou = float(np.nanmean(iou))
        macc = float(np.nanmean(acc))
        pacc = float(tp.sum() / max(cm.sum(), 1.0))
        return {
            "miou": miou,
            "macc": macc,
            "pacc": pacc,
            "iou_per_class": iou.tolist(),
            "acc_per_class": acc.tolist(),
        }
