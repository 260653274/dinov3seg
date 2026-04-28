"""Quick CPU/GPU forward-pass sanity check (no real DINOv3 weights needed).

Mocks the backbone with a random projection so you can verify the rest of the
pipeline (PPM + head + losses + metrics) end-to-end without downloading the
DINOv3 checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ppm import PPM  # noqa: E402
from models.aux_head import AuxHead  # noqa: E402
from utils.losses import CEAuxLoss  # noqa: E402
from utils.metrics import SegMeter  # noqa: E402


class FakeBackbone(nn.Module):
    def __init__(self, embed_dim=384, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.aux_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, return_aux=False):
        f = self.proj(x)
        if return_aux:
            return f, self.aux_proj(x)
        return f


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    B, num_classes = 2, 21
    x = torch.randn(B, 3, 512, 512, device=device)
    target = torch.randint(0, num_classes, (B, 512, 512), device=device).long()

    backbone = FakeBackbone(embed_dim=384, patch_size=16).to(device)
    ppm = PPM(in_channels=384, pool_sizes=(1, 2, 3, 6), reduction_channels=96).to(device)
    head = nn.Sequential(
        nn.Conv2d(ppm.out_channels, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(256, num_classes, 1),
    ).to(device)
    aux_head = AuxHead(in_channels=384, num_classes=num_classes).to(device)
    criterion = CEAuxLoss(ignore_index=255, aux_weight=0.4).to(device)

    feat, feat_aux = backbone(x, return_aux=True)
    print(f"[smoke] feat={tuple(feat.shape)}  feat_aux={tuple(feat_aux.shape)}")

    pooled = ppm(feat)
    print(f"[smoke] ppm out shape = {tuple(pooled.shape)}  (expect channels={ppm.out_channels})")

    logits = head(pooled)
    print(f"[smoke] logits shape = {tuple(logits.shape)}")

    logits_full = torch.nn.functional.interpolate(
        logits, size=x.shape[-2:], mode="bilinear", align_corners=True
    )
    aux_logits = torch.nn.functional.interpolate(
        aux_head(feat_aux), size=x.shape[-2:], mode="bilinear", align_corners=True
    )
    loss, parts = criterion(logits_full, target, aux_logits)
    loss.backward()
    print(f"[smoke] loss={loss.item():.4f}  main={parts['loss_main'].item():.4f}  "
          f"aux={parts['loss_aux'].item():.4f}")

    pred = logits_full.argmax(dim=1)
    meter = SegMeter(num_classes=num_classes)
    meter.update(pred, target)
    m = meter.compute()
    print(f"[smoke] dummy mIoU={m['miou']*100:.2f} pAcc={m['pacc']*100:.2f}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
