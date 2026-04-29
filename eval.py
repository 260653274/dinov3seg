"""Evaluate a trained checkpoint on Pascal VOC val set.

Supports single-scale and multi-scale + flip test-time augmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VOCSegmentation, VOC_CLASSES, build_val_transforms
from models import DINOv3PSPNet
from utils import SegMeter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/voc_config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--multi-scale", action="store_true")
    p.add_argument("--flip", action="store_true")
    p.add_argument("--scales", type=float, nargs="+", default=None)
    p.add_argument("--output", type=str, default=None,
                   help="optional path to write metrics JSON")
    return p.parse_args()


def build_model_from_ckpt(cfg, ckpt) -> DINOv3PSPNet:
    mcfg = cfg["model"]
    msfa_cfg = mcfg.get("msfa", {}) or {}
    model = DINOv3PSPNet(
        num_classes=mcfg["num_classes"],
        backbone_name=mcfg["backbone"]["name"],
        backbone_weights=mcfg["backbone"].get("weights_path"),
        embed_dim=mcfg["backbone"]["embed_dim"],
        aux_layer_idx=mcfg["backbone"].get("aux_layer_idx", 6),
        freeze_backbone=mcfg["backbone"].get("freeze", True),
        backbone_freeze_until_block=mcfg["backbone"].get("freeze_until_block"),
        ppm_pool_sizes=tuple(mcfg["ppm"]["pool_sizes"]),
        ppm_reduction=mcfg["ppm"]["reduction_channels"],
        head_hidden=mcfg["head"]["hidden_channels"],
        head_dropout=mcfg["head"]["dropout"],
        aux_hidden=mcfg["aux_head"]["hidden_channels"],
        aux_dropout=mcfg["aux_head"]["dropout"],
        use_aux=False,
        msfa_enabled=bool(msfa_cfg.get("enabled", False)),
        msfa_layers=tuple(msfa_cfg.get("layers", (3, 6, 9, 11))),
        msfa_per_layer_channels=int(msfa_cfg.get("per_layer_channels", 96)),
        msfa_out_channels=int(msfa_cfg.get("out_channels", mcfg["backbone"]["embed_dim"])),
        msfa_upsample=bool(msfa_cfg.get("upsample", False)),
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    return model


def round_to_multiple(x: int, k: int) -> int:
    return ((x + k - 1) // k) * k


@torch.no_grad()
def predict_logits(model, image: torch.Tensor, scales, flip: bool, patch_size: int = 16) -> torch.Tensor:
    """Average softmax over multi-scale (and optional flip) augmentations."""
    _, _, H, W = image.shape
    accum = None
    for s in scales:
        new_h = round_to_multiple(int(round(H * s)), patch_size)
        new_w = round_to_multiple(int(round(W * s)), patch_size)
        scaled = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=True)
        logits = model(scaled)
        prob = F.softmax(logits, dim=1)
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=True)
        accum = prob if accum is None else accum + prob
        if flip:
            scaled_f = torch.flip(scaled, dims=[3])
            logits_f = model(scaled_f)
            prob_f = F.softmax(logits_f, dim=1)
            prob_f = torch.flip(prob_f, dims=[3])
            prob_f = F.interpolate(prob_f, size=(H, W), mode="bilinear", align_corners=True)
            accum = accum + prob_f
    return accum / accum.sum(dim=1, keepdim=True).clamp_min(1e-8)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_ckpt(cfg, ckpt).to(args.device)
    model.eval()

    val_set = VOCSegmentation(
        root=cfg["data"]["root"], split="val",
        transforms=build_val_transforms(cfg),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    multi_scale = args.multi_scale or cfg["eval"].get("multi_scale", False)
    flip = args.flip or cfg["eval"].get("flip", False)
    scales = args.scales or cfg["eval"].get("scales", [1.0])
    if not multi_scale:
        scales = [1.0]

    meter = SegMeter(num_classes=cfg["model"]["num_classes"],
                     ignore_index=cfg["data"]["ignore_index"])

    for image, mask in tqdm(val_loader, desc="eval"):
        image = image.to(args.device, non_blocking=True)
        mask = mask.to(args.device, non_blocking=True)
        if multi_scale or flip:
            prob = predict_logits(model, image, scales=scales, flip=flip)
            pred = prob.argmax(dim=1)
        else:
            pred = model(image).argmax(dim=1)
        meter.update(pred, mask)

    metrics = meter.compute()
    print(f"mIoU = {metrics['miou']*100:.2f}")
    print(f"pAcc = {metrics['pacc']*100:.2f}")
    print(f"mAcc = {metrics['macc']*100:.2f}")
    print("Per-class IoU:")
    for cls_name, iou in zip(VOC_CLASSES, metrics["iou_per_class"]):
        print(f"  {cls_name:15s}: {iou*100:6.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
