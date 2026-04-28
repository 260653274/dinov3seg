"""Single-image / directory inference & visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from models import DINOv3PSPNet
from utils import colorize_mask, overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/voc_config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True,
                   help="image path or directory of images")
    p.add_argument("--output", type=str, default="runs/infer_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="overlay transparency for the colored mask")
    p.add_argument("--max-size", type=int, default=1024,
                   help="resize so the longer side is at most this many pixels")
    return p.parse_args()


def build_model(cfg, ckpt) -> DINOv3PSPNet:
    mcfg = cfg["model"]
    model = DINOv3PSPNet(
        num_classes=mcfg["num_classes"],
        backbone_name=mcfg["backbone"]["name"],
        backbone_weights=mcfg["backbone"].get("weights_path"),
        embed_dim=mcfg["backbone"]["embed_dim"],
        aux_layer_idx=mcfg["backbone"].get("aux_layer_idx", 6),
        freeze_backbone=mcfg["backbone"].get("freeze", True),
        ppm_pool_sizes=tuple(mcfg["ppm"]["pool_sizes"]),
        ppm_reduction=mcfg["ppm"]["reduction_channels"],
        head_hidden=mcfg["head"]["hidden_channels"],
        head_dropout=mcfg["head"]["dropout"],
        use_aux=False,
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


def round_to_multiple(x: int, k: int) -> int:
    return ((x + k - 1) // k) * k


@torch.no_grad()
def infer_one(model, img_path: Path, out_dir: Path, cfg, device, alpha: float, max_size: int):
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size  # (w, h)
    w, h = orig_size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        w, h = int(round(w * scale)), int(round(h * scale))
        img_resized = img.resize((w, h), Image.BILINEAR)
    else:
        img_resized = img
    patch = 16
    pad_w = round_to_multiple(w, patch) - w
    pad_h = round_to_multiple(h, patch) - h
    if pad_w or pad_h:
        canvas = Image.new("RGB", (w + pad_w, h + pad_h), tuple(int(round(m * 255)) for m in cfg["data"]["mean"]))
        canvas.paste(img_resized, (0, 0))
        img_padded = canvas
    else:
        img_padded = img_resized

    tensor = TF.to_tensor(img_padded)
    tensor = TF.normalize(tensor, cfg["data"]["mean"], cfg["data"]["std"]).unsqueeze(0).to(device)
    logits = model(tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    pred = pred[:h, :w]

    color = colorize_mask(pred)
    color = color.resize(orig_size, Image.NEAREST)
    overlaid = overlay(img, color, alpha=alpha)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    color.save(out_dir / f"{stem}_mask.png")
    overlaid.save(out_dir / f"{stem}_overlay.png")
    Image.fromarray(pred.astype(np.uint8)).resize(orig_size, Image.NEAREST).save(
        out_dir / f"{stem}_pred.png"
    )
    return out_dir / f"{stem}_overlay.png"


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(cfg, ckpt).to(args.device)
    model.eval()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if in_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in exts)
    else:
        files = [in_path]

    print(f"Running inference on {len(files)} image(s) -> {out_dir}")
    for path in files:
        result = infer_one(model, path, out_dir, cfg, args.device, args.alpha, args.max_size)
        print(f"  {path.name} -> {result}")


if __name__ == "__main__":
    main()
