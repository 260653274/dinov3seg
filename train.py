"""Training entry point for DINOv3 + PSPNet on Pascal VOC 2012."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import VOCSegmentation, build_train_transforms, build_val_transforms
from models import DINOv3PSPNet
from utils import CEAuxLoss, SegMeter, PolyLRWithWarmup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/voc_config.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume from")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-aux", action="store_true",
                   help="disable auxiliary loss for ablation")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, use_aux: bool) -> DINOv3PSPNet:
    mcfg = cfg["model"]
    msfa_cfg = mcfg.get("msfa", {}) or {}
    return DINOv3PSPNet(
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
        use_aux=use_aux,
        msfa_enabled=bool(msfa_cfg.get("enabled", False)),
        msfa_layers=tuple(msfa_cfg.get("layers", (3, 6, 9, 11))),
        msfa_per_layer_channels=int(msfa_cfg.get("per_layer_channels", 96)),
        msfa_out_channels=int(msfa_cfg.get("out_channels", mcfg["backbone"]["embed_dim"])),
        msfa_upsample=bool(msfa_cfg.get("upsample", False)),
    )


def build_loaders(cfg):
    data = cfg["data"]
    train_split = "trainaug" if data.get("use_sbd", False) else "train"
    train_set = VOCSegmentation(
        root=data["root"], split=train_split,
        transforms=build_train_transforms(cfg),
        sbd_root=data.get("sbd_root") if data.get("use_sbd", False) else None,
    )
    val_set = VOCSegmentation(
        root=data["root"], split="val",
        transforms=build_val_transforms(cfg),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def evaluate(model, loader, num_classes, ignore_index, device, amp: bool):
    model.eval()
    meter = SegMeter(num_classes=num_classes, ignore_index=ignore_index)
    autocast = torch.cuda.amp.autocast if device == "cuda" else _NullCtx
    with torch.no_grad():
        for image, mask in tqdm(loader, desc="val", leave=False):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast(enabled=amp):
                logits = model(image)
            pred = logits.argmax(dim=1)
            meter.update(pred, mask)
    return meter.compute()


class _NullCtx:
    def __init__(self, *_, **__): pass
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"].get("seed", 42))

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    with open(out_dir / "config.snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    device = args.device
    use_aux = not args.no_aux

    model = build_model(cfg, use_aux=use_aux).to(device)
    train_loader, val_loader = build_loaders(cfg)

    # Optionally warm-start model weights from a previous checkpoint
    # (Strategy A: continue from the frozen-baseline best.pth)
    init_from = cfg["train"].get("init_from")
    if init_from and os.path.isfile(init_from):
        ck0 = torch.load(init_from, map_location="cpu")
        state0 = ck0["model"] if isinstance(ck0, dict) and "model" in ck0 else ck0
        missing, unexpected = model.load_state_dict(state0, strict=cfg["train"].get("init_strict", False))
        print(f"[init_from] {init_from}")
        print(f"  loaded missing={len(missing)} unexpected={len(unexpected)}")

    # Split parameters into two groups: backbone (lower LR) and decoder (full LR).
    backbone_lr_mult = float(cfg["train"].get("backbone_lr_mult", 1.0))
    backbone_params, decoder_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else decoder_params).append(p)
    n_back = sum(p.numel() for p in backbone_params)
    n_dec  = sum(p.numel() for p in decoder_params)
    print(f"Trainable parameters: backbone={n_back/1e6:.2f}M (lr_mult={backbone_lr_mult}) "
          f"decoder={n_dec/1e6:.2f}M  total={(n_back+n_dec)/1e6:.2f}M")

    opt_cfg = cfg["train"]
    base_lr = opt_cfg["lr"]
    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": base_lr,
                             "name": "decoder"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult,
                             "name": "backbone_unfrozen"})

    if opt_cfg["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr,
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        )
    elif opt_cfg["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=base_lr, momentum=0.9,
            weight_decay=opt_cfg["weight_decay"], nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['optimizer']}")
    trainable = backbone_params + decoder_params

    iters_per_epoch = len(train_loader)
    total_iters = iters_per_epoch * opt_cfg["epochs"]
    scheduler = PolyLRWithWarmup(
        optimizer,
        total_iters=total_iters,
        power=opt_cfg["poly_power"],
        warmup_iters=opt_cfg.get("warmup_iters", 0),
        warmup_ratio=opt_cfg.get("warmup_ratio", 0.1),
    )

    criterion = CEAuxLoss(
        ignore_index=cfg["data"]["ignore_index"],
        aux_weight=opt_cfg["aux_loss_weight"] if use_aux else 0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=opt_cfg["amp"] and device == "cuda")
    autocast = torch.cuda.amp.autocast if device == "cuda" else _NullCtx

    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.last_iter = ck.get("scheduler_last_iter", 0)
        start_epoch = ck.get("epoch", 0) + 1
        best_miou = ck.get("best_miou", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    global_step = start_epoch * iters_per_epoch

    for epoch in range(start_epoch, opt_cfg["epochs"]):
        model.train()
        # Keep frozen backbone in eval() so BN stats etc. don't update
        if cfg["model"]["backbone"].get("freeze", True):
            model.backbone.eval()

        running = {"loss": 0.0, "loss_main": 0.0, "loss_aux": 0.0, "n": 0}
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for image, mask in pbar:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=opt_cfg["amp"] and device == "cuda"):
                out = model(image)
                if isinstance(out, tuple):
                    logits, aux_logits = out
                else:
                    logits, aux_logits = out, None
                loss, parts = criterion(logits, mask, aux_logits)

            scaler.scale(loss).backward()
            if opt_cfg.get("grad_clip", 0) and opt_cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, opt_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bsz = image.size(0)
            running["loss"] += loss.item() * bsz
            running["loss_main"] += parts["loss_main"].item() * bsz
            running["loss_aux"] += parts["loss_aux"].item() * bsz
            running["n"] += bsz

            global_step += 1
            if global_step % opt_cfg["log_interval"] == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_main", parts["loss_main"].item(), global_step)
                writer.add_scalar("train/loss_aux", parts["loss_aux"].item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

        n = max(running["n"], 1)
        avg_loss = running["loss"] / n
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f} "
              f"time={time.time() - t0:.1f}s")

        if (epoch + 1) % opt_cfg.get("val_interval", 1) == 0:
            metrics = evaluate(
                model, val_loader,
                num_classes=cfg["model"]["num_classes"],
                ignore_index=cfg["data"]["ignore_index"],
                device=device,
                amp=opt_cfg["amp"] and device == "cuda",
            )
            print(f"[epoch {epoch}] mIoU={metrics['miou']*100:.2f} "
                  f"pAcc={metrics['pacc']*100:.2f} "
                  f"mAcc={metrics['macc']*100:.2f}")
            writer.add_scalar("val/miou", metrics["miou"], epoch)
            writer.add_scalar("val/pacc", metrics["pacc"], epoch)
            writer.add_scalar("val/macc", metrics["macc"], epoch)
            with open(out_dir / f"val_epoch_{epoch:03d}.json", "w") as f:
                json.dump(metrics, f, indent=2)

            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler_last_iter": scheduler.last_iter,
                    "best_miou": best_miou,
                    "config": cfg,
                }, out_dir / "ckpts" / "best.pth")
                print(f"  -> new best mIoU={best_miou*100:.2f}, saved best.pth")

        if (epoch + 1) % opt_cfg.get("ckpt_interval", 5) == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_last_iter": scheduler.last_iter,
                "best_miou": best_miou,
                "config": cfg,
            }, out_dir / "ckpts" / f"epoch_{epoch:03d}.pth")

    writer.close()
    print(f"Training finished. Best mIoU = {best_miou*100:.2f}")


if __name__ == "__main__":
    main()
