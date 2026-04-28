# DINOv3 + PSPNet on Pascal VOC 2012

A frozen [DINOv3](https://github.com/facebookresearch/dinov3) ViT-S/16 backbone
paired with a [PSPNet](https://arxiv.org/abs/1612.01105) Pyramid Pooling Module
decoder for semantic segmentation. Trains 2.81 M parameters in ~30 minutes on a
single RTX 5060 Ti and reaches **86.17 mIoU** on the Pascal VOC 2012 val split —
above the original PSPNet (ResNet-101 + ImageNet+COCO, 85.4) despite using a
~10× smaller, fully frozen backbone.

Implementation follows [DINOv3_PSPNet_Project_Plan.md](DINOv3_PSPNet_Project_Plan.md).

## Headline result

| Method | Backbone | Backbone params | Trainable | VOC val mIoU |
|---|---|---:|---:|---:|
| PSPNet (Zhao et al. 2017) | ResNet-101 (ImageNet) | ~45 M | full | 82.6 |
| PSPNet (Zhao et al. 2017) | ResNet-101 (ImageNet + COCO) | ~45 M | full | 85.4 |
| **This repo, single-scale** | DINOv3 ViT-S/16 *(frozen)* | 21 M | **2.81 M** | **85.53** |
| **This repo, multi-scale + flip** | DINOv3 ViT-S/16 *(frozen)* | 21 M | **2.81 M** | **86.17** |

Side-by-side qualitative comparison on six val images:
[runs/dinov3_vits16_pspnet_voc/infer_outputs/_compare_grid.png](runs/dinov3_vits16_pspnet_voc/infer_outputs/_compare_grid.png)

Per-class IoU (multi-scale + flip):

| class | IoU | class | IoU | class | IoU |
|---|---:|---|---:|---|---:|
| background | 96.50 | cow | 93.52 | person | 93.16 |
| aeroplane | 94.54 | diningtable | 74.21 | pottedplant | 73.94 |
| bicycle | 72.08 | dog | 94.22 | sheep | 91.66 |
| bird | 93.61 | horse | 92.50 | sofa | 69.58 |
| boat | 82.00 | motorbike | 92.67 | train | 93.48 |
| bottle | 86.29 | bus | 95.84 | tvmonitor | 82.78 |
| car | 92.22 | cat | 95.48 | chair | 49.31 |

The traditional VOC hard categories (chair, sofa, diningtable, potted plant,
bicycle) are still the weak spots, matching the failure mode reported in the
original PSPNet paper.

## Architecture

```
            input image (B, 3, H, W),  H,W multiples of 16
                            │
              ┌─────────────▼─────────────┐
              │  DINOv3 ViT-S/16 (frozen) │   21 M params
              │  patch tokens, embed=384  │
              └──────┬─────────────┬──────┘
            block 6 │             │ final
       (B,384,h,w) │             ▼ (B,384,h,w),  h=H/16, w=W/16
                    │      ┌──────────────┐
                    │      │  PPM         │  pyramid pool 1×1, 2×2, 3×3, 6×6
                    │      │  + 1×1 conv  │  reduce each branch to 96 ch
                    │      │  + upsample  │  concat with input → (B,768,h,w)
                    │      └──────┬───────┘
                    │             ▼
                    │      ┌──────────────┐
                    │      │ Conv 3×3 256 │
                    │      │ BN, ReLU     │       2.81 M trainable
                    │      │ Dropout 0.1  │
                    │      │ Conv 1×1 21  │
                    │      └──────┬───────┘
                    │             │ logits (B,21,h,w)
                    │             ▼
                    │   bilinear ↑16  →  (B,21,H,W) ── main loss (CE, ignore=255)
                    │
                    ▼
              ┌──────────────┐
              │ Aux head     │  Conv3×3 → BN → ReLU → Drop → Conv1×1
              │ (train-only) │  upsample to (B,21,H,W)  ── aux loss × 0.4
              └──────────────┘
```

Frozen DINOv3 means the backbone runs in `eval()` and never sees gradients;
only PPM, the segmentation head, and the auxiliary head update.

## Repository layout

```
configs/voc_config.yaml           # all hyperparameters
models/
  backbone.py                     # DINOv3 wrapper (bypasses torch.hub.load)
  ppm.py                          # Pyramid Pooling Module
  aux_head.py                     # auxiliary segmentation head
  segmentor.py                    # full DINOv3PSPNet
datasets/
  voc_dataset.py                  # VOC2012 + optional SBD trainaug split
  transforms.py                   # joint image+mask transforms
utils/
  losses.py                       # CE + aux CE
  metrics.py                      # streaming confusion-matrix mIoU
  scheduler.py                    # poly LR with warmup
  visualize.py                    # VOC-palette colorization, overlay
train.py                          # training entry point
eval.py                           # validation (single- or multi-scale + flip TTA)
infer.py                          # directory / single-image inference
scripts/smoke_test.py             # CPU/GPU forward-pass check, no weights needed
weights/                          # put dinov3_vits16_*.pth here
data/                             # put VOCdevkit/ (and optional VOCaug/) here
runs/                             # outputs (logs, ckpts, tb, eval json, infer png)
```

## Setup

### 1. Python environment

```bash
conda create -n dinov3seg python=3.10 -y
conda activate dinov3seg
pip install -r requirements.txt
```

PyTorch 2.x with CUDA support is required for any reasonable training speed.
Other requirements live in [requirements.txt](requirements.txt).

### 2. Pascal VOC 2012

```bash
mkdir -p data && cd data
wget https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
# yields data/VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation,...}
cd ..
```

Sanity check: `JPEGImages/` should contain 17,125 jpgs and
`SegmentationClass/` 2,913 pngs (1,464 train + 1,449 val).

(Optional, for the 10,582-image trainaug split.) Download SBD and either drop
pre-converted `SegmentationClassAug/*.png` into `data/VOCaug/`, or keep the
original `dataset/cls/*.mat` files there — the dataset loader handles both.
Then set `data.use_sbd: true` in the config.

### 3. DINOv3 ViT-S/16 weights

DINOv3 weights are gated. Request access at the
[DINOv3 repo](https://github.com/facebookresearch/dinov3); Meta will email a
signed URL for `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`. Then:

```bash
wget -O weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    "<your signed URL here>"
```

The path is read from
[configs/voc_config.yaml](configs/voc_config.yaml) → `model.backbone.weights_path`.

The repo itself (the Python package, used to instantiate the ViT) is fetched on
first run via `torch.hub` and cached at `~/.cache/torch/hub/facebookresearch_dinov3_main/`.

## Reproduce

### Smoke test (no real weights required)

```bash
python scripts/smoke_test.py
```
Runs PPM + head + losses + metrics end-to-end with a randomly-initialized fake
backbone — confirms shapes, gradient flow, and mIoU computation in seconds.

### Train

```bash
python train.py --config configs/voc_config.yaml
```

- Backbone frozen; ~2.81 M params train.
- AdamW + AMP + poly-LR with linear warmup. 60 epochs on the 1,464-image train
  split takes about 30 minutes on a single RTX 5060 Ti.
- Outputs:
  - [runs/dinov3_vits16_pspnet_voc/ckpts/best.pth](runs/dinov3_vits16_pspnet_voc/ckpts/best.pth) — best mIoU on val
  - `epoch_NNN.pth` every 5 epochs
  - `val_epoch_NNN.json` per-epoch full metrics (incl. per-class IoU)
  - TensorBoard logs in [runs/dinov3_vits16_pspnet_voc/tb/](runs/dinov3_vits16_pspnet_voc/tb/)
  - [train.log](runs/dinov3_vits16_pspnet_voc/train.log) full stdout/stderr

Resume:
```bash
python train.py --config configs/voc_config.yaml \
                --resume runs/dinov3_vits16_pspnet_voc/ckpts/epoch_039.pth
```

### Evaluate

Single-scale:
```bash
python eval.py --config configs/voc_config.yaml \
               --checkpoint runs/dinov3_vits16_pspnet_voc/ckpts/best.pth
```

Multi-scale + flip TTA (the headline 86.17 number):
```bash
python eval.py --config configs/voc_config.yaml \
               --checkpoint runs/dinov3_vits16_pspnet_voc/ckpts/best.pth \
               --multi-scale --flip \
               --output runs/dinov3_vits16_pspnet_voc/eval_msflip.json
```
Default scales `[0.5, 0.75, 1.0, 1.25, 1.5]`. Each scale + its horizontal flip
produces a softmax map; the maps are bilinearly resized back to the original
resolution and averaged before argmax.

### Inference / visualization

```bash
python infer.py --config configs/voc_config.yaml \
                --checkpoint runs/dinov3_vits16_pspnet_voc/ckpts/best.pth \
                --input path/to/image_or_dir \
                --output runs/dinov3_vits16_pspnet_voc/infer_outputs
```
Each input image yields three outputs:
`<stem>_pred.png` (raw class indices), `<stem>_mask.png` (VOC-palette colored),
`<stem>_overlay.png` (input image blended with the colored mask).

## Implementation notes

A few non-obvious things worth knowing if you want to read or extend the code.

- **DINOv3 token layout.** Token sequences are
  `[CLS, register_tokens (×4), patch_tokens (×N)]`. Only patch tokens go into
  the segmentor, reshaped as `(B, 384, H/16, W/16)`.

- **Bypassing torch.hub.load.** DINOv3's `hubconf.py` imports its segmentor /
  detector / depther entrypoints at module top level, which transitively pulls
  `torchmetrics`, `omegaconf`, and a custom `MultiScaleDeformableAttention`
  CUDA extension that we don't need. [models/backbone.py](models/backbone.py)
  imports `dinov3.hub.backbones.dinov3_vits16` directly from the cached repo
  and skips `hubconf.py` entirely.

- **Auxiliary supervision.** Features from transformer block `aux_layer_idx`
  (default 6) are extracted via `model.get_intermediate_layers` and fed to a
  small auxiliary head, weighted by `aux_loss_weight=0.4`. The auxiliary head
  is dropped at eval / inference.

- **Patch-size alignment.** Inputs must be multiples of 16. Training uses
  512×512 random crops; evaluation right-pads the bottom/right edges with the
  mean color and pads the mask with `ignore_index=255` so the original aspect
  ratio is preserved.

- **`ignore_index=255`.** Both the cross-entropy loss and the streaming
  confusion-matrix meter exclude pixels with value 255, matching VOC's
  boundary-region annotations.

- **Frozen-backbone discipline.** `model.train()` is overridden so the frozen
  DINOv3 stays in `eval()` even during the train loop — important because
  ViT's LayerNorms have no running stats but the surrounding code shouldn't
  try to put them in train mode either.

## Files generated by the canonical training run

```
runs/dinov3_vits16_pspnet_voc/
├── config.snapshot.yaml         # exact config used
├── train.log                    # full training log
├── ckpts/
│   ├── best.pth                 # best mIoU = 85.53 (single-scale)
│   └── epoch_004.pth ... epoch_059.pth
├── tb/                          # TensorBoard event files
├── val_epoch_000.json ... val_epoch_059.json
├── eval_msflip.json             # multi-scale + flip metrics → mIoU 86.17
└── infer_outputs/
    ├── 2010_003597_pred.png  + _mask.png  + _overlay.png   (×6 val images)
    └── _compare_grid.png        # 4-panel side-by-side comparison
```

## References

- Siméoni et al., *DINOv3* — <https://arxiv.org/abs/2508.10104>,
  <https://github.com/facebookresearch/dinov3>
- Zhao et al., *Pyramid Scene Parsing Network* — <https://arxiv.org/abs/1612.01105>
- Pascal VOC 2012 — <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>
- DINOv3 segmentation walkthrough (Debugger Cafe) —
  <https://debuggercafe.com/semantic-segmentation-with-dinov3/>

DINOv3 weights are subject to the
[DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
Pascal VOC 2012 is subject to its own terms.
