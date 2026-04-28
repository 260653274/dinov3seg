"""Pascal VOC 2012 semantic segmentation dataset.

Supports the original 1,464/1,449 train/val split and the SBD-augmented
10,582 train split (a.k.a. `trainaug`). Masks are read with PIL in 'P' mode
which yields class indices 0..20 with 255 marking the ignore region.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


VOC_CLASSES: Tuple[str, ...] = (
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Pascal VOC color palette (used for visualization)
VOC_PALETTE: np.ndarray = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def _read_split(split_file: str) -> List[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


class VOCSegmentation(Dataset):
    """Pascal VOC 2012 segmentation.

    Args:
        root: VOC2012 root dir, containing JPEGImages/, SegmentationClass/, ImageSets/Segmentation/
        split: 'train' | 'val' | 'trainaug' (requires sbd_root)
        transforms: callable(image, mask) -> (image_tensor, mask_tensor)
        sbd_root: optional path to SBD VOCaug dir containing
            'dataset/cls/<id>.mat' or pre-converted PNGs at 'SegmentationClassAug/<id>.png'
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        sbd_root: Optional[str] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.transforms = transforms
        self.sbd_root = sbd_root

        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")
        split_dir = os.path.join(root, "ImageSets", "Segmentation")

        if split == "trainaug":
            ids = self._collect_trainaug_ids(split_dir)
        else:
            split_file = os.path.join(split_dir, f"{split}.txt")
            if not os.path.isfile(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            ids = _read_split(split_file)

        self.samples: List[Tuple[str, str]] = []
        missing = 0
        for image_id in ids:
            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            mask_path = self._resolve_mask_path(image_id)
            if mask_path is None or not os.path.isfile(img_path):
                missing += 1
                continue
            self.samples.append((img_path, mask_path))

        if missing > 0:
            print(f"[VOCSegmentation] split={split}: skipped {missing} missing samples; "
                  f"kept {len(self.samples)}.")
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split '{split}' under '{root}'.")

    # ---- helpers ---------------------------------------------------------

    def _collect_trainaug_ids(self, split_dir: str) -> List[str]:
        if self.sbd_root is None:
            raise ValueError("trainaug split requested but sbd_root not provided")

        # Prefer a pre-existing trainaug.txt (common in the community)
        candidate = os.path.join(split_dir, "trainaug.txt")
        if os.path.isfile(candidate):
            return _read_split(candidate)

        # Otherwise: union of VOC train.txt + SBD train ids minus VOC val ids
        voc_train = set(_read_split(os.path.join(split_dir, "train.txt")))
        voc_val = set(_read_split(os.path.join(split_dir, "val.txt")))

        sbd_train_file = os.path.join(self.sbd_root, "dataset", "train.txt")
        if not os.path.isfile(sbd_train_file):
            raise FileNotFoundError(
                f"Cannot build trainaug: missing {candidate} and {sbd_train_file}"
            )
        sbd_ids = set(_read_split(sbd_train_file))
        return sorted((voc_train | sbd_ids) - voc_val)

    def _resolve_mask_path(self, image_id: str) -> Optional[str]:
        # 1) Native VOC mask
        voc_mask = os.path.join(self.mask_dir, f"{image_id}.png")
        if os.path.isfile(voc_mask):
            return voc_mask
        # 2) SBD pre-converted png mask (community 'SegmentationClassAug' layout)
        if self.sbd_root is not None:
            aug_png = os.path.join(self.sbd_root, "SegmentationClassAug", f"{image_id}.png")
            if os.path.isfile(aug_png):
                return aug_png
            # 3) Original SBD .mat (we'll decode lazily in __getitem__)
            mat_path = os.path.join(self.sbd_root, "dataset", "cls", f"{image_id}.mat")
            if os.path.isfile(mat_path):
                return mat_path
        return None

    @staticmethod
    def _load_mask(path: str) -> Image.Image:
        if path.endswith(".mat"):
            from scipy.io import loadmat  # local import to keep optional
            data = loadmat(path)
            arr = data["GTcls"][0, 0]["Segmentation"].astype(np.uint8)
            return Image.fromarray(arr, mode="P")
        return Image.open(path)

    # ---- Dataset API -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = self._load_mask(mask_path)
        if mask.mode not in ("L", "P"):
            mask = mask.convert("P")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask
