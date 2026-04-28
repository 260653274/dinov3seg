"""Joint image+mask transforms for semantic segmentation.

We avoid albumentations to keep dependencies minimal — pure torchvision.transforms.functional
with PIL. All transforms operate on (PIL.Image image, PIL.Image mask) where mask
is mode 'L' (or 'P') with class indices and `ignore_index` for ignored pixels.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


class Compose:
    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomScale:
    """Scale image by a random factor in [scale_min, scale_max]."""

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0)) -> None:
        self.scale_min, self.scale_max = scale_range

    def __call__(self, image: Image.Image, mask: Image.Image):
        scale = random.uniform(self.scale_min, self.scale_max)
        w, h = image.size
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask


class RandomCrop:
    """Random crop with padding when the image is smaller than crop_size."""

    def __init__(self, size: int, ignore_index: int = 255, mean=(0.485, 0.456, 0.406)):
        self.size = size
        self.ignore_index = ignore_index
        self.pad_color = tuple(int(round(m * 255)) for m in mean)

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        pad_w = max(self.size - w, 0)
        pad_h = max(self.size - h, 0)
        if pad_w > 0 or pad_h > 0:
            image = self._pad(image, pad_w, pad_h, self.pad_color)
            mask = self._pad(mask, pad_w, pad_h, self.ignore_index)
            w, h = image.size

        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        image = image.crop((x, y, x + self.size, y + self.size))
        mask = mask.crop((x, y, x + self.size, y + self.size))
        return image, mask

    @staticmethod
    def _pad(img: Image.Image, pad_w: int, pad_h: int, fill):
        new_w, new_h = img.size[0] + pad_w, img.size[1] + pad_h
        canvas = Image.new(img.mode, (new_w, new_h), fill)
        canvas.paste(img, (0, 0))
        return canvas


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class PhotometricDistortion:
    """ColorJitter + small Gaussian blur on image only (mask untouched)."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        blur_p: float = 0.2,
    ):
        from torchvision.transforms import ColorJitter

        self.color_jitter = ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )
        self.blur_p = blur_p

    def __call__(self, image, mask):
        image = self.color_jitter(image)
        if random.random() < self.blur_p:
            radius = random.uniform(0.1, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, mask


class Resize:
    """Resize the shorter side to `size` (image bilinear, mask nearest)."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        scale = self.size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask


class CenterCrop:
    def __init__(self, size: int, ignore_index: int = 255, mean=(0.485, 0.456, 0.406)):
        self.size = size
        self.ignore_index = ignore_index
        self.pad_color = tuple(int(round(m * 255)) for m in mean)

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        pad_w = max(self.size - w, 0)
        pad_h = max(self.size - h, 0)
        if pad_w > 0 or pad_h > 0:
            image = RandomCrop._pad(image, pad_w, pad_h, self.pad_color)
            mask = RandomCrop._pad(mask, pad_w, pad_h, self.ignore_index)
            w, h = image.size
        x = (w - self.size) // 2
        y = (h - self.size) // 2
        image = image.crop((x, y, x + self.size, y + self.size))
        mask = mask.crop((x, y, x + self.size, y + self.size))
        return image, mask


class PadToMultiple:
    """Pad image+mask so each side is a multiple of `divisor` (used at eval)."""

    def __init__(self, divisor: int = 16, ignore_index: int = 255, mean=(0.485, 0.456, 0.406)):
        self.divisor = divisor
        self.ignore_index = ignore_index
        self.pad_color = tuple(int(round(m * 255)) for m in mean)

    def __call__(self, image, mask):
        w, h = image.size
        new_w = (w + self.divisor - 1) // self.divisor * self.divisor
        new_h = (h + self.divisor - 1) // self.divisor * self.divisor
        if (new_w, new_h) != (w, h):
            image = RandomCrop._pad(image, new_w - w, new_h - h, self.pad_color)
            mask = RandomCrop._pad(mask, new_w - w, new_h - h, self.ignore_index)
        return image, mask


class ToTensorNormalize:
    """Image -> float tensor (normalized); mask -> long tensor."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.mean, self.std)
        mask_np = np.array(mask, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np).long()
        return image_t, mask_t


def build_train_transforms(cfg) -> Compose:
    return Compose([
        RandomScale(tuple(cfg["data"]["scale_range"])),
        RandomCrop(cfg["data"]["crop_size"],
                   ignore_index=cfg["data"]["ignore_index"],
                   mean=cfg["data"]["mean"]),
        RandomHorizontalFlip(p=0.5),
        PhotometricDistortion(),
        ToTensorNormalize(mean=cfg["data"]["mean"], std=cfg["data"]["std"]),
    ])


def build_val_transforms(cfg) -> Compose:
    crop = cfg["data"]["crop_size"]
    return Compose([
        Resize(crop),
        CenterCrop(crop,
                   ignore_index=cfg["data"]["ignore_index"],
                   mean=cfg["data"]["mean"]),
        ToTensorNormalize(mean=cfg["data"]["mean"], std=cfg["data"]["std"]),
    ])
