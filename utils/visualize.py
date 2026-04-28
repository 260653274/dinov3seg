"""Visualization helpers: paint a class-index map with the VOC palette."""

from __future__ import annotations

import numpy as np
from PIL import Image

from datasets.voc_dataset import VOC_PALETTE


def colorize_mask(mask: np.ndarray, palette: np.ndarray = VOC_PALETTE) -> Image.Image:
    """mask: (H, W) int array of class indices (255 -> ignored, painted black)."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask < palette.shape[0]
    out[valid] = palette[mask[valid]]
    return Image.fromarray(out)


def overlay(image: Image.Image, color_mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    image = image.convert("RGBA")
    color_mask = color_mask.convert("RGBA")
    color_mask.putalpha(int(alpha * 255))
    return Image.alpha_composite(image, color_mask).convert("RGB")
