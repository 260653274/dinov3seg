from .voc_dataset import VOCSegmentation, VOC_CLASSES, VOC_PALETTE
from .transforms import build_train_transforms, build_val_transforms

__all__ = [
    "VOCSegmentation", "VOC_CLASSES", "VOC_PALETTE",
    "build_train_transforms", "build_val_transforms",
]
