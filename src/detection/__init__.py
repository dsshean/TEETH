"""
Detection module for object detection of dental diseases.

Supports Faster R-CNN, RetinaNet, and FCOS models for detecting:
- Caries
- Ulcer
- Tooth Discoloration
- Gingivitis
"""

from .dataset import (
    DentalDetectionDataset,
    get_train_transforms,
    get_val_transforms,
    collate_fn
)
from .model import get_model
from .utils import (
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    calculate_map,
    calculate_iou,
    filter_predictions
)

__all__ = [
    # Dataset
    "DentalDetectionDataset",
    "get_train_transforms",
    "get_val_transforms",
    "collate_fn",
    # Model
    "get_model",
    # Utils
    "AverageMeter",
    "save_checkpoint",
    "load_checkpoint",
    "calculate_map",
    "calculate_iou",
    "filter_predictions",
]
