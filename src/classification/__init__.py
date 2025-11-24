"""
Classification module for image-level classification of dental diseases.

Supports multiple architectures for classifying:
- Calculus
- Caries
- Gingivitis
- Hypodontia
- Mouth Ulcer
- Tooth Discoloration
"""

from .dataset import (
    DentalClassificationDataset,
    get_classification_train_transforms,
    get_classification_val_transforms
)
from .model import (
    DentalClassifier,
    get_classification_model
)

__all__ = [
    # Dataset
    "DentalClassificationDataset",
    "get_classification_train_transforms",
    "get_classification_val_transforms",
    # Model
    "DentalClassifier",
    "get_classification_model",
]
