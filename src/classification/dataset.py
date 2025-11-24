import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class DentalClassificationDataset(Dataset):
    """
    PyTorch Dataset for dental disease classification (image-level labels, no bounding boxes).

    Classes:
    0: Calculus
    1: Caries
    2: Gingivitis
    3: Hypodontia
    4: Mouth Ulcer
    5: Tooth Discoloration
    """

    def __init__(self, root_dir, transform=None, split='train', test_size=0.2, val_size=0.1, random_state=42):
        """
        Args:
            root_dir (str): Root directory containing class folders
            transform (albumentations.Compose): Albumentations transformations
            split (str): 'train', 'val', or 'test'
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set (from remaining after test split)
            random_state (int): Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Class names (folder names)
        self.class_names = [
            'Calculus',
            'Data caries',  # Note: folder is "Data caries"
            'Gingivitis',
            'hypodontia',
            'Mouth Ulcer',
            'Tooth Discoloration'
        ]

        self.num_classes = len(self.class_names)

        # Collect all image paths and labels
        all_images = []
        all_labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name

            # Get all images recursively (handles nested folders)
            images = list(class_dir.rglob("*.jpg")) + \
                    list(class_dir.rglob("*.png")) + \
                    list(class_dir.rglob("*.jpeg"))

            all_images.extend(images)
            all_labels.extend([class_idx] * len(images))

        # Split into train/val/test
        # First split: train+val vs test
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            all_images, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
        )

        # Second split: train vs val
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels, test_size=val_size/(1-test_size),
            random_state=random_state, stratify=train_val_labels
        )

        # Select the appropriate split
        if split == 'train':
            self.images = train_images
            self.labels = train_labels
        elif split == 'val':
            self.images = val_images
            self.labels = val_labels
        elif split == 'test':
            self.images = test_images
            self.labels = test_labels
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Get label
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label

    def get_class_distribution(self):
        """Calculate class distribution in the dataset"""
        from collections import Counter
        label_counts = Counter(self.labels)
        return {self.class_names[i]: label_counts[i] for i in range(self.num_classes)}


def get_classification_train_transforms(img_size=224):
    """Training augmentation for classification"""
    return A.Compose([
        # Resize
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5),

        # Color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=25, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.CLAHE(clip_limit=4.0, p=0.3),

        # Noise and blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Quality
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),

        # Normalization (ImageNet statistics)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_classification_val_transforms(img_size=224):
    """Validation/test transforms (no augmentation)"""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# Example usage
if __name__ == "__main__":
    # Create datasets
    train_dataset = DentalClassificationDataset(
        root_dir="data",
        transform=get_classification_train_transforms(img_size=224),
        split='train'
    )

    val_dataset = DentalClassificationDataset(
        root_dir="data",
        transform=get_classification_val_transforms(img_size=224),
        split='val'
    )

    test_dataset = DentalClassificationDataset(
        root_dir="data",
        transform=get_classification_val_transforms(img_size=224),
        split='test'
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class names: {train_dataset.class_names}")

    # Check class distribution
    print("\nTraining class distribution:")
    for class_name, count in train_dataset.get_class_distribution().items():
        print(f"  {class_name}: {count} images")

    # Test loading a sample
    image, label = train_dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample label: {label} ({train_dataset.class_names[label]})")
