import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DentalDetectionDataset(Dataset):
    """
    PyTorch Dataset for dental disease detection with YOLO format annotations.

    YOLO format: class_id x_center y_center width height (normalized 0-1)
    Classes: 0=Caries, 1=Ulcer, 2=Tooth Discoloration, 3=Gingivitis
    """

    def __init__(self, image_dir, label_dir, transform=None, img_size=640):
        """
        Args:
            image_dir (str): Directory containing images
            label_dir (str): Directory containing YOLO format .txt labels
            transform (albumentations.Compose): Albumentations transformations
            img_size (int): Target image size (default 640 for YOLO)
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_size = img_size

        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) +
                                  list(self.image_dir.glob("*.png")) +
                                  list(self.image_dir.glob("*.jpeg")))

        # Class names
        self.classes = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Load YOLO format labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():  # Skip empty lines
                        values = line.strip().split()
                        if len(values) == 5:  # class_id x_center y_center width height
                            class_id = int(values[0])
                            x_center, y_center, width, height = map(float, values[1:])

                            # Convert YOLO format (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
                            # All values are normalized [0, 1]
                            x_min = x_center - width / 2
                            y_min = y_center - height / 2
                            x_max = x_center + width / 2
                            y_max = y_center + height / 2

                            boxes.append([x_min, y_min, x_max, y_max])
                            # CRITICAL: torchvision uses 0 for background, so add 1 to class_id
                            labels.append(class_id + 1)

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Get original image dimensions for scaling
        orig_h, orig_w = image.shape[:2]

        # Apply transformations
        if self.transform:
            # Albumentations expects boxes in [x_min, y_min, x_max, y_max] format (normalized or pixel coords)
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

        # Handle cases with no boxes after augmentation
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            # CRITICAL FIX: Convert normalized boxes [0,1] to pixel coordinates
            # Torchvision models expect boxes in pixel coordinates!
            _, img_h, img_w = image.shape  # After transform
            boxes_pixel = boxes.copy()
            boxes_pixel[:, [0, 2]] *= img_w  # x coordinates
            boxes_pixel[:, [1, 3]] *= img_h  # y coordinates

            # Clamp boxes to image boundaries
            boxes_pixel[:, [0, 2]] = np.clip(boxes_pixel[:, [0, 2]], 0, img_w)
            boxes_pixel[:, [1, 3]] = np.clip(boxes_pixel[:, [1, 3]], 0, img_h)

            # Filter out degenerate boxes (where x_min >= x_max or y_min >= y_max)
            valid_boxes = (boxes_pixel[:, 2] > boxes_pixel[:, 0]) & (boxes_pixel[:, 3] > boxes_pixel[:, 1])
            boxes_pixel = boxes_pixel[valid_boxes]
            labels = labels[valid_boxes]

            boxes = torch.as_tensor(boxes_pixel, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Calculate area (required by some models)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Create target dictionary (compatible with torchvision detection models)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area
        }

        return image, target

    def get_class_distribution(self):
        """Calculate class distribution in the dataset"""
        class_counts = {i: 0 for i in range(self.num_classes)}

        for img_path in self.image_files:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id] += 1

        return class_counts


def get_train_transforms(img_size=640):
    """Get training augmentation pipeline using Albumentations"""
    return A.Compose([
        # Resize
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(114, 114, 114)),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),  # Less common for dental images but still valid
        A.Rotate(limit=20, p=0.5, border_mode=0),  # ±20 degrees rotation
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5,
            border_mode=0
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),  # Simulate viewing angle changes
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-10, 10),
            shear=(-5, 5),
            p=0.3,
            mode=0
        ),

        # Spatial augmentations
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Subtle tissue deformation
        A.GridDistortion(p=0.2),

        # Cropping (helps with scale invariance)
        A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, erosion_rate=0.2, p=0.3),

        # Color/lighting augmentations (important for medical imaging)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=25,
            p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.CLAHE(clip_limit=4.0, p=0.3),  # Contrast Limited Adaptive Histogram Equalization
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),

        # Noise and blur (simulate imaging conditions)
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),

        # Quality degradation (simulate poor imaging conditions)
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
        A.Downscale(scale_min=0.75, scale_max=0.95, p=0.2),

        # Final normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='albumentations',
        label_fields=['labels'],
        min_visibility=0.3,  # Keep boxes with at least 30% visible
        min_area=100  # Filter out very small boxes after augmentation
    ))


def get_val_transforms(img_size=640):
    """Get validation/test transforms (no augmentation)"""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(114, 114, 114)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))


def collate_fn(batch):
    """Custom collate function for batching variable-size bounding boxes"""
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets


# Example usage
if __name__ == "__main__":
    # Paths
    train_img_dir = "data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/train"
    train_label_dir = "data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/train"

    # Create dataset
    train_dataset = DentalDetectionDataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        transform=get_train_transforms(img_size=640)
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class names: {train_dataset.classes}")

    # Check class distribution
    class_dist = train_dataset.get_class_distribution()
    print("\nClass distribution:")
    for class_id, count in class_dist.items():
        print(f"  {train_dataset.classes[class_id]}: {count} instances")

    # Test loading a sample
    image, target = train_dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample boxes shape: {target['boxes'].shape}")
    print(f"Sample labels: {target['labels']}")
