"""
Test script to verify dataset is loading correctly with proper box coordinates.
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.detection.dataset import DentalDetectionDataset, get_train_transforms

# Paths
train_img_dir = "data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/train"
train_label_dir = "data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/train"

# Create dataset
print("Creating dataset...")
dataset = DentalDetectionDataset(
    image_dir=train_img_dir,
    label_dir=train_label_dir,
    transform=get_train_transforms(img_size=640)
)

print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# Test loading a sample
print("\nTesting sample loading...")
image, target = dataset[0]

print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Boxes shape: {target['boxes'].shape}")
print(f"Boxes dtype: {target['boxes'].dtype}")
print(f"Labels: {target['labels']}")
print(f"Labels range: [{target['labels'].min()}, {target['labels'].max()}]")
print(f"Area: {target['area']}")

# Check if boxes are in pixel coordinates
if len(target['boxes']) > 0:
    _, H, W = image.shape
    print(f"\nImage dimensions: {W}x{H}")
    print(f"Box coordinates (should be in pixel range [0, {W}] x [0, {H}]):")
    for i, box in enumerate(target['boxes']):
        print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        # Verify boxes are in pixel coordinates
        assert box[0] >= 0 and box[2] <= W + 1, f"Box x coords out of range: {box}"
        assert box[1] >= 0 and box[3] <= H + 1, f"Box y coords out of range: {box}"
        assert box[2] > box[0], f"Box width invalid: {box}"
        assert box[3] > box[1], f"Box height invalid: {box}"

    # Verify labels are 1-indexed (1-4, not 0-3)
    assert target['labels'].min() >= 1, "Labels should start from 1 (0 is reserved for background)"
    assert target['labels'].max() <= 4, "Labels should be <= 4 for 4 classes"

    print("\n✓ All validations passed!")
    print("✓ Boxes are in pixel coordinates")
    print("✓ Labels are correctly 1-indexed (1-4)")

    # Visualize
    print("\nGenerating visualization...")

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = image * std + mean
    img_vis = img_vis.permute(1, 2, 0).cpu().numpy()
    img_vis = np.clip(img_vis, 0, 1)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_vis)

    colors = ['red', 'blue', 'green', 'orange']
    class_names_display = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']

    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Label is 1-4, so subtract 1 for indexing
        color = colors[label - 1]
        class_name = class_names_display[label - 1]

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            x1, y1 - 5,
            f'{class_name} (label={label})',
            fontsize=10, color='white',
            bbox=dict(facecolor=color, alpha=0.8)
        )

    ax.set_title('Sample with Bounding Boxes (Pixel Coordinates)', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('test_dataset_sample.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to: test_dataset_sample.png")

else:
    print("\nNo boxes in this sample")

print("\n" + "="*60)
print("Dataset test completed successfully!")
print("="*60)
