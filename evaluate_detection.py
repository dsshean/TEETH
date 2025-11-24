import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.detection.dataset import DentalDetectionDataset, get_val_transforms, collate_fn
from src.detection.model import get_model
from src.detection.utils import calculate_map, calculate_iou, filter_predictions, load_checkpoint


@torch.no_grad()
def evaluate_model(model, dataloader, device, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Evaluate model on validation/test set.

    Args:
        model: Detection model
        dataloader: DataLoader for evaluation
        device: Device to run on
        confidence_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for mAP calculation

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    print("Running evaluation...")
    for images, targets in tqdm(dataloader):
        # Move to device
        images = [img.to(device) for img in images]

        # Get predictions
        predictions = model(images)

        # Filter predictions
        predictions = filter_predictions(predictions, confidence_threshold=confidence_threshold)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Calculate mAP
    print("Calculating mAP...")
    mAP, class_aps = calculate_map(all_predictions, all_targets, iou_threshold=iou_threshold)

    # Calculate additional metrics
    total_predictions = sum(len(pred['boxes']) for pred in all_predictions)
    total_targets = sum(len(target['boxes']) for target in all_targets)

    # Calculate per-class statistics
    class_names = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
    class_stats = {}

    for class_id, class_name in enumerate(class_names):
        class_preds = sum((pred['labels'] == class_id).sum().item() for pred in all_predictions)
        class_gts = sum((target['labels'] == class_id).sum().item() for target in all_targets)

        class_stats[class_name] = {
            'ap': class_aps[class_id],
            'predictions': class_preds,
            'ground_truths': class_gts
        }

    results = {
        'mAP': mAP,
        'class_aps': {name: ap for name, ap in zip(class_names, class_aps)},
        'class_stats': class_stats,
        'total_predictions': total_predictions,
        'total_targets': total_targets,
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold
    }

    return results


def visualize_predictions(model, dataset, device, num_samples=5, confidence_threshold=0.5, save_dir='visualizations'):
    """
    Visualize model predictions on sample images.

    Args:
        model: Detection model
        dataset: Dataset to sample from
        device: Device to run on
        num_samples: Number of samples to visualize
        confidence_threshold: Confidence threshold for predictions
        save_dir: Directory to save visualizations
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    class_names = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
    colors = ['red', 'blue', 'green', 'orange']

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print(f"Generating visualizations for {len(indices)} samples...")
    for idx in tqdm(indices):
        # Get image and target
        image, target = dataset[idx]

        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_vis = image * std + mean
        image_vis = image_vis.permute(1, 2, 0).cpu().numpy()
        image_vis = np.clip(image_vis, 0, 1)

        # Get predictions
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            predictions = model(image_tensor)[0]

        # Filter predictions by confidence
        conf_mask = predictions['scores'] >= confidence_threshold
        pred_boxes = predictions['boxes'][conf_mask].cpu().numpy()
        pred_labels = predictions['labels'][conf_mask].cpu().numpy()
        pred_scores = predictions['scores'][conf_mask].cpu().numpy()

        # Get ground truth
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot ground truth
        ax1.imshow(image_vis)
        ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
        ax1.axis('off')

        h, w = image_vis.shape[:2]
        for box, label in zip(gt_boxes, gt_labels):
            # Convert normalized coords to pixels
            x1, y1, x2, y2 = box * np.array([w, h, w, h])
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=colors[label], facecolor='none'
            )
            ax1.add_patch(rect)
            ax1.text(
                x1, y1 - 5,
                class_names[label],
                fontsize=12, color='white',
                bbox=dict(facecolor=colors[label], alpha=0.8)
            )

        # Plot predictions
        ax2.imshow(image_vis)
        ax2.set_title('Predictions', fontsize=16, fontweight='bold')
        ax2.axis('off')

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            # Convert normalized coords to pixels
            x1, y1, x2, y2 = box * np.array([w, h, w, h])
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=colors[label], facecolor='none', linestyle='--'
            )
            ax2.add_patch(rect)
            ax2.text(
                x1, y1 - 5,
                f'{class_names[label]}: {score:.2f}',
                fontsize=12, color='white',
                bbox=dict(facecolor=colors[label], alpha=0.8)
            )

        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to: {save_dir}")


def main(args):
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config if available
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = checkpoint_dir / 'config.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Loaded config from checkpoint directory")
    else:
        config = vars(args)

    # Create dataset
    print("Creating dataset...")
    eval_dataset = DentalDetectionDataset(
        image_dir=args.img_dir,
        label_dir=args.label_dir,
        transform=get_val_transforms(img_size=args.img_size)
    )

    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Create model
    print(f"Creating model: {args.model_type} with {args.backbone} backbone")
    model = get_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        backbone=args.backbone,
        pretrained=False
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model)
    model.to(device)

    # Evaluate
    results = evaluate_model(
        model,
        eval_loader,
        device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"mAP @ IoU={args.iou_threshold}: {results['mAP']:.4f}")
    print("\nPer-class Average Precision:")
    for class_name, ap in results['class_aps'].items():
        print(f"  {class_name}: {ap:.4f}")

    print("\nPer-class Statistics:")
    for class_name, stats in results['class_stats'].items():
        print(f"  {class_name}:")
        print(f"    Predictions: {stats['predictions']}")
        print(f"    Ground Truths: {stats['ground_truths']}")

    print(f"\nTotal Predictions: {results['total_predictions']}")
    print(f"Total Ground Truths: {results['total_targets']}")
    print("=" * 80)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_predictions(
            model,
            eval_dataset,
            device,
            num_samples=args.num_visualizations,
            confidence_threshold=args.confidence_threshold,
            save_dir=output_dir / 'visualizations'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate dental disease detection model')

    # Data parameters
    parser.add_argument('--img_dir', type=str,
                        default='data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/val',
                        help='Images directory')
    parser.add_argument('--label_dir', type=str,
                        default='data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/val',
                        help='Labels directory')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='fasterrcnn',
                        choices=['fasterrcnn', 'retinanet', 'fcos'],
                        help='Detection model type')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of detection classes')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for mAP calculation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Generate prediction visualizations')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of samples to visualize')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory')

    args = parser.parse_args()
    main(args)
