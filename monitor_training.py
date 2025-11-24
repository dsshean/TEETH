"""
Script to monitor training progress and compute accuracy metrics periodically.
Run this after a few epochs to see current model performance.
"""
import torch
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from dataset import DentalDetectionDataset, get_val_transforms, collate_fn
from model import get_model
from utils import load_checkpoint, calculate_map, filter_predictions


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, val_loader, device, confidence_threshold=0.5):
    """Evaluate a checkpoint and return metrics"""

    # Load model
    model = get_model(num_classes=4, model_type='fasterrcnn', backbone='resnet50', pretrained=False)
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()

    # Collect predictions and targets
    all_predictions = []
    all_targets = []

    print("Running inference on validation set...")
    for images, targets in tqdm(val_loader):
        images = [img.to(device) for img in images]

        # Get predictions
        predictions = model(images)

        # Filter predictions
        predictions = filter_predictions(predictions, confidence_threshold=confidence_threshold)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Calculate mAP
    print("Calculating mAP...")
    mAP, class_aps = calculate_map(all_predictions, all_targets, iou_threshold=0.5)

    # Calculate detection statistics
    total_predictions = sum(len(pred['boxes']) for pred in all_predictions)
    total_targets = sum(len(target['boxes']) for target in all_targets)

    class_names = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
    class_stats = {}

    for class_id, class_name in enumerate(class_names):
        class_preds = sum((pred['labels'] == class_id + 1).sum().item() for pred in all_predictions)
        class_gts = sum((target['labels'] == class_id + 1).sum().item() for target in all_targets)

        class_stats[class_name] = {
            'ap': class_aps[class_id],
            'predictions': class_preds,
            'ground_truths': class_gts,
            'recall_estimate': class_preds / (class_gts + 1e-6)
        }

    return {
        'mAP': mAP,
        'class_aps': {name: ap for name, ap in zip(class_names, class_aps)},
        'class_stats': class_stats,
        'total_predictions': total_predictions,
        'total_targets': total_targets
    }


def main():
    parser = argparse.ArgumentParser(description='Monitor training and evaluate current checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Training output directory')
    parser.add_argument('--checkpoint_type', type=str, default='best',
                        choices=['best', 'latest'],
                        help='Which checkpoint to evaluate')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    # Find checkpoint
    if args.checkpoint_type == 'best':
        checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'
    else:
        checkpoint_path = checkpoint_dir / 'checkpoint_latest.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Load training history
    history_path = checkpoint_dir / 'history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        print("\n" + "="*80)
        print("TRAINING PROGRESS")
        print("="*80)
        print(f"Epochs completed: {len(history['train_loss'])}")
        print(f"Current training loss: {history['train_loss'][-1]:.4f}")
        print(f"Current validation loss: {history['val_loss'][-1]:.4f}")
        if 'val_box_loss' in history:
            print(f"  ├─ Box regression loss: {history['val_box_loss'][-1]:.4f}")
            print(f"  ├─ Classifier loss: {history['val_cls_loss'][-1]:.4f}")
            print(f"  ├─ Objectness loss: {history['val_obj_loss'][-1]:.4f}")
            print(f"  └─ RPN box loss: {history['val_rpn_loss'][-1]:.4f}")
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        print("="*80)

    # Create validation dataset
    print("\nLoading validation dataset...")
    val_dataset = DentalDetectionDataset(
        image_dir="data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/val",
        label_dir="data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/val",
        transform=get_val_transforms(img_size=640)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluating checkpoint: {checkpoint_path}")

    results = evaluate_checkpoint(checkpoint_path, val_loader, device, args.confidence_threshold)

    # Print results
    print("\n" + "="*80)
    print("ACCURACY METRICS")
    print("="*80)
    print(f"mAP @ IoU=0.5: {results['mAP']:.2%}")
    print("\nPer-class Average Precision:")
    for class_name, ap in results['class_aps'].items():
        print(f"  {class_name:25s}: {ap:.2%}")

    print("\nDetection Statistics:")
    for class_name, stats in results['class_stats'].items():
        print(f"  {class_name}:")
        print(f"    Predictions: {stats['predictions']:4d} | Ground Truths: {stats['ground_truths']:4d} | Recall Est: {stats['recall_estimate']:.2%}")

    print(f"\nTotal Predictions: {results['total_predictions']}")
    print(f"Total Ground Truths: {results['total_targets']}")
    print("="*80)

    # Save results
    results_path = checkpoint_dir / f'accuracy_metrics_{args.checkpoint_type}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
