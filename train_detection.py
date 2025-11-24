import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.detection.dataset import DentalDetectionDataset, get_train_transforms, get_val_transforms, collate_fn
from src.detection.model import get_model
from src.detection.utils import AverageMeter, save_checkpoint, load_checkpoint, calculate_map, filter_predictions


def train_one_epoch(model, dataloader, optimizer, device, epoch, print_freq=50):
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]")

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(losses.item(), len(images))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls_loss': f'{loss_dict.get("loss_classifier", torch.tensor(0.0)):.4f}',
            'box_loss': f'{loss_dict.get("loss_box_reg", torch.tensor(0.0)):.4f}',
            'obj_loss': f'{loss_dict.get("loss_objectness", torch.tensor(0.0)):.4f}',
            'rpn_loss': f'{loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)):.4f}'
        })

    return loss_meter.avg


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    """Validate model"""
    model.eval()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    rpn_loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [VAL]")

    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        model.train()  # Need train mode to get losses
        loss_dict = model(images, targets)
        model.eval()

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Update metrics
        loss_meter.update(losses.item(), len(images))
        cls_loss_meter.update(loss_dict.get('loss_classifier', torch.tensor(0.0)).item(), len(images))
        box_loss_meter.update(loss_dict.get('loss_box_reg', torch.tensor(0.0)).item(), len(images))
        obj_loss_meter.update(loss_dict.get('loss_objectness', torch.tensor(0.0)).item(), len(images))
        rpn_loss_meter.update(loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item(), len(images))

        # Update progress bar
        pbar.set_postfix({
            'val_loss': f'{loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}',
            'box': f'{box_loss_meter.avg:.4f}'
        })

    return loss_meter.avg, cls_loss_meter.avg, box_loss_meter.avg, obj_loss_meter.avg, rpn_loss_meter.avg


@torch.no_grad()
def compute_accuracy_metrics(model, dataloader, device, confidence_threshold=0.5):
    """Compute mAP and per-class accuracy metrics"""
    model.eval()

    all_predictions = []
    all_targets = []

    print("  Computing accuracy metrics...", end='', flush=True)

    for images, targets in dataloader:
        images = [img.to(device) for img in images]

        # Get predictions
        predictions = model(images)

        # Filter predictions
        predictions = filter_predictions(predictions, confidence_threshold=confidence_threshold)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Calculate mAP
    mAP, class_aps = calculate_map(all_predictions, all_targets, iou_threshold=0.5)

    # Calculate detection statistics
    class_names = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
    class_stats = {}

    for class_id, class_name in enumerate(class_names):
        class_preds = sum((pred['labels'] == class_id + 1).sum().item() for pred in all_predictions)
        class_gts = sum((target['labels'] == class_id + 1).sum().item() for target in all_targets)

        class_stats[class_name] = {
            'ap': class_aps[class_id],
            'predictions': class_preds,
            'ground_truths': class_gts
        }

    print(" Done!")

    return mAP, class_aps, class_stats, class_names


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create datasets
    print("Creating datasets...")
    train_dataset = DentalDetectionDataset(
        image_dir=args.train_img_dir,
        label_dir=args.train_label_dir,
        transform=get_train_transforms(img_size=args.img_size)
    )

    val_dataset = DentalDetectionDataset(
        image_dir=args.val_img_dir,
        label_dir=args.val_label_dir,
        transform=get_val_transforms(img_size=args.img_size)
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Print class distribution
    train_dist = train_dataset.get_class_distribution()
    print("\nTraining class distribution:")
    for class_id, count in train_dist.items():
        print(f"  {train_dataset.classes[class_id]}: {count} instances")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print(f"\nCreating model: {args.model_type} with {args.backbone} backbone")
    model = get_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        backbone=args.backbone,
        pretrained=args.pretrained
    )
    model.to(device)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Create learning rate scheduler
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None

    # Load checkpoint if resume
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler)
        else:
            print(f"Checkpoint not found: {args.resume}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_box_loss': [],
        'val_obj_loss': [],
        'val_rpn_loss': [],
        'lr': []
    }

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss, val_cls_loss, val_box_loss, val_obj_loss, val_rpn_loss = validate(model, val_loader, device, epoch + 1)
        history['val_loss'].append(val_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_box_loss'].append(val_box_loss)
        history['val_obj_loss'].append(val_obj_loss)
        history['val_rpn_loss'].append(val_rpn_loss)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        if scheduler is not None:
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Compute accuracy metrics at specified intervals
        compute_metrics = False
        if args.compute_map_every > 0:
            # Compute on first epoch, last epoch, and every N epochs
            if (epoch + 1) == 1 or (epoch + 1) % args.compute_map_every == 0 or (epoch + 1) == args.epochs:
                compute_metrics = True

        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs} Summary:")
        print(f"{'='*70}")
        print(f"  Training Loss:        {train_loss:.4f}")
        print(f"  Validation Loss:      {val_loss:.4f}")
        print(f"    ├─ Classifier Loss: {val_cls_loss:.4f}")
        print(f"    ├─ Box Reg Loss:    {val_box_loss:.4f}")
        print(f"    ├─ Objectness Loss: {val_obj_loss:.4f}")
        print(f"    └─ RPN Box Loss:    {val_rpn_loss:.4f}")
        print(f"  Learning Rate:        {current_lr:.6f}")

        # Compute and display accuracy metrics if requested
        if compute_metrics:
            mAP, class_aps, class_stats, class_names = compute_accuracy_metrics(
                model, val_loader, device, confidence_threshold=args.confidence_threshold
            )

            print(f"\n  {'─'*66}")
            print(f"  ACCURACY METRICS @ Epoch {epoch + 1}")
            print(f"  {'─'*66}")
            print(f"  mAP @ IoU=0.5:        {mAP:.2%}")
            print(f"\n  Per-class Average Precision:")
            for class_name, stats in class_stats.items():
                print(f"    {class_name:25s}: {stats['ap']:.2%} "
                      f"({stats['predictions']} pred / {stats['ground_truths']} gt)")

            # Save metrics to history
            if 'mAP' not in history:
                history['mAP'] = []
                history['mAP_epochs'] = []
                for class_name in class_names:
                    history[f'AP_{class_name}'] = []

            history['mAP'].append(mAP)
            history['mAP_epochs'].append(epoch + 1)
            for class_name, stats in class_stats.items():
                history[f'AP_{class_name}'].append(stats['ap'])

        print(f"{'='*70}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            },
            is_best=is_best,
            output_dir=output_dir,
            filename=f'checkpoint_epoch_{epoch + 1}.pth'
        )

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dental disease detection model')

    # Data parameters
    parser.add_argument('--train_img_dir', type=str,
                        default='data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/train',
                        help='Training images directory')
    parser.add_argument('--train_label_dir', type=str,
                        default='data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/train',
                        help='Training labels directory')
    parser.add_argument('--val_img_dir', type=str,
                        default='data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/images/val',
                        help='Validation images directory')
    parser.add_argument('--val_label_dir', type=str,
                        default='data/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data/labels/val',
                        help='Validation labels directory')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='fasterrcnn',
                        choices=['fasterrcnn', 'retinanet', 'fcos'],
                        help='Detection model type')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of detection classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'reduce_on_plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')

    # Accuracy metrics parameters
    parser.add_argument('--compute_map_every', type=int, default=5,
                        help='Compute mAP every N epochs (0 to disable, default: 5)')
    parser.add_argument('--confidence_threshold', type=float, default=0.01,
                        help='Confidence threshold for predictions during mAP computation (default: 0.01 for early training)')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'{args.model_type}_{args.backbone}_{timestamp}')

    train(args)
