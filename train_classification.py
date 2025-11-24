import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.classification.dataset import DentalClassificationDataset, get_classification_train_transforms, get_classification_val_transforms
from src.classification.model import get_classification_model


def setup_flash_attention():
    """
    Setup optimized attention mechanisms (Flash Attention if available).
    PyTorch 2.0+ automatically uses Flash Attention via SDPA when available.
    """
    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])

    if torch_version >= (2, 0):
        # PyTorch 2.0+ has built-in SDPA which can use Flash Attention
        try:
            # Enable optimized attention backends
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("✓ Optimized attention enabled (Flash Attention via SDPA)")
            return True
        except Exception as e:
            print(f"⚠ Flash Attention not available: {e}")
            return False
    else:
        print(f"⚠ PyTorch {torch.__version__} detected. Flash Attention requires 2.0+")
        print("  Upgrade with: pip install torch>=2.0.0")
        return False


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, grad_clip=None):
    """Train for one epoch with optional mixed precision and gradient clipping"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]")

    all_preds = []
    all_labels = []

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

        # Store for metrics
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, class_names):
    """Validate model with probability statistics"""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_preds = []
    all_labels = []
    all_probs = []  # Store max probabilities (confidence scores)
    all_full_probs = []  # Store full probability distributions

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [VAL]")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)

        # Calculate accuracy
        acc = (preds == labels).float().mean()

        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

        # Store for metrics
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(max_probs.cpu().numpy())
        all_full_probs.extend(probs.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'val_loss': f'{loss_meter.avg:.4f}',
            'val_acc': f'{acc_meter.avg:.4f}'
        })

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_full_probs = np.array(all_full_probs)

    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Confidence statistics
    correct_mask = all_preds == all_labels
    correct_confidence = all_probs[correct_mask].mean() if correct_mask.sum() > 0 else 0.0
    incorrect_confidence = all_probs[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0.0

    # Per-class metrics
    per_class_acc = []
    per_class_confidence = []
    for class_idx in range(len(class_names)):
        class_mask = all_labels == class_idx
        if class_mask.sum() > 0:
            class_preds = all_preds[class_mask]
            class_labels = all_labels[class_mask]
            class_probs = all_probs[class_mask]

            class_acc = (class_preds == class_labels).mean()
            class_conf = class_probs.mean()

            per_class_acc.append(class_acc)
            per_class_confidence.append(class_conf)
        else:
            per_class_acc.append(0.0)
            per_class_confidence.append(0.0)

    # Package probability statistics
    prob_stats = {
        'overall_confidence': all_probs.mean(),
        'correct_confidence': correct_confidence,
        'incorrect_confidence': incorrect_confidence,
        'per_class_confidence': per_class_confidence
    }

    return loss_meter.avg, acc_meter.avg, precision, recall, f1, per_class_acc, prob_stats


def save_checkpoint(state, output_dir, filename='checkpoint.pth'):
    """Save checkpoint"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    torch.save(state, filepath)


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup Flash Attention if requested
    if args.use_flash_attention:
        flash_enabled = setup_flash_attention()
        if flash_enabled and 'vit' in args.backbone.lower():
            print("  → Flash Attention will accelerate ViT transformer layers")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create datasets
    print("Creating datasets...")
    train_dataset = DentalClassificationDataset(
        root_dir=args.data_dir,
        transform=get_classification_train_transforms(img_size=args.img_size),
        split='train',
        test_size=args.test_size,
        val_size=args.val_size
    )

    val_dataset = DentalClassificationDataset(
        root_dir=args.data_dir,
        transform=get_classification_val_transforms(img_size=args.img_size),
        split='val',
        test_size=args.test_size,
        val_size=args.val_size
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")

    # Print class distribution
    print("\nTraining class distribution:")
    for class_name, count in train_dataset.get_class_distribution().items():
        print(f"  {class_name}: {count} images")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model with backbone-specific dropout
    print(f"\nCreating model: {args.backbone}")

    # Auto-adjust dropout for different backbones
    if args.dropout is None:
        if 'vit' in args.backbone or 'deit' in args.backbone:
            dropout = 0.1  # ViT/DeiT works better with lower dropout
        elif 'swin' in args.backbone:
            dropout = 0.1  # Swin Transformer also uses low dropout
        elif 'maxvit' in args.backbone:
            dropout = 0.15  # MaxViT uses slightly higher
        elif 'convnext' in args.backbone:
            dropout = 0.2
        else:
            dropout = 0.5  # Default for CNNs
        print(f"Auto-adjusted dropout: {dropout} (optimal for {args.backbone})")
    else:
        dropout = args.dropout
        print(f"Using specified dropout: {dropout}")

    model = get_classification_model(
        num_classes=train_dataset.num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=dropout
    )
    model.to(device)

    # Loss function (with class weights for imbalanced data)
    if args.use_class_weights:
        class_dist = train_dataset.get_class_distribution()
        total_samples = sum(class_dist.values())
        # Calculate weights, avoid division by zero
        class_weights = []
        for name in train_dataset.class_names:
            count = class_dist[name]
            if count > 0:
                class_weights.append(total_samples / count)
            else:
                class_weights.append(0.0)  # Classes with 0 samples get 0 weight
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Scheduler with optional warmup
    if args.warmup_epochs > 0:
        print(f"\nUsing learning rate warmup for {args.warmup_epochs} epochs")
        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs
        )

        # Create main scheduler
        if args.scheduler == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.warmup_epochs,
                eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_gamma
            )
        else:
            main_scheduler = None

        # Combine warmup + main scheduler
        if main_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )
        else:
            scheduler = warmup_scheduler
    else:
        # No warmup, just main scheduler
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_gamma
            )
        else:
            scheduler = None

    # Mixed precision scaler
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = GradScaler()
        print(f"\nMixed precision training enabled (2x speedup expected)")
    elif args.use_amp:
        print(f"\nWarning: Mixed precision requested but CUDA not available, using FP32")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'lr': []
    }

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1,
            scaler=scaler, grad_clip=args.grad_clip
        )

        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, per_class_acc, prob_stats = validate(
            model, val_loader, criterion, device, epoch + 1, train_dataset.class_names
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Print summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs} Summary:")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Precision:  {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

        # Confidence statistics
        print(f"\n  Confidence Statistics:")
        print(f"    Overall Avg:     {prob_stats['overall_confidence']:.2%}")
        print(f"    Correct Preds:   {prob_stats['correct_confidence']:.2%}")
        print(f"    Incorrect Preds: {prob_stats['incorrect_confidence']:.2%}")

        print(f"\n  Per-class Accuracy & Confidence:")
        for class_name, class_acc, class_conf in zip(train_dataset.class_names, per_class_acc, prob_stats['per_class_confidence']):
            print(f"    {class_name:25s}: {class_acc:.2%} (conf: {class_conf:.2%})")
        print(f"{'='*80}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
        }, output_dir, filename=f'checkpoint_epoch_{epoch+1}.pth')

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
            }, output_dir, filename='checkpoint_best.pth')
            print(f"  Best model saved! (Val Acc: {best_val_acc:.4f})")

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dental disease classification model')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root directory containing class folders')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set proportion')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3',
                                 'vit_tiny_patch16_224', 'vit_small_patch16_224',
                                 'vit_base_patch16_224', 'vit_base_patch32_224',
                                 'vit_base_patch16_384', 'vit_base_patch32_384',
                                 'vit_large_patch14_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
                                 'vit_large_patch16_384',
                                 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
                                 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224',
                                 'deit3_small_patch16_224', 'deit3_base_patch16_224', 'deit3_large_patch16_224',
                                 'maxvit_tiny_tf_224', 'maxvit_small_tf_224', 'maxvit_base_tf_224',
                                 'convnext_tiny', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (auto-adjusted if not specified: ViT=0.1, ConvNeXt=0.2, others=0.5)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Step size for StepLR')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for StepLR')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')

    # Optimization enhancements
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision training (2x speedup, recommended for ViT)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs (recommended: 5 for ViT, 0 for others)')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient clipping max norm (recommended: 1.0 for ViT, None for others)')
    parser.add_argument('--use_flash_attention', action='store_true', default=False,
                        help='Enable Flash Attention if available (2-4x faster attention, requires PyTorch 2.0+)')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='classification_outputs',
                        help='Output directory')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'{args.backbone}_{timestamp}')

    train(args)
