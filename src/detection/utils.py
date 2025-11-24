import os
import torch
import shutil
from pathlib import Path


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


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """Save checkpoint to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    torch.save(state, filepath)

    # Save as latest
    latest_path = output_dir / 'checkpoint_latest.pth'
    shutil.copyfile(filepath, latest_path)

    # Save best model
    if is_best:
        best_path = output_dir / 'checkpoint_best.pth'
        shutil.copyfile(filepath, best_path)
        print(f"Best model saved to: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint from disk"""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return epoch, best_val_loss


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU score
    """
    # Get intersection coordinates
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    if x_max < x_min or y_max < y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)

    return iou


def calculate_map(predictions, targets, iou_threshold=0.5, num_classes=4):
    """
    Calculate mean Average Precision (mAP) for object detection.

    Args:
        predictions (list): List of predictions [{boxes, labels, scores}, ...]
        targets (list): List of targets [{boxes, labels}, ...]
        iou_threshold (float): IoU threshold for considering a detection as correct
        num_classes (int): Number of classes

    Returns:
        mAP score and per-class AP scores
    """
    # Initialize per-class metrics
    class_aps = []

    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        # NOTE: Labels are 1-indexed (1-4), so add 1 to class_id
        actual_class_id = class_id + 1
        class_predictions = []
        class_targets = []

        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Filter predictions for this class
            class_mask = pred['labels'] == actual_class_id
            if class_mask.sum() > 0:
                class_pred_boxes = pred['boxes'][class_mask]
                class_pred_scores = pred['scores'][class_mask]

                for box, score in zip(class_pred_boxes, class_pred_scores):
                    class_predictions.append({
                        'box': box.cpu().numpy(),
                        'score': score.cpu().item(),
                        'image_idx': img_idx  # Track which image this prediction belongs to
                    })

            # Filter targets for this class
            target_class_mask = target['labels'] == actual_class_id
            if target_class_mask.sum() > 0:
                class_target_boxes = target['boxes'][target_class_mask]
                class_targets.append({
                    'boxes': class_target_boxes.cpu().numpy(),
                    'detected': [False] * len(class_target_boxes)
                })
            else:
                class_targets.append({
                    'boxes': [],
                    'detected': []
                })

        # Calculate AP for this class
        if len(class_predictions) == 0:
            class_aps.append(0.0)
            continue

        # Sort predictions by score (descending)
        class_predictions.sort(key=lambda x: x['score'], reverse=True)

        # Calculate TP and FP
        tp = []
        fp = []

        for pred in class_predictions:
            pred_box = pred['box']
            img_idx = pred['image_idx']  # Get the correct image index for this prediction
            max_iou = 0.0
            max_idx = -1

            # Find best matching ground truth in the SAME image
            if img_idx < len(class_targets) and len(class_targets[img_idx]['boxes']) > 0:
                for gt_idx, gt_box in enumerate(class_targets[img_idx]['boxes']):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = gt_idx

            # Check if detection is correct
            if max_iou >= iou_threshold and max_idx >= 0 and not class_targets[img_idx]['detected'][max_idx]:
                tp.append(1)
                fp.append(0)
                class_targets[img_idx]['detected'][max_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        # Calculate precision and recall
        tp_cumsum = torch.tensor(tp).cumsum(0)
        fp_cumsum = torch.tensor(fp).cumsum(0)

        # Count total ground truths across ALL images
        total_gt = sum(len(target['boxes']) for target in class_targets)

        recalls = tp_cumsum / (total_gt + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            if torch.sum(recalls >= t) == 0:
                p = 0
            else:
                p = torch.max(precisions[recalls >= t])
            ap += p / 11

        class_aps.append(ap.item())

    # Calculate mAP
    mAP = sum(class_aps) / len(class_aps)

    return mAP, class_aps


def denormalize_boxes(boxes, img_width, img_height):
    """
    Convert normalized boxes [0, 1] to pixel coordinates.

    Args:
        boxes: Tensor of shape (N, 4) with normalized coordinates
        img_width, img_height: Image dimensions

    Returns:
        Boxes in pixel coordinates
    """
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= img_width
    boxes[:, [1, 3]] *= img_height
    return boxes


def normalize_boxes(boxes, img_width, img_height):
    """
    Convert pixel coordinates to normalized boxes [0, 1].

    Args:
        boxes: Tensor of shape (N, 4) with pixel coordinates
        img_width, img_height: Image dimensions

    Returns:
        Normalized boxes
    """
    boxes = boxes.clone()
    boxes[:, [0, 2]] /= img_width
    boxes[:, [1, 3]] /= img_height
    return boxes


def filter_predictions(predictions, confidence_threshold=0.5, nms_threshold=0.5):
    """
    Filter predictions by confidence and apply NMS.

    Args:
        predictions (list): List of prediction dictionaries
        confidence_threshold (float): Confidence threshold
        nms_threshold (float): NMS IoU threshold

    Returns:
        Filtered predictions
    """
    from torchvision.ops import nms

    filtered_predictions = []

    for pred in predictions:
        # Filter by confidence
        conf_mask = pred['scores'] >= confidence_threshold
        boxes = pred['boxes'][conf_mask]
        scores = pred['scores'][conf_mask]
        labels = pred['labels'][conf_mask]

        # Apply NMS
        if len(boxes) > 0:
            keep_indices = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        filtered_predictions.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })

    return filtered_predictions
