# Dental Disease Detection with PyTorch

A complete PyTorch implementation for detecting dental diseases using object detection models with YOLO-format annotations.

## Overview

This project implements a deep learning pipeline for detecting and localizing dental diseases in clinical images. The system uses state-of-the-art object detection models (Faster R-CNN, RetinaNet, FCOS) to identify and locate:

- **Caries** (cavities)
- **Ulcer** (mouth ulcers)
- **Tooth Discoloration**
- **Gingivitis** (gum inflammation)

## Dataset Structure

The project uses YOLO-format annotations with the following structure:

```
data/
├── Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/
│   └── Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/
│       └── Data/
│           ├── images/
│           │   ├── train/  (864 images)
│           │   └── val/    (49 images)
│           ├── labels/
│           │   ├── train/  (YOLO .txt annotations)
│           │   └── val/    (YOLO .txt annotations)
│           └── data.yaml
```

### YOLO Annotation Format

Each `.txt` file contains one line per bounding box:

```
class_id x_center y_center width height
```

All coordinates are normalized to [0, 1] range.

**Classes:**

- 0: Caries
- 1: Ulcer
- 2: Tooth Discoloration
- 3: Gingivitis

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 8GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone <repository_url>
cd TEETH

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (makes src module importable)
pip install -e .
```

**Note:** The `-e` flag installs the package in "editable" mode, allowing you to modify the source code without reinstalling. This is required for the `src.detection` and `src.classification` imports to work properly.

## Project Structure

```
TEETH/
├── src/
│   ├── __init__.py
│   ├── detection/          # Object detection module
│   │   ├── __init__.py
│   │   ├── dataset.py     # Detection dataset with augmentations
│   │   ├── model.py       # Detection model architectures
│   │   └── utils.py       # Detection utilities (IoU, mAP, checkpoints)
│   └── classification/     # Image classification module
│       ├── __init__.py
│       ├── dataset.py     # Classification dataset
│       └── model.py       # Classification models
├── scripts/
│   ├── train_detection.py         # Train detection models
│   ├── evaluate_detection.py      # Evaluate detection models
│   ├── inference_detection.py     # Run detection inference
│   ├── train_classification.py    # Train classification models
│   ├── classify_image.py          # Classify single images
│   ├── predict.py                 # Prediction with visualization
│   └── monitor_training.py        # Monitor training progress
├── tests/
│   ├── __init__.py
│   └── test_dataset.py    # Dataset tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### 1. Training

Train a Faster R-CNN model with ResNet-50 backbone:

```bash
python scripts/train_detection.py \
    --model_type fasterrcnn \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.005 \
    --optimizer sgd \
    --scheduler cosine \
    --img_size 640
```

**Key Arguments:**

- `--model_type`: Detection model (`fasterrcnn`, `retinanet`, `fcos`)
- `--backbone`: Backbone network (`resnet50`, `resnet101`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.005)
- `--optimizer`: Optimizer (`sgd`, `adam`, `adamw`)
- `--scheduler`: LR scheduler (`cosine`, `step`, `reduce_on_plateau`)
- `--img_size`: Input image size (default: 640)
- `--output_dir`: Output directory (default: `outputs/`)
- `--resume`: Resume from checkpoint path

**Training Output:**

Models are saved to `outputs/<model>_<backbone>_<timestamp>/`:

- `checkpoint_latest.pth`: Latest checkpoint
- `checkpoint_best.pth`: Best model by validation loss
- `checkpoint_epoch_N.pth`: Checkpoint for each epoch
- `history.json`: Training history (losses, learning rates)
- `config.json`: Training configuration

### 2. Evaluation

Evaluate a trained model on the validation set:

```bash
python scripts/evaluate_detection.py \
    --checkpoint outputs/<model_dir>/checkpoint_best.pth \
    --model_type fasterrcnn \
    --backbone resnet50 \
    --confidence_threshold 0.5 \
    --iou_threshold 0.5 \
    --visualize \
    --num_visualizations 10
```

**Key Arguments:**

- `--checkpoint`: Path to model checkpoint (required)
- `--confidence_threshold`: Confidence threshold for predictions (default: 0.5)
- `--iou_threshold`: IoU threshold for mAP calculation (default: 0.5)
- `--visualize`: Generate prediction visualizations
- `--num_visualizations`: Number of samples to visualize (default: 10)
- `--output_dir`: Output directory (default: `evaluation_results/`)

**Evaluation Metrics:**

- **mAP (mean Average Precision)**: Overall detection performance
- **Per-class AP**: Average Precision for each disease class
- **Class statistics**: Number of predictions vs ground truths

### 3. Inference

Run inference on new images:

**Single Image:**

```bash
python scripts/inference_detection.py \
    --checkpoint outputs/<model_dir>/checkpoint_best.pth \
    --input_image path/to/image.jpg \
    --confidence_threshold 0.5 \
    --show
```

**Batch Processing:**

```bash
python scripts/inference_detection.py \
    --checkpoint outputs/<model_dir>/checkpoint_best.pth \
    --input_dir path/to/images/ \
    --output_dir inference_results/ \
    --confidence_threshold 0.5
```

**Key Arguments:**

- `--checkpoint`: Path to model checkpoint (required)
- `--input_image`: Path to single image
- `--input_dir`: Directory containing images (for batch processing)
- `--confidence_threshold`: Confidence threshold (default: 0.5)
- `--output_dir`: Output directory (default: `inference_results/`)
- `--show`: Display visualization (single image only)

**Output:**

For each image, generates:

- Annotated image with bounding boxes and labels
- JSON file with detection results

## Model Architectures

### 1. Faster R-CNN

Two-stage detector with Region Proposal Network (RPN).

**Pros:**

- High accuracy
- Good for small objects
- Robust to occlusions

**Cons:**

- Slower inference
- More memory intensive

### 2. RetinaNet

Single-stage detector with Feature Pyramid Network (FPN).

**Pros:**

- Balanced speed/accuracy
- Handles scale variation well
- Focal loss for class imbalance

**Cons:**

- More hyperparameters to tune

### 3. FCOS

Fully Convolutional One-Stage detector (anchor-free).

**Pros:**

- Fast inference
- No anchor tuning needed
- Simpler architecture

**Cons:**

- May struggle with overlapping objects

## Data Augmentation

The training pipeline includes extensive augmentations via Albumentations:

- **Geometric**: Horizontal flip, random crop
- **Color**: ColorJitter, brightness/contrast adjustment, HSV shifts
- **Quality**: Gaussian blur
- **Normalization**: ImageNet mean/std

All augmentations preserve bounding box annotations.

## Evaluation Metrics

### mAP (mean Average Precision)

- Calculated at IoU threshold (default: 0.5)
- Average of per-class AP scores
- Standard metric for object detection

### IoU (Intersection over Union)

- Measures overlap between predicted and ground truth boxes
- Used to determine true positives
- Formula: `IoU = (Area of Overlap) / (Area of Union)`

### Per-class Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **AP**: Area under precision-recall curve

## Training Tips

### For Best Results:

1. **Learning Rate**: Start with 0.005 for SGD, 0.001 for Adam
2. **Batch Size**: Use largest batch that fits in GPU memory (4-8 recommended)
3. **Epochs**: Train for 50-100 epochs with early stopping
4. **Augmentation**: Enabled by default, reduces overfitting
5. **Backbone**: ResNet-50 for speed, ResNet-101 for accuracy

### Handling Class Imbalance:

The dataset has imbalanced classes. Consider:

- Using weighted loss (implemented in models)
- Oversampling minority classes
- Focal loss (available in RetinaNet)

### GPU Memory Issues:

If you encounter OOM errors:

- Reduce batch size (`--batch_size 2`)
- Reduce image size (`--img_size 512`)
- Use mixed precision training (add to train.py)

## Results Tracking

Training generates several files for monitoring:

1. **history.json**: Loss curves and learning rates
2. **Checkpoints**: Model weights at each epoch
3. **Visualizations**: Sample predictions on validation set

## Example Workflow

Complete workflow from training to inference:

```bash
# 1. Train model
python scripts/train_detection.py --model_type fasterrcnn --epochs 50 --batch_size 4

# 2. Evaluate model
python scripts/evaluate_detection.py \
    --checkpoint outputs/fasterrcnn_resnet50_<timestamp>/checkpoint_best.pth \
    --visualize

# 3. Run inference
python scripts/inference_detection.py \
    --checkpoint outputs/fasterrcnn_resnet50_<timestamp>/checkpoint_best.pth \
    --input_dir test_images/ \
    --output_dir results/
```

## Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**

```
Solution: Reduce batch_size or img_size
python scripts/train_detection.py --batch_size 2 --img_size 512
```

**2. No detections in output**

```
Solution: Lower confidence threshold
python scripts/inference_detection.py --confidence_threshold 0.3
```

**3. Low mAP scores**

```
Solutions:
- Train longer (more epochs)
- Use data augmentation
- Try different model architecture
- Check annotation quality
```

**4. Slow training**

```
Solutions:
- Reduce batch_size but increase accumulation steps
- Use smaller image size
- Use faster model (RetinaNet vs Faster R-CNN)
```

## Advanced Features

### Custom Dataset

To use your own YOLO-format dataset:

1. Update paths in training script
2. Modify class names in `dataset.py`
3. Update `num_classes` argument

### Transfer Learning

Models use ImageNet pretrained backbones by default (`--pretrained` flag).

### Checkpoint Resume

Resume interrupted training:

```bash
python scripts/train_detection.py --resume outputs/<model_dir>/checkpoint_latest.pth
```

## Citation

If you use this code, please cite:

```bibtex
@software{dental_detection_pytorch,
  title = {Dental Disease Detection with PyTorch},
  year = {2024},
  author = {Doug Shean},
  url = {https://github.com/dsshean/TEETH}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- PyTorch and torchvision for detection models
- Albumentations for data augmentation
- YOLO annotation format for efficient labeling

## Classification Pipeline (Image-Level Classification)

In addition to object detection, this project includes a **classification-only pipeline** for identifying dental diseases at the image level (without bounding boxes).

### Dataset

Uses all 6 class folders from the `data/` directory:

- **Calculus** (1,296 images)
- **Data caries** (219 images)
- **Gingivitis** (2,340 images)
- **Hypodontia** (342 images)
- **Mouth Ulcer** (265 images)
- **Tooth Discoloration** (183 images)

Total: ~4,645 images with automatic 70/10/20 train/val/test split.

### Supported Models

The classification pipeline supports multiple state-of-the-art architectures:

1. **Swin Transformer (Tiny/Small/Base/Large)** - Hierarchical transformer, best for medical imaging (RECOMMENDED)
2. **Vision Transformer (ViT)** - Multiple sizes and resolutions (Base/Large, 224/384)
3. **DeiT3 (Small/Base/Large)** - Data-efficient image transformers
4. **MaxViT (Tiny/Small/Base)** - Hybrid CNN+Transformer architecture
5. **ConvNeXtV2 (Tiny/Base/Large)** - Modern CNN with transformer design principles
6. **ResNet-50/101** - CNN baseline, fast and reliable
7. **EfficientNet-B0/B3** - Efficient architecture, good accuracy/speed trade-off
8. **ConvNeXt-Tiny** - Original ConvNeXt architecture

### Training Classification Models

#### Swin Transformer (RECOMMENDED for Dental Imaging)

```bash
python scripts/train_classification.py \
    --backbone swin_base_patch4_window7_224 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --optimizer adamw \
    --weight_decay 0.05 \
    --use_amp \
    --warmup_epochs 5 \
    --grad_clip 5.0
```

**Swin Advantages:**

- ✅ Hierarchical architecture captures multi-scale features (ideal for dental details)
- ✅ Local window attention reduces computational cost vs standard ViT
- ✅ State-of-the-art performance on medical imaging tasks
- ✅ Better inductive bias for image data compared to pure transformers
- ✅ Expected accuracy: **93-95%** on validation set

**Why Swin for Dental Imaging:**

- Hierarchical feature maps capture both fine details (cracks, discoloration) and larger patterns (inflammation, calculus buildup)
- Shifted window approach provides better spatial understanding than ViT's fixed patches
- Proven track record in medical image analysis (CT scans, X-rays, fundus images)

#### ResNet-50 (Good baseline)

```bash
python scripts/train_classification.py \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --optimizer adam
```

#### Vision Transformer (ViT) - Alternative transformer option

```bash
# Optimal ViT training with all enhancements
python scripts/train_classification.py \
    --backbone vit_base_patch16_224 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --optimizer adam \
    --img_size 224 \
    --use_amp \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    --use_flash_attention
```

**ViT Advantages:**

- ✅ Better at capturing global tooth structure patterns
- ✅ Attention mechanism provides interpretability
- ✅ Superior performance on fine-grained details (cracks, discoloration)

**ViT Optimizations (Included Above):**

- **Mixed Precision (`--use_amp`)**: 2x faster training, same accuracy
- **Warmup (`--warmup_epochs 5`)**: Stabilizes early training for ViT
- **Gradient Clipping (`--grad_clip 1.0`)**: Prevents gradient explosions
- **Auto Dropout (0.1)**: Automatically adjusted for ViT (vs 0.5 for CNNs)
- **Flash Attention (`--use_flash_attention`)**: 2-4x faster attention if available (PyTorch 2.0+)

**ViT Considerations:**

- Requires lower learning rate (0.0001 vs 0.001)
- Needs smaller batch size (16 vs 32) due to memory
- With mixed precision: training speed comparable to ResNet!
- May need 50-70 epochs to fully converge

#### EfficientNet-B0 (Best speed/accuracy trade-off)

```bash
python scripts/train_classification.py \
    --backbone efficientnet_b0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --optimizer adam
```

#### ConvNeXt (Modern CNN alternative)

```bash
python scripts/train_classification.py \
    --backbone convnext_tiny \
    --epochs 50 \
    --batch_size 24 \
    --lr 0.0005 \
    --optimizer adamw
```

### Classification Training Arguments

```bash
# Data parameters
--data_dir DATA_DIR              # Root directory with class folders (default: data)
--test_size 0.2                  # Test set proportion (default: 20%)
--val_size 0.1                   # Validation set proportion (default: 10%)

# Model parameters
--backbone MODEL                 # resnet50, resnet101, efficientnet_b0, efficientnet_b3,
                                 # vit_base_patch16_224, convnext_tiny
--dropout 0.5                    # Dropout rate (default: 0.5)

# Training parameters
--epochs 50                      # Number of epochs
--batch_size 32                  # Batch size (reduce for ViT/larger models)
--img_size 224                   # Input image size
--lr 0.001                       # Learning rate (use 0.0001 for ViT)
--optimizer adam                 # sgd, adam, adamw
--scheduler cosine               # cosine, step, none
--use_class_weights              # Handle imbalanced classes (default: True)

# Output
--output_dir OUTPUT_DIR          # Output directory (default: classification_outputs)
--num_workers 4                  # DataLoader workers
```

### Classification Inference

Classify a single image:

```bash
python scripts/classify_image.py \
    --checkpoint classification_outputs/resnet50_<timestamp>/checkpoint_best.pth \
    --image path/to/tooth.jpg \
    --backbone resnet50
```

**For Swin Transformer:**

```bash
python scripts/classify_image.py \
    --checkpoint classification_outputs/swin_base_patch4_window7_224_<timestamp>/checkpoint_best.pth \
    --image path/to/tooth.jpg \
    --backbone swin_base_patch4_window7_224
```

**For Vision Transformer:**

```bash
python scripts/classify_image.py \
    --checkpoint classification_outputs/vit_base_patch16_224_<timestamp>/checkpoint_best.pth \
    --image path/to/tooth.jpg \
    --backbone vit_base_patch16_224
```

### Out-of-Distribution Detection (Healthy Teeth)

The classifier includes confidence-based OOD detection to identify images that don't match any trained disease class (e.g., healthy teeth):

```bash
# Default threshold (50% confidence)
python scripts/classify_image.py \
    --checkpoint model.pth \
    --image tooth.jpg \
    --ood_threshold 0.5
```

**Tuning the threshold:**

- `--ood_threshold 0.3`: More sensitive, fewer "Unknown" classifications
- `--ood_threshold 0.5`: Balanced (default)
- `--ood_threshold 0.7`: More conservative, requires high confidence for disease

**Example output:**

```
Predicted Class: Unknown/Healthy
Confidence:      23%
⚠️  OUT-OF-DISTRIBUTION DETECTED
This image likely does not match any trained disease classes.
```

### Classification vs Detection

| Feature               | Classification         | Object Detection                |
| --------------------- | ---------------------- | ------------------------------- |
| **Output**            | Single label per image | Boxes + labels for each disease |
| **Classes**           | 6 classes              | 4 classes (with boxes)          |
| **Images**            | ~4,645 images          | ~1,493 images                   |
| **Speed**             | Very fast              | Slower                          |
| **Use case**          | "What disease?"        | "Where is each disease?"        |
| **Healthy detection** | ✅ OOD detection       | ❌ Not applicable               |

### Model Comparison

Expected performance after training:

| Model           | Accuracy   | Speed  | Memory | Best For                                 |
| --------------- | ---------- | ------ | ------ | ---------------------------------------- |
| **Swin-Base**   | **93-95%** | Medium | High   | **Medical imaging (RECOMMENDED)**        |
| ViT-Base        | 87-92%     | Slow   | High   | Global patterns, alternative transformer |
| DeiT3-Base      | 88-93%     | Medium | High   | Data-efficient training                  |
| MaxViT-Base     | 90-94%     | Medium | High   | Hybrid CNN+Transformer                   |
| ConvNeXtV2-Base | 89-93%     | Medium | Medium | Modern CNN, efficient                    |
| EfficientNet-B3 | 85-90%     | Fast   | Medium | Speed/accuracy trade-off                 |
| ResNet-50       | 80-85%     | Fast   | Low    | Quick baseline                           |
| ConvNeXt-Tiny   | 84-89%     | Fast   | Low    | Lightweight modern CNN                   |

### Classification Tips

1. **Start with Swin Transformer** for best results on dental imaging (hierarchical features, medical imaging proven)
2. **Try ResNet-50 first** if you want a quick baseline (fast training, good results)
3. **Use ViT or DeiT3** as alternative transformers if Swin doesn't fit your setup
4. **Use class weights** (`--use_class_weights`) to handle imbalanced data
5. **Monitor per-class accuracy** to identify problematic classes
6. **Tune OOD threshold** after training on healthy tooth samples

### Handling Class Imbalance

The classification dataset is highly imbalanced (Gingivitis: 2,340 vs Tooth Discoloration: 183). The pipeline handles this with:

- **Class weights**: Automatically computed and applied to loss function
- **Stratified splitting**: Maintains class proportions across train/val/test
- **Heavy augmentation**: Rotations, flips, color jitter, blur, etc.

### GPU Memory Requirements

| Model           | Batch 32 | Batch 16 | Batch 8 |
| --------------- | -------- | -------- | ------- |
| ResNet-50       | 6GB      | 3GB      | 2GB     |
| EfficientNet-B0 | 5GB      | 3GB      | 2GB     |
| **ViT-Base**    | **12GB** | **6GB**  | **3GB** |
| ConvNeXt-Tiny   | 7GB      | 4GB      | 2GB     |

If you encounter OOM errors with ViT, use `--batch_size 8` or `--batch_size 4`.

### Complete Classification Workflow

```bash
# 1. Train Swin Transformer for best accuracy (RECOMMENDED)
python scripts/train_classification.py \
    --backbone swin_base_patch4_window7_224 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --optimizer adamw \
    --use_amp

# 2. Optional: Train ResNet-50 baseline for comparison
python scripts/train_classification.py --backbone resnet50 --epochs 50 --batch_size 32

# 3. Compare results and select best model

# 4. Test on single image
python scripts/classify_image.py \
    --checkpoint classification_outputs/swin_base_patch4_window7_224_<timestamp>/checkpoint_best.pth \
    --image test_tooth.jpg \
    --backbone swin_base_patch4_window7_224 \
    --ood_threshold 0.5

# 5. Calibrate OOD threshold with healthy tooth images
python scripts/classify_image.py \
    --checkpoint classification_outputs/swin_base_patch4_window7_224_<timestamp>/checkpoint_best.pth \
    --image healthy_tooth.jpg \
    --ood_threshold 0.6
```

================================================================================
Epoch 50/50 Summary:
================================================================================
Train Loss: 0.0953 | Train Acc: 0.9548
Val Loss: 0.1434 | Val Acc: 0.9351
Precision: 0.9459 | Recall: 0.9351 | F1: 0.9362

Confidence Statistics:
Overall Avg: 95.92%
Correct Preds: 97.70%
Incorrect Preds: 70.28%

Per-class Accuracy & Confidence:
Calculus : 90.77% (conf: 84.57%)
Data caries : 99.62% (conf: 99.96%)
Gingivitis : 73.62% (conf: 87.94%)
hypodontia : 99.20% (conf: 99.54%)
Mouth Ulcer : 99.64% (conf: 99.98%)
Tooth Discoloration : 98.51% (conf: 99.45%)
================================================================================
