"""
Simple prediction script for dental disease classification.
Usage:
    python predict.py --checkpoint model.pth --image tooth.jpg
    python predict.py --checkpoint model.pth --image tooth.jpg --show
"""

import sys
import torch
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.classification.model import get_classification_model
from src.classification.dataset import get_classification_val_transforms


class DentalPredictor:
    """Simple predictor for dental disease classification"""

    def __init__(self, checkpoint_path, backbone='vit_base_patch16_224', device=None):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to trained model checkpoint
            backbone: Model architecture (default: vit_base_patch16_224)
            device: Device to use (default: auto-detect)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = backbone

        # Class names
        self.class_names = [
            'Calculus',
            'Caries',
            'Gingivitis',
            'Hypodontia',
            'Mouth Ulcer',
            'Tooth Discoloration'
        ]

        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = get_classification_model(
            num_classes=len(self.class_names),
            backbone=backbone,
            pretrained=False
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = get_classification_val_transforms(img_size=224)

        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Backbone: {backbone}")
        print(f"✓ Classes: {len(self.class_names)}")

    @torch.no_grad()
    def predict(self, image_path, ood_threshold=0.5, top_k=3):
        """
        Predict dental disease from image.

        Args:
            image_path: Path to image file
            ood_threshold: Confidence threshold for OOD detection
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and confidence scores
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # Transform
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Predict
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

        # Get top predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))

        max_confidence = top_probs[0].item()
        is_ood = max_confidence < ood_threshold

        # Format results
        results = {
            'predicted_class': 'Unknown/Healthy' if is_ood else self.class_names[top_indices[0].item()],
            'confidence': max_confidence,
            'is_healthy': is_ood,
            'top_predictions': [
                {
                    'class': self.class_names[idx.item()],
                    'confidence': prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
            'all_probabilities': {
                self.class_names[i]: probs[i].item()
                for i in range(len(self.class_names))
            }
        }

        return results

    def predict_and_visualize(self, image_path, ood_threshold=0.5, save_path=None):
        """
        Predict and create visualization.

        Args:
            image_path: Path to image
            ood_threshold: OOD threshold
            save_path: Optional path to save visualization

        Returns:
            Prediction results
        """
        results = self.predict(image_path, ood_threshold)

        # Load original image
        image = Image.open(image_path).convert('RGB')

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Show image
        ax1.imshow(image)
        ax1.axis('off')

        # Add prediction box
        prediction = results['predicted_class']
        confidence = results['confidence']

        if results['is_healthy']:
            color = 'green'
            title = f"✓ HEALTHY\nConfidence: {confidence:.1%}"
        else:
            color = 'red'
            title = f"⚠ {prediction}\nConfidence: {confidence:.1%}"

        ax1.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)

        # Show top predictions bar chart
        top_preds = results['top_predictions']
        classes = [p['class'] for p in top_preds]
        confidences = [p['confidence'] for p in top_preds]

        colors_bar = ['red' if c == prediction and not results['is_healthy'] else 'gray' for c in classes]

        ax2.barh(classes, confidences, color=colors_bar, alpha=0.7)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)

        # Add confidence values
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            ax2.text(conf + 0.02, i, f'{conf:.1%}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")

        plt.show()

        return results


def main():
    parser = argparse.ArgumentParser(description='Predict dental disease from image')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_224',
                        help='Model backbone (default: vit_base_patch16_224)')
    parser.add_argument('--ood_threshold', type=float, default=0.5,
                        help='OOD threshold for healthy detection (default: 0.5)')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to file')

    args = parser.parse_args()

    # Create predictor
    predictor = DentalPredictor(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone
    )

    # Predict
    if args.show or args.save:
        results = predictor.predict_and_visualize(
            args.image,
            ood_threshold=args.ood_threshold,
            save_path=args.save
        )
    else:
        results = predictor.predict(args.image, ood_threshold=args.ood_threshold)

    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {args.image}")
    print(f"Prediction: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.2%}")

    if results['is_healthy']:
        print(f"\n✓ HEALTHY - No disease detected")
    else:
        print(f"\n⚠ DISEASE DETECTED")

    print(f"\nTop Predictions:")
    for i, pred in enumerate(results['top_predictions'], 1):
        print(f"  {i}. {pred['class']:20s}: {pred['confidence']:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
