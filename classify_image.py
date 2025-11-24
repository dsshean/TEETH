import sys
import torch
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.classification.model import get_classification_model
from src.classification.dataset import get_classification_val_transforms


class DentalImageClassifier:
    """Class for classifying dental images"""

    def __init__(self, checkpoint_path, backbone='resnet50', num_classes=6, device=None, ood_threshold=0.5):
        """
        Initialize classifier.

        Args:
            checkpoint_path: Path to model checkpoint
            backbone: Backbone architecture
            num_classes: Number of classes
            device: Device to run on
            ood_threshold: Confidence threshold for out-of-distribution detection (0.0-1.0)
                          If max confidence < threshold, classify as "Unknown/Healthy"
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ood_threshold = ood_threshold

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
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = get_classification_val_transforms(img_size=224)

        print(f"Model loaded successfully on {self.device}")

    @torch.no_grad()
    def predict(self, image_path):
        """
        Classify a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Get predictions
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

        # Get top predictions
        top5_probs, top5_indices = torch.topk(probs, k=min(5, len(self.class_names)))

        max_confidence = top5_probs[0].item()

        # Out-of-distribution detection
        is_ood = max_confidence < self.ood_threshold

        results = {
            'predicted_class': 'Unknown/Healthy' if is_ood else self.class_names[top5_indices[0].item()],
            'confidence': max_confidence,
            'is_out_of_distribution': is_ood,
            'ood_threshold': self.ood_threshold,
            'top5': [
                {
                    'class': self.class_names[idx.item()],
                    'confidence': prob.item()
                }
                for prob, idx in zip(top5_probs, top5_indices)
            ],
            'all_probabilities': {
                self.class_names[i]: probs[i].item()
                for i in range(len(self.class_names))
            }
        }

        return results


def main():
    parser = argparse.ArgumentParser(description='Classify dental images')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes')
    parser.add_argument('--ood_threshold', type=float, default=0.5,
                        help='Out-of-distribution threshold (0.0-1.0). Images with max confidence below this are classified as Unknown/Healthy')

    args = parser.parse_args()

    # Create classifier
    classifier = DentalImageClassifier(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        num_classes=args.num_classes,
        ood_threshold=args.ood_threshold
    )

    # Classify image
    results = classifier.predict(args.image)

    # Print results
    print(f"\n{'='*60}")
    print(f"Classification Results for: {args.image}")
    print(f"{'='*60}")
    print(f"Predicted Class: {results['predicted_class']}")
    print(f"Confidence:      {results['confidence']:.2%}")

    if results['is_out_of_distribution']:
        print(f"\n⚠️  OUT-OF-DISTRIBUTION DETECTED")
        print(f"Max confidence ({results['confidence']:.2%}) is below threshold ({results['ood_threshold']:.2%})")
        print(f"This image likely does not match any trained disease classes.")
        print(f"Possible reasons: Healthy teeth, different condition, or poor image quality.")

    print(f"\nTop 5 Disease Predictions:")
    for i, pred in enumerate(results['top5'], 1):
        print(f"  {i}. {pred['class']:25s}: {pred['confidence']:.2%}")
    print(f"{'='*60}")
    print(f"\nNote: Adjust --ood_threshold (currently {results['ood_threshold']:.2f}) to tune sensitivity.")
    print(f"      Lower threshold = more likely to classify as disease")
    print(f"      Higher threshold = more likely to classify as Unknown/Healthy")


if __name__ == "__main__":
    main()
