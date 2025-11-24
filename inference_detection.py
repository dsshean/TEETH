import sys
import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.detection.model import get_model
from src.detection.utils import load_checkpoint, filter_predictions
from src.detection.dataset import get_val_transforms


class DentalDetector:
    """Class for running inference on dental images"""

    def __init__(self, checkpoint_path, model_type='fasterrcnn', backbone='resnet50',
                 num_classes=4, confidence_threshold=0.5, device=None):
        """
        Initialize detector.

        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Model architecture type
            backbone: Backbone network
            num_classes: Number of detection classes
            confidence_threshold: Minimum confidence for detections
            device: Device to run on (cuda/cpu)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Class names and colors
        self.class_names = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']
        self.colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 165, 0)]  # BGR format for OpenCV

        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = get_model(
            num_classes=num_classes,
            model_type=model_type,
            backbone=backbone,
            pretrained=False
        )

        load_checkpoint(checkpoint_path, self.model)
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = get_val_transforms(img_size=640)

        print(f"Model loaded successfully on {self.device}")

    @torch.no_grad()
    def predict(self, image):
        """
        Run detection on a single image.

        Args:
            image: PIL Image or numpy array (RGB)

        Returns:
            Dictionary with boxes, labels, and scores
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Store original size
        orig_width, orig_height = image.size

        # Convert to numpy for transforms
        image_np = np.array(image)

        # Apply transforms
        transformed = self.transform(image=image_np, bboxes=[], labels=[])
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Run inference
        predictions = self.model(image_tensor)[0]

        # Filter by confidence
        conf_mask = predictions['scores'] >= self.confidence_threshold
        boxes = predictions['boxes'][conf_mask].cpu()
        labels = predictions['labels'][conf_mask].cpu()
        scores = predictions['scores'][conf_mask].cpu()

        # Scale boxes back to original image size
        # Boxes are in normalized coordinates [0, 1] after transforms
        h, w = transformed['image'].shape[1:]
        scale_x = orig_width / w
        scale_y = orig_height / h

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        return {
            'boxes': boxes.numpy(),
            'labels': labels.numpy(),
            'scores': scores.numpy()
        }

    def visualize(self, image, predictions, save_path=None, show=True):
        """
        Visualize detections on image.

        Args:
            image: PIL Image or numpy array
            predictions: Dictionary with boxes, labels, scores
            save_path: Optional path to save visualization
            show: Whether to display the image

        Returns:
            Annotated image as numpy array
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert RGB to BGR for OpenCV
        image_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

        # Draw detections
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            x1, y1, x2, y2 = box.astype(int)
            # Labels are 1-indexed (1-4), so subtract 1 for array indexing
            color = self.colors[label - 1]

            # Draw box
            cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{self.class_names[label - 1]}: {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw background for text
            cv2.rectangle(image_vis, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

            # Draw text
            cv2.putText(image_vis, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save if path provided
        if save_path:
            cv2.imwrite(str(save_path), image_vis)
            print(f"Saved visualization to: {save_path}")

        # Show if requested
        if show:
            # Convert back to RGB for matplotlib
            image_vis_rgb = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(15, 10))
            plt.imshow(image_vis_rgb)
            plt.axis('off')
            plt.title('Dental Disease Detection', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

        return image_vis

    def process_directory(self, input_dir, output_dir, extensions=('.jpg', '.jpeg', '.png')):
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results
            extensions: Tuple of valid image extensions
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))

        print(f"Found {len(image_files)} images in {input_dir}")

        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Run detection
            predictions = self.predict(image)

            # Visualize and save
            save_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
            self.visualize(image, predictions, save_path=save_path, show=False)

            # Save predictions as JSON
            json_path = output_dir / f"{img_path.stem}_predictions.json"
            import json
            pred_dict = {
                'image': str(img_path),
                'detections': [
                    {
                        'class': self.class_names[label - 1],  # Labels are 1-indexed
                        'confidence': float(score),
                        'box': [float(x) for x in box]
                    }
                    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores'])
                ]
            }
            with open(json_path, 'w') as f:
                json.dump(pred_dict, f, indent=2)

        print(f"Processing complete! Results saved to: {output_dir}")


def main(args):
    """Main inference function"""
    # Create detector
    detector = DentalDetector(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        backbone=args.backbone,
        num_classes=args.num_classes,
        confidence_threshold=args.confidence_threshold
    )

    # Process based on input type
    if args.input_dir:
        # Process directory
        detector.process_directory(args.input_dir, args.output_dir)

    elif args.input_image:
        # Process single image
        image = Image.open(args.input_image).convert('RGB')
        predictions = detector.predict(image)

        print(f"\nDetected {len(predictions['boxes'])} objects:")
        for i, (label, score) in enumerate(zip(predictions['labels'], predictions['scores'])):
            print(f"  {i+1}. {detector.class_names[label - 1]}: {score:.3f}")  # Labels are 1-indexed

        # Visualize
        output_path = Path(args.output_dir) / f"detected_{Path(args.input_image).name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        detector.visualize(image, predictions, save_path=output_path, show=args.show)

    else:
        print("Error: Must provide either --input_image or --input_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on dental images')

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
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')

    # Input/output parameters
    parser.add_argument('--input_image', type=str,
                        help='Path to input image')
    parser.add_argument('--input_dir', type=str,
                        help='Path to directory containing images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory')
    parser.add_argument('--show', action='store_true',
                        help='Display visualization (only for single image)')

    args = parser.parse_args()
    main(args)
