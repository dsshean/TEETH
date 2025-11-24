import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import timm


def create_fasterrcnn_model(num_classes=4, backbone_name='resnet50', pretrained=True):
    """
    Create Faster R-CNN model with custom backbone.

    Args:
        num_classes (int): Number of detection classes (4 for dental diseases)
        backbone_name (str): Backbone architecture ('resnet50', 'resnet101', 'efficientnet_b3', etc.)
        pretrained (bool): Use pretrained weights

    Returns:
        FasterRCNN model
    """
    if 'resnet' in backbone_name:
        # Load pretrained ResNet backbone
        if backbone_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone_name == 'resnet101':
            backbone = torchvision.models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")

        # Remove the average pooling and fully connected layers
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048 if 'resnet' in backbone_name else 1024

    elif 'efficientnet' in backbone_name:
        # Use timm for EfficientNet backbones
        backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        backbone.out_channels = backbone.feature_info[-1]['num_chs']

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Define anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Define ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,  # +1 for background class
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


def create_retinanet_model(num_classes=4, backbone_name='resnet50', pretrained=True):
    """
    Create RetinaNet model for object detection.

    Args:
        num_classes (int): Number of detection classes
        backbone_name (str): Backbone architecture
        pretrained (bool): Use pretrained weights

    Returns:
        RetinaNet model
    """
    from torchvision.models.detection import retinanet_resnet50_fpn
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    # Load pretrained RetinaNet
    model = retinanet_resnet50_fpn(pretrained=pretrained)

    # Get number of anchors
    num_anchors = model.head.classification_head.num_anchors

    # Replace classification head
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model


def create_fcos_model(num_classes=4, backbone_name='resnet50', pretrained=True):
    """
    Create FCOS (Fully Convolutional One-Stage) model.

    Args:
        num_classes (int): Number of detection classes
        backbone_name (str): Backbone architecture
        pretrained (bool): Use pretrained weights

    Returns:
        FCOS model
    """
    try:
        from torchvision.models.detection import fcos_resnet50_fpn
        from torchvision.models.detection.fcos import FCOSClassificationHead

        # Load pretrained FCOS
        model = fcos_resnet50_fpn(pretrained=pretrained)

        # Get number of anchors
        num_anchors = model.head.classification_head.num_anchors

        # Replace classification head
        model.head.classification_head = FCOSClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

        return model
    except ImportError:
        print("FCOS not available in this PyTorch version. Using Faster R-CNN instead.")
        return create_fasterrcnn_model(num_classes, backbone_name, pretrained)


class DentalDetector(nn.Module):
    """
    Wrapper class for dental disease detection models.
    Provides a unified interface for different detection architectures.
    """

    def __init__(self, num_classes=4, model_type='fasterrcnn', backbone='resnet50', pretrained=True):
        """
        Args:
            num_classes (int): Number of detection classes (4 for dental diseases)
            model_type (str): Detection model type ('fasterrcnn', 'retinanet', 'fcos')
            backbone (str): Backbone architecture
            pretrained (bool): Use pretrained weights
        """
        super(DentalDetector, self).__init__()

        self.num_classes = num_classes
        self.model_type = model_type
        self.backbone = backbone

        # Create model based on type
        if model_type == 'fasterrcnn':
            self.model = create_fasterrcnn_model(num_classes, backbone, pretrained)
        elif model_type == 'retinanet':
            self.model = create_retinanet_model(num_classes, backbone, pretrained)
        elif model_type == 'fcos':
            self.model = create_fcos_model(num_classes, backbone, pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor] or Tensor): Images to detect
            targets (list[Dict[Tensor]]): Ground truth boxes and labels (for training)

        Returns:
            During training: Dict with losses
            During inference: List of detection results
        """
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)


def get_model(num_classes=4, model_type='fasterrcnn', backbone='resnet50', pretrained=True):
    """
    Factory function to create detection models.

    Args:
        num_classes (int): Number of classes (default: 4 for dental diseases)
        model_type (str): Model architecture ('fasterrcnn', 'retinanet', 'fcos')
        backbone (str): Backbone network ('resnet50', 'resnet101', 'efficientnet_b3')
        pretrained (bool): Use ImageNet pretrained weights

    Returns:
        Detection model
    """
    model = DentalDetector(
        num_classes=num_classes,
        model_type=model_type,
        backbone=backbone,
        pretrained=pretrained
    )

    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = get_model(num_classes=4, model_type='fasterrcnn', backbone='resnet50', pretrained=True)

    print(f"Model created: {model.model_type}")
    print(f"Backbone: {model.backbone}")
    print(f"Number of classes: {model.num_classes}")

    # Test forward pass
    model.eval()
    dummy_images = [torch.rand(3, 640, 640) for _ in range(2)]

    with torch.no_grad():
        predictions = model(dummy_images)

    print(f"\nNumber of predictions: {len(predictions)}")
    print(f"Prediction keys: {predictions[0].keys()}")
    print(f"Boxes shape: {predictions[0]['boxes'].shape}")
    print(f"Labels shape: {predictions[0]['labels'].shape}")
    print(f"Scores shape: {predictions[0]['scores'].shape}")
