import torch
import torch.nn as nn
import torchvision.models as models
import timm


class DentalClassifier(nn.Module):
    """
    Classification model for dental diseases.
    Supports multiple backbone architectures with pretrained weights.
    """

    def __init__(self, num_classes=6, backbone='resnet50', pretrained=True, dropout=0.5):
        """
        Args:
            num_classes (int): Number of disease classes (default: 6)
            backbone (str): Backbone architecture
            pretrained (bool): Use ImageNet pretrained weights
            dropout (float): Dropout rate before final classification layer
        """
        super(DentalClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original FC layer

        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'efficientnet_b3':
            self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'vit_base_patch16_224':
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_base_patch32_224':
            self.backbone = timm.create_model('vit_base_patch32_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_base_patch16_384':
            self.backbone = timm.create_model('vit_base_patch16_384', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_base_patch32_384':
            self.backbone = timm.create_model('vit_base_patch32_384', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_large_patch14_224':
            self.backbone = timm.create_model('vit_large_patch14_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_large_patch16_224':
            self.backbone = timm.create_model('vit_large_patch16_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_large_patch32_224':
            self.backbone = timm.create_model('vit_large_patch32_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_large_patch16_384':
            self.backbone = timm.create_model('vit_large_patch16_384', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_small_patch16_224':
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'vit_tiny_patch16_224':
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif backbone == 'convnext_tiny':
            self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained)
            num_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        # Swin Transformer models
        elif backbone == 'swin_tiny_patch4_window7_224':
            self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'swin_small_patch4_window7_224':
            self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'swin_base_patch4_window7_224':
            self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'swin_large_patch4_window7_224':
            self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        # DeiT3 models
        elif backbone == 'deit3_small_patch16_224':
            self.backbone = timm.create_model('deit3_small_patch16_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'deit3_base_patch16_224':
            self.backbone = timm.create_model('deit3_base_patch16_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'deit3_large_patch16_224':
            self.backbone = timm.create_model('deit3_large_patch16_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        # MaxViT models
        elif backbone == 'maxvit_tiny_tf_224':
            self.backbone = timm.create_model('maxvit_tiny_tf_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'maxvit_small_tf_224':
            self.backbone = timm.create_model('maxvit_small_tf_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'maxvit_base_tf_224':
            self.backbone = timm.create_model('maxvit_base_tf_224', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        # ConvNeXtV2 models
        elif backbone == 'convnextv2_tiny':
            self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'convnextv2_base':
            self.backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        elif backbone == 'convnextv2_large':
            self.backbone = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_classification_model(num_classes=6, backbone='resnet50', pretrained=True, dropout=0.5):
    """
    Factory function to create classification models.

    Args:
        num_classes (int): Number of classes
        backbone (str): Backbone architecture
        pretrained (bool): Use ImageNet pretrained weights
        dropout (float): Dropout rate

    Returns:
        DentalClassifier model
    """
    model = DentalClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = get_classification_model(num_classes=6, backbone='resnet50', pretrained=True)

    print(f"Model created: {model.backbone_name}")
    print(f"Number of classes: {model.num_classes}")

    # Test forward pass
    dummy_input = torch.rand(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (4, 6)
    print(f"Output logits (sample): {output[0]}")
