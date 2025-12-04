"""
Model architectures for SimDrift.

Provides factory functions for different model architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from simple_classifier import SimpleCNN


def get_model(
    architecture: str,
    n_channels: int = 3,
    n_classes: int = 9,
    pretrained: bool = False
) -> nn.Module:
    """
    Get a model by architecture name.
    
    Args:
        architecture: Model architecture name
            Options: 'simple_cnn', 'resnet18', 'resnet34', 'efficientnet_b0', 
                     'efficientnet_b1', 'vit_tiny'
        n_channels: Number of input channels
        n_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ImageNet)
        
    Returns:
        PyTorch model
    """
    architecture = architecture.lower()
    
    if architecture == 'simple_cnn':
        return SimpleCNN(
            n_channels=n_channels,
            n_classes=n_classes,
            dropout_rate=0.3
        )
    
    elif architecture == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        
        # Modify first conv layer if needed
        if n_channels != 3:
            model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final layer
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    
    elif architecture == 'resnet34':
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)
        
        if n_channels != 3:
            model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    
    elif architecture == 'efficientnet_b0':
        if pretrained:
            model = timm.create_model('efficientnet_b0', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Modify classifier
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)
        
        # Handle channel mismatch
        if n_channels != 3:
            model.conv_stem = nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        return model
    
    elif architecture == 'efficientnet_b1':
        if pretrained:
            model = timm.create_model('efficientnet_b1', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b1', pretrained=False)
        
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)
        
        if n_channels != 3:
            model.conv_stem = nn.Conv2d(n_channels, 40, kernel_size=3, stride=2, padding=1, bias=False)
        
        return model
    
    elif architecture == 'vit_tiny':
        # Vision Transformer (tiny variant for efficiency)
        model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
        
        # Modify head
        if hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, n_classes)
        
        # Note: ViT typically requires larger images (224x224)
        # For MedMNIST (28x28), we'll resize during preprocessing
        
        return model
    
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: simple_cnn, resnet18, resnet34, efficientnet_b0, efficientnet_b1, vit_tiny"
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module, architecture: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        architecture: Architecture name
        
    Returns:
        Dictionary with model information
    """
    total_params = count_parameters(model)
    
    # Calculate model size (MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'architecture': architecture,
        'total_parameters': total_params,
        'size_mb': round(size_mb, 2),
        'trainable': True
    }


if __name__ == '__main__':
    # Test all architectures
    print("Testing model architectures...\n")
    
    architectures = ['simple_cnn', 'resnet18', 'efficientnet_b0']
    test_input = torch.randn(4, 3, 28, 28)
    
    for arch in architectures:
        print(f"Testing {arch}...")
        model = get_model(arch, n_channels=3, n_classes=9, pretrained=False)
        
        # For ViT, need larger images
        if 'vit' in arch:
            test_input_resized = torch.nn.functional.interpolate(test_input, size=(224, 224))
            output = model(test_input_resized)
        else:
            output = model(test_input)
        
        info = get_model_info(model, arch)
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Parameters: {info['total_parameters']:,}")
        print(f"  ✓ Size: {info['size_mb']:.2f} MB\n")
    
    print("All architectures tested successfully!")
