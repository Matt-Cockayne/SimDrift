"""
Simple CNN classifier for MedMNIST datasets.

Lightweight model for demonstration purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """
    Simple CNN for MedMNIST classification.
    
    Architecture:
    - 3 conv blocks with increasing channels
    - Batch normalization and dropout
    - Adaptive pooling for flexible input sizes
    - Fully connected classifier
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 9,
        dropout_rate: float = 0.3
    ):
        """
        Initialize SimpleCNN.
        
        Args:
            n_channels: Number of input channels (1 for grayscale, 3 for RGB)
            n_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(SimpleCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probas = F.softmax(logits, dim=1)
        return probas


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 20,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Train the model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=verbose
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}') if verbose else train_loader
        
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            if verbose:
                iterator.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, device, verbose=False)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch + 1
        
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    return_predictions: bool = False
) -> Tuple:
    """
    Evaluate the model.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        verbose: Whether to print progress
        return_predictions: Whether to return predictions and probabilities
        
    Returns:
        If return_predictions is False: (test_loss, test_acc)
        If return_predictions is True: (test_loss, test_acc, y_true, y_pred, y_proba)
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        iterator = tqdm(test_loader, desc='Evaluating') if verbose else test_loader
        
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Statistics
            test_loss += loss.item() * images.size(0)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)
            
            if return_predictions:
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    if verbose:
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    if return_predictions:
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        return test_loss, test_acc, y_true, y_pred, y_proba
    
    return test_loss, test_acc


def save_model(model: nn.Module, path: str):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': model.n_channels,
            'n_classes': model.n_classes
        }
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: str = 'cpu') -> nn.Module:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    config = checkpoint['model_config']
    model = SimpleCNN(
        n_channels=config['n_channels'],
        n_classes=config['n_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from {path}")
    return model


if __name__ == '__main__':
    # Test model
    print("Testing SimpleCNN...")
    
    # Create dummy data
    batch_size = 16
    n_channels = 3
    n_classes = 9
    img_size = 28
    
    dummy_images = torch.randn(batch_size, n_channels, img_size, img_size)
    dummy_labels = torch.randint(0, n_classes, (batch_size,))
    
    # Create model
    model = SimpleCNN(n_channels=n_channels, n_classes=n_classes)
    print(f"\n✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    outputs = model(dummy_images)
    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {dummy_images.shape}")
    print(f"  Output shape: {outputs.shape}")
    
    # Test probability prediction
    probas = model.predict_proba(dummy_images)
    print(f"\n✓ Probability prediction successful")
    print(f"  Probabilities shape: {probas.shape}")
    print(f"  Sum of probabilities: {probas[0].sum():.4f}")
    
    print("\n✓ All tests passed!")
