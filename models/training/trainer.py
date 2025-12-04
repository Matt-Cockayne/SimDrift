"""
Model trainer with comprehensive tracking and visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.calibration import calibration_curve


class ModelTrainer:
    """
    Comprehensive model trainer with tracking and visualization.
    
    Tracks:
    - Training/validation metrics
    - Confusion matrices
    - Calibration curves
    - Training time
    - Learning curves
    """
    
    def __init__(
        self,
        model: nn.Module,
        architecture: str,
        dataset_name: str,
        device: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            architecture: Architecture name
            dataset_name: Name of the dataset
            device: Device to train on (auto-detected if None)
            output_dir: Directory to save outputs
        """
        self.model = model
        self.architecture = architecture
        self.dataset_name = dataset_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup output directory
        if output_dir is None:
            output_dir = f"model_zoo/{dataset_name}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 30,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model with comprehensive tracking.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=verbose
        )
        
        start_time = time.time()
        epochs_without_improvement = 0
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, epoch, n_epochs, verbose
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                val_loader, criterion, verbose
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
                self._save_checkpoint('best')
                if verbose:
                    print(f"  ✓ New best model saved! Val Acc: {val_acc:.4f}")
            else:
                epochs_without_improvement += 1
            
            # Print epoch summary
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
                      f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        self.history['total_training_time'] = total_time
        self.history['best_val_acc'] = self.best_val_acc
        self.history['best_epoch'] = self.best_epoch
        
        if verbose:
            print(f"\n✓ Training completed in {total_time/60:.1f} minutes")
            print(f"  Best Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        n_epochs: int,
        verbose: bool
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        iterator = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{n_epochs} [Train]',
            leave=False
        ) if verbose else train_loader
        
        for images, labels in iterator:
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            if verbose:
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        verbose: bool
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        save_visualizations: bool = True
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            test_loader: Test data loader
            class_names: Names of classes for visualization
            save_visualizations: Whether to save plots
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.squeeze().long()
                
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Generate visualizations
        if save_visualizations:
            self._generate_visualizations(
                all_labels, all_preds, all_probs, class_names, metrics
            )
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict:
        """Calculate comprehensive metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calibration metrics (Expected Calibration Error)
        ece = self._calculate_ece(y_true, y_probs)
        mce = self._calculate_mce(y_true, y_probs)
        
        # Try to calculate AUC if multi-class
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except:
            auc = None
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'ece': float(ece),
            'mce': float(mce),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': cm.tolist()
        }
    
    def _calculate_ece(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_size = mask.sum()
                ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)
        
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                mce = max(mce, abs(bin_acc - bin_conf))
        
        return mce
    
    def _generate_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        class_names: Optional[List[str]],
        metrics: Dict
    ):
        """Generate and save visualization plots."""
        # Create visualizations directory
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Training curves
        self._plot_training_curves(vis_dir)
        
        # 2. Confusion matrix
        self._plot_confusion_matrix(metrics['confusion_matrix'], class_names, vis_dir)
        
        # 3. Calibration plot
        self._plot_calibration(y_true, y_probs, vis_dir)
        
        print(f"✓ Visualizations saved to {vis_dir}")
    
    def _plot_training_curves(self, save_dir: Path):
        """Plot training/validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Validation', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], label='Train', marker='o')
        axes[1].plot(epochs, self.history['val_acc'], label='Validation', marker='s')
        axes[1].axhline(y=self.best_val_acc, color='r', linestyle='--', label='Best')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm: List, class_names: Optional[List[str]], save_dir: Path):
        """Plot confusion matrix."""
        import seaborn as sns
        
        cm_array = np.array(cm)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm_array, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names if class_names else range(len(cm)),
            yticklabels=class_names if class_names else range(len(cm)),
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration(self, y_true: np.ndarray, y_probs: np.ndarray, save_dir: Path):
        """Plot calibration curve."""
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            accuracies, confidences, n_bins=10, strategy='uniform'
        )
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(prob_pred, prob_true, marker='o', label='Model')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'calibration_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_checkpoint(self, checkpoint_name: str = 'best'):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f'{self.architecture}_{checkpoint_name}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'dataset': self.dataset_name,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, checkpoint_path)
    
    def save_metadata(self, test_metrics: Dict, n_classes: int):
        """Save model metadata as JSON."""
        metadata = {
            'architecture': self.architecture,
            'dataset': self.dataset_name,
            'n_classes': n_classes,
            'training': {
                'best_epoch': self.best_epoch,
                'best_val_acc': self.best_val_acc,
                'total_time_minutes': self.history.get('total_training_time', 0) / 60,
                'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else 0,
            },
            'test_performance': test_metrics,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': self.device
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / f'{self.architecture}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to {metadata_path}")
        
        return metadata
