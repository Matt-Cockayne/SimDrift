"""
Model manager for loading and managing trained models.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.training.architectures import get_model


@dataclass
class ModelInfo:
    """Information about a trained model."""
    dataset: str
    architecture: str
    test_accuracy: float
    test_f1: float
    ece: float
    parameters: int
    training_time_minutes: float
    checkpoint_path: Path
    metadata_path: Path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dataset': self.dataset,
            'architecture': self.architecture,
            'test_accuracy': self.test_accuracy,
            'test_f1': self.test_f1,
            'ece': self.ece,
            'parameters': self.parameters,
            'training_time_minutes': self.training_time_minutes,
            'checkpoint_path': str(self.checkpoint_path),
            'metadata_path': str(self.metadata_path)
        }


class ModelManager:
    """
    Manager for loading and comparing trained models.
    
    Provides:
    - Model loading from model zoo
    - Model metadata retrieval
    - Model comparison across datasets/architectures
    - Leaderboard generation
    """
    
    def __init__(self, model_zoo_dir: str = 'model_zoo'):
        """
        Initialize model manager.
        
        Args:
            model_zoo_dir: Path to model zoo directory
        """
        self.model_zoo_dir = Path(model_zoo_dir)
        self._model_cache = {}
        self._metadata_cache = {}
    
    def get_available_datasets(self) -> List[str]:
        """Get list of datasets with trained models."""
        if not self.model_zoo_dir.exists():
            return []
        
        datasets = [
            d.name for d in self.model_zoo_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        return sorted(datasets)
    
    def get_available_architectures(self, dataset: str) -> List[str]:
        """
        Get available architectures for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of architecture names
        """
        dataset_dir = self.model_zoo_dir / dataset
        if not dataset_dir.exists():
            return []
        
        # Find all checkpoint files
        checkpoints = list(dataset_dir.glob('*_best.pth'))
        architectures = [
            ckpt.stem.replace('_best', '') 
            for ckpt in checkpoints
        ]
        return sorted(architectures)
    
    def load_pretrained(
        self,
        dataset: str,
        architecture: str,
        device: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load a pretrained model.
        
        Args:
            dataset: Dataset name
            architecture: Architecture name
            device: Device to load model on (default: CPU)
            
        Returns:
            Loaded PyTorch model
        """
        device = device or 'cpu'
        
        # Check cache
        cache_key = f"{dataset}_{architecture}_{device}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Find checkpoint
        checkpoint_path = self.model_zoo_dir / dataset / f'{architecture}_best.pth'
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for {architecture} on {dataset}. "
                f"Expected: {checkpoint_path}"
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get metadata to determine model config
        metadata = self.get_model_metadata(dataset, architecture)
        n_classes = metadata['n_classes']
        
        # Infer number of channels from the checkpoint itself
        # Look for the first/stem conv layer to determine input channels
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Try to find the stem/first conv layer by common naming patterns
            stem_keys = [
                'conv_stem.weight',  # EfficientNet
                'conv1.weight',      # ResNet
                'conv1.conv.weight', # Some ResNet variants
                'features.0.0.weight',  # SimpleCNN and others
                'features.conv0.weight',
            ]
            
            n_channels = None
            for key in stem_keys:
                if key in state_dict:
                    # Shape is [out_channels, in_channels, H, W]
                    n_channels = state_dict[key].shape[1]
                    break
            
            # Fallback: find any conv layer with smallest number of input channels
            if n_channels is None:
                conv_keys = [k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) == 4]
                if conv_keys:
                    # Get the layer with minimum input channels (likely the first layer)
                    min_channels_key = min(conv_keys, key=lambda k: state_dict[k].shape[1])
                    n_channels = state_dict[min_channels_key].shape[1]
                else:
                    n_channels = 3
        else:
            n_channels = 3
        
        # Create model
        model = get_model(
            architecture,
            n_channels=n_channels,
            n_classes=n_classes,
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Cache model
        self._model_cache[cache_key] = model
        
        return model
    
    def get_model_metadata(
        self,
        dataset: str,
        architecture: str
    ) -> Dict:
        """
        Get metadata for a model.
        
        Args:
            dataset: Dataset name
            architecture: Architecture name
            
        Returns:
            Metadata dictionary
        """
        # Check cache
        cache_key = f"{dataset}_{architecture}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        # Load metadata
        metadata_path = self.model_zoo_dir / dataset / f'{architecture}_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata found for {architecture} on {dataset}. "
                f"Expected: {metadata_path}"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cache metadata
        self._metadata_cache[cache_key] = metadata
        
        return metadata
    
    def get_model_info(
        self,
        dataset: str,
        architecture: str
    ) -> ModelInfo:
        """
        Get structured model information.
        
        Args:
            dataset: Dataset name
            architecture: Architecture name
            
        Returns:
            ModelInfo object
        """
        metadata = self.get_model_metadata(dataset, architecture)
        
        checkpoint_path = self.model_zoo_dir / dataset / f'{architecture}_best.pth'
        metadata_path = self.model_zoo_dir / dataset / f'{architecture}_metadata.json'
        
        return ModelInfo(
            dataset=dataset,
            architecture=architecture,
            test_accuracy=metadata['test_performance']['accuracy'],
            test_f1=metadata['test_performance']['f1_score'],
            ece=metadata['test_performance']['ece'],
            parameters=metadata['model_info']['parameters'],
            training_time_minutes=metadata['training']['total_time_minutes'],
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path
        )
    
    def compare_models(
        self,
        dataset: Optional[str] = None,
        architecture: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare models across datasets and/or architectures.
        
        Args:
            dataset: Filter by dataset (None = all)
            architecture: Filter by architecture (None = all)
            
        Returns:
            DataFrame with model comparison
        """
        results = []
        
        datasets = [dataset] if dataset else self.get_available_datasets()
        
        for ds in datasets:
            architectures = [architecture] if architecture else self.get_available_architectures(ds)
            
            for arch in architectures:
                try:
                    info = self.get_model_info(ds, arch)
                    results.append(info.to_dict())
                except Exception as e:
                    print(f"Warning: Could not load {arch} on {ds}: {e}")
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Sort by test accuracy
        df = df.sort_values('test_accuracy', ascending=False)
        
        return df
    
    def get_leaderboard(
        self,
        metric: str = 'test_accuracy',
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get leaderboard of models sorted by metric.
        
        Args:
            metric: Metric to sort by
            top_k: Return only top k models (None = all)
            
        Returns:
            DataFrame with leaderboard
        """
        df = self.compare_models()
        
        if df.empty:
            return df
        
        # Sort by metric
        df = df.sort_values(metric, ascending=False)
        
        # Add rank
        df.insert(0, 'rank', range(1, len(df) + 1))
        
        # Return top k
        if top_k:
            df = df.head(top_k)
        
        return df
    
    def get_best_model(
        self,
        dataset: str,
        metric: str = 'test_accuracy'
    ) -> Tuple[str, ModelInfo]:
        """
        Get the best model for a dataset.
        
        Args:
            dataset: Dataset name
            metric: Metric to use for selection
            
        Returns:
            Tuple of (architecture_name, ModelInfo)
        """
        df = self.compare_models(dataset=dataset)
        
        if df.empty:
            raise ValueError(f"No models found for dataset: {dataset}")
        
        best_row = df.loc[df[metric].idxmax()]
        architecture = best_row['architecture']
        
        info = self.get_model_info(dataset, architecture)
        
        return architecture, info
    
    def print_summary(self):
        """Print summary of available models."""
        datasets = self.get_available_datasets()
        
        if not datasets:
            print("No models found in model zoo.")
            return
        
        print(f"\n{'='*70}")
        print("MODEL ZOO SUMMARY")
        print(f"{'='*70}\n")
        
        total_models = 0
        
        for dataset in datasets:
            architectures = self.get_available_architectures(dataset)
            total_models += len(architectures)
            print(f"{dataset:20s}: {len(architectures)} models")
            for arch in architectures:
                try:
                    info = self.get_model_info(dataset, arch)
                    print(f"  ✓ {arch:20s} - Acc: {info.test_accuracy:.4f}, "
                          f"F1: {info.test_f1:.4f}, ECE: {info.ece:.4f}")
                except Exception as e:
                    print(f"  ✗ {arch:20s} - Error loading metadata")
        
        print(f"\nTotal: {total_models} models across {len(datasets)} datasets")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    # Test model manager
    manager = ModelManager('model_zoo')
    
    print("Testing ModelManager...\n")
    
    # Print summary
    manager.print_summary()
    
    # Test loading
    datasets = manager.get_available_datasets()
    if datasets:
        dataset = datasets[0]
        architectures = manager.get_available_architectures(dataset)
        
        if architectures:
            arch = architectures[0]
            print(f"\nLoading {arch} on {dataset}...")
            
            try:
                model = manager.load_pretrained(dataset, arch, device='cpu')
                print(f"✓ Model loaded successfully")
                
                metadata = manager.get_model_metadata(dataset, arch)
                print(f"✓ Metadata loaded: Test Acc = {metadata['test_performance']['accuracy']:.4f}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Test comparison
    print("\n" + "="*70)
    print("Model Comparison (if available):")
    print("="*70)
    df = manager.compare_models()
    if not df.empty:
        print(df.to_string())
    else:
        print("No models available for comparison.")
