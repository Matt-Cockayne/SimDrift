"""MedMNIST dataset loader with support for multiple medical imaging datasets."""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO


class MedMNISTLoader:
    """
    Unified loader for MedMNIST datasets.
    
    Supports: PathMNIST, DermaMNIST, RetinaMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, etc.
    """
    
    AVAILABLE_DATASETS = {
        'pathmnist': medmnist.PathMNIST,
        'dermamnist': medmnist.DermaMNIST,
        'retinamnist': medmnist.RetinaMNIST,
        'bloodmnist': medmnist.BloodMNIST,
        'tissuemnist': medmnist.TissueMNIST,
        'pneumoniamnist': medmnist.PneumoniaMNIST,
        'organamnist': medmnist.OrganAMNIST,
        'organcmnist': medmnist.OrganCMNIST,
        'organsmnist': medmnist.OrganSMNIST,
    }
    
    def __init__(self, dataset_name: str = 'pathmnist', data_dir: str = './data/medmnist'):
        """
        Initialize MedMNIST loader.
        
        Args:
            dataset_name: Name of the MedMNIST dataset (e.g., 'pathmnist')
            data_dir: Directory to store downloaded data
        """
        if dataset_name.lower() not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not available. "
                f"Choose from: {list(self.AVAILABLE_DATASETS.keys())}"
            )
        
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataset info
        self.info = INFO[self.dataset_name]
        self.n_classes = len(self.info['label'])
        self.task = self.info['task']
        
        # Default transforms
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        # Load datasets
        self._load_datasets()
    
    def _load_datasets(self):
        """Load train, validation, and test sets."""
        dataset_class = self.AVAILABLE_DATASETS[self.dataset_name]
        
        print(f"Loading {self.dataset_name}...")
        self.train_dataset = dataset_class(
            split='train',
            transform=self.train_transform,
            download=True,
            root=str(self.data_dir)
        )
        
        self.val_dataset = dataset_class(
            split='val',
            transform=self.eval_transform,
            download=True,
            root=str(self.data_dir)
        )
        
        self.test_dataset = dataset_class(
            split='test',
            transform=self.eval_transform,
            download=True,
            root=str(self.data_dir)
        )
        
        print(f"✓ Loaded {self.dataset_name}:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val:   {len(self.val_dataset)} samples")
        print(f"  Test:  {len(self.test_dataset)} samples")
        print(f"  Classes: {self.n_classes}")
        print(f"  Task: {self.task}")
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for all splits.
        
        Args:
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        dataloaders = {
            'train': DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'val': DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        }
        return dataloaders
    
    def get_numpy_data(self, split: str = 'test', limit: Optional[int] = None, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get raw numpy arrays without transforms (useful for drift simulation).
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            limit: Maximum number of samples to load (None = load all)
            offset: Starting index for loading samples
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        dataset_class = self.AVAILABLE_DATASETS[self.dataset_name]
        
        # Load without transforms
        dataset = dataset_class(
            split=split,
            transform=None,
            download=False,
            root=str(self.data_dir)
        )
        
        images = []
        labels = []
        
        # Calculate range to load
        start_idx = offset
        end_idx = len(dataset) if limit is None else min(offset + limit, len(dataset))
        
        for idx in range(start_idx, end_idx):
            img, label = dataset[idx]
            images.append(np.array(img))
            labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels
    
    def get_class_names(self) -> Dict[int, str]:
        """Get mapping from class indices to names."""
        return {i: name for i, name in enumerate(self.info['label'].values())}
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        return {
            'name': self.dataset_name,
            'n_classes': self.n_classes,
            'task': self.task,
            'n_channels': self.info['n_channels'],
            'n_samples': {
                'train': len(self.train_dataset),
                'val': len(self.val_dataset),
                'test': len(self.test_dataset)
            },
            'class_names': self.get_class_names(),
            'description': self.info.get('description', 'N/A')
        }


class ReferenceProductionSplit:
    """
    Split a dataset into 'reference' (training distribution) and 'production' (monitoring) sets.
    Used for simulating production monitoring scenarios.
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        reference_ratio: float = 0.5
    ):
        """
        Args:
            images: Image array
            labels: Label array
            reference_ratio: Proportion of data to use as reference
        """
        n_samples = len(images)
        n_reference = int(n_samples * reference_ratio)
        
        # Random split
        indices = np.random.permutation(n_samples)
        
        self.reference_indices = indices[:n_reference]
        self.production_indices = indices[n_reference:]
        
        self.reference_images = images[self.reference_indices]
        self.reference_labels = labels[self.reference_indices]
        
        self.production_images = images[self.production_indices]
        self.production_labels = labels[self.production_indices]
    
    def get_reference_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get reference (training) distribution data."""
        return self.reference_images, self.reference_labels
    
    def get_production_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get production (monitoring) distribution data."""
        return self.production_images, self.production_labels


if __name__ == '__main__':
    # Example usage
    print("Testing MedMNIST loader...")
    
    # Load PathMNIST
    loader = MedMNISTLoader('pathmnist')
    print("\nDataset info:")
    print(loader.get_dataset_info())
    
    # Get dataloaders
    dataloaders = loader.get_dataloaders(batch_size=64)
    
    # Test batch
    batch_images, batch_labels = next(iter(dataloaders['test']))
    print(f"\nBatch shape: {batch_images.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    
    # Get numpy data for drift simulation
    test_images, test_labels = loader.get_numpy_data('test')
    print(f"\nNumpy arrays: {test_images.shape}, {test_labels.shape}")
    
    print("\n✓ All tests passed!")
