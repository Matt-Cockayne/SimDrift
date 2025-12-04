"""
Train models on all MedMNIST datasets.

This script trains multiple architectures on each dataset and saves
checkpoints and metadata for the model zoo.
"""

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.medmnist_loader import MedMNISTLoader
from models.training.architectures import get_model, get_model_info
from models.training.trainer import ModelTrainer


# Dataset configurations
DATASETS = [
    'pathmnist',      # Pathology (9 classes)
    'dermamnist',     # Dermatology (7 classes)
    'octmnist',       # OCT (4 classes)
    'pneumoniamnist', # Pneumonia (2 classes)
    'retinamnist',    # Retina (5 classes)
    'breastmnist',    # Breast (2 classes)
    'bloodmnist',     # Blood (8 classes)
    'tissuemnist',    # Tissue (8 classes)
]

# Architecture configurations
ARCHITECTURES = {
    'simple_cnn': {
        'epochs': 30,
        'lr': 0.001,
        'batch_size': 128
    },
    'resnet18': {
        'epochs': 40,
        'lr': 0.0001,
        'batch_size': 64,
        'pretrained': False
    },
    'efficientnet_b0': {
        'epochs': 40,
        'lr': 0.0001,
        'batch_size': 64,
        'pretrained': False
    }
}


def train_model_for_dataset(
    dataset_name: str,
    architecture: str,
    config: dict,
    output_dir: str = 'model_zoo',
    device: str = None
):
    """
    Train a single model on a dataset.
    
    Args:
        dataset_name: Name of MedMNIST dataset
        architecture: Model architecture name
        config: Training configuration
        output_dir: Base directory for model zoo
        device: Device to train on
    """
    print(f"\n{'='*70}")
    print(f"Training {architecture} on {dataset_name}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    loader = MedMNISTLoader(dataset_name)
    
    # Get dataloaders
    dataloaders = loader.get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=4,
        shuffle_train=True
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Get model
    print(f"Creating {architecture} model...")
    n_channels = loader.info['n_channels']
    n_classes = len(loader.info['label'])
    
    model = get_model(
        architecture,
        n_channels=n_channels,
        n_classes=n_classes,
        pretrained=config.get('pretrained', False)
    )
    
    # Print model info
    info = get_model_info(model, architecture)
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Size: {info['size_mb']:.2f} MB")
    
    # Create trainer
    model_output_dir = Path(output_dir) / dataset_name
    trainer = ModelTrainer(
        model=model,
        architecture=architecture,
        dataset_name=dataset_name,
        device=device,
        output_dir=str(model_output_dir)
    )
    
    # Train
    print(f"\nTraining for {config['epochs']} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config['epochs'],
        learning_rate=config['lr'],
        patience=10,
        verbose=True
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = trainer.evaluate(
        test_loader=test_loader,
        class_names=loader.info['label'].values(),
        save_visualizations=True
    )
    
    # Save metadata
    metadata = trainer.save_metadata(test_metrics, n_classes)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Training Summary for {architecture} on {dataset_name}")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"ECE: {test_metrics['ece']:.4f}")
    print(f"Training Time: {history['total_training_time']/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    # Clean up memory to prevent OOM errors
    print("Cleaning up memory...")
    del model, trainer, train_loader, val_loader, test_loader, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"✓ Memory cleared\n")
    
    return metadata


def train_all(
    datasets: list = None,
    architectures: list = None,
    output_dir: str = 'model_zoo',
    device: str = None
):
    """
    Train all specified models on all specified datasets.
    
    Args:
        datasets: List of dataset names (None = all)
        architectures: List of architecture names (None = all)
        output_dir: Base directory for model zoo
        device: Device to train on
    """
    datasets = datasets or DATASETS
    architectures = architectures or list(ARCHITECTURES.keys())
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    total_models = len(datasets) * len(architectures)
    current = 0
    
    results = []
    
    for dataset_name in datasets:
        for architecture in architectures:
            current += 1
            print(f"\n\n{'#'*70}")
            print(f"Progress: {current}/{total_models}")
            print(f"{'#'*70}")
            
            try:
                config = ARCHITECTURES[architecture]
                metadata = train_model_for_dataset(
                    dataset_name=dataset_name,
                    architecture=architecture,
                    config=config,
                    output_dir=output_dir,
                    device=device
                )
                results.append({
                    'dataset': dataset_name,
                    'architecture': architecture,
                    'success': True,
                    'test_acc': metadata['test_performance']['accuracy']
                })
                
            except Exception as e:
                print(f"\n❌ Error training {architecture} on {dataset_name}:")
                print(f"   {str(e)}")
                results.append({
                    'dataset': dataset_name,
                    'architecture': architecture,
                    'success': False,
                    'error': str(e)
                })
    
    # Print final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successfully trained: {len(successful)}/{total_models} models\n")
    
    if successful:
        print("Successful models:")
        for r in successful:
            print(f"  ✓ {r['dataset']:15s} - {r['architecture']:20s} - Test Acc: {r['test_acc']:.4f}")
    
    if failed:
        print(f"\n❌ Failed models ({len(failed)}):")
        for r in failed:
            print(f"  ✗ {r['dataset']:15s} - {r['architecture']:20s} - {r['error']}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train models on MedMNIST datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of datasets to train on (default: all)'
    )
    parser.add_argument(
        '--architectures',
        nargs='+',
        default=None,
        help='List of architectures to train (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_zoo',
        help='Output directory for model zoo'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (default: auto-detect)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with 1 dataset and 1 architecture'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick test mode...")
        datasets = ['pathmnist']
        architectures = ['simple_cnn']
        # Reduce epochs for quick test
        ARCHITECTURES['simple_cnn']['epochs'] = 3
    else:
        datasets = args.datasets
        architectures = args.architectures
    
    train_all(
        datasets=datasets,
        architectures=architectures,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
