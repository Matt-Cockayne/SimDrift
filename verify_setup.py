#!/usr/bin/env python3
"""
Quick verification script to check SimDrift setup.
Run this to ensure all components are working.
"""

import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(message, status='info'):
    """Print colored status message."""
    colors = {
        'success': Colors.GREEN + '✓',
        'error': Colors.RED + '✗',
        'warning': Colors.YELLOW + '⚠',
        'info': Colors.BLUE + 'ℹ'
    }
    print(f"{colors.get(status, Colors.BLUE + 'ℹ')} {message}{Colors.END}")

def check_imports():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies...")
    print("="*60)
    
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'medmnist': 'medmnist',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'scipy': 'scipy',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib'
    }
    
    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print_status(f"{package_name:20s} OK", 'success')
        except ImportError:
            print_status(f"{package_name:20s} MISSING", 'error')
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "="*60)
    print("Checking Project Structure...")
    print("="*60)
    
    required_paths = [
        'data/medmnist_loader.py',
        'data/drift_generators.py',
        'models/simple_classifier.py',
        'models/model_manager.py',
        'models/training/architectures.py',
        'models/training/trainer.py',
        'monitoring/drift_detectors.py',
        'monitoring/performance_trackers.py',
        'monitoring/alerting.py',
        'dashboard/enhanced_app.py',
        'dashboard/pages/2_🏛️_Model_Zoo.py',
        'simulations/scenarios.py',
        'train_all_models.py'
    ]
    
    all_ok = True
    for path in required_paths:
        if Path(path).exists():
            print_status(f"{path:40s} OK", 'success')
        else:
            print_status(f"{path:40s} MISSING", 'error')
            all_ok = False
    
    return all_ok

def check_model_zoo():
    """Check if any models are trained."""
    print("\n" + "="*60)
    print("Checking Model Zoo...")
    print("="*60)
    
    model_zoo_path = Path('model_zoo')
    
    if not model_zoo_path.exists():
        print_status("Model zoo directory not found", 'warning')
        print_status("Run: python train_all_models.py --quick-test", 'info')
        return False
    
    datasets = [d for d in model_zoo_path.iterdir() if d.is_dir()]
    
    if not datasets:
        print_status("No trained models found", 'warning')
        print_status("Dashboard will work with simulated models", 'info')
        print_status("For full experience, run: python train_all_models.py --quick-test", 'info')
        return False
    
    print_status(f"Found {len(datasets)} dataset(s) with trained models", 'success')
    
    total_models = 0
    for dataset_dir in datasets:
        models = list(dataset_dir.glob('*_best.pth'))
        if models:
            print_status(f"  {dataset_dir.name:15s}: {len(models)} model(s)", 'success')
            total_models += len(models)
    
    print_status(f"Total: {total_models} trained model(s)", 'success')
    return True

def test_imports_quick():
    """Quick test of key imports."""
    print("\n" + "="*60)
    print("Testing Key Components...")
    print("="*60)
    
    try:
        from data.medmnist_loader import MedMNISTLoader
        print_status("MedMNISTLoader import", 'success')
        
        from data.drift_generators import DriftGenerator
        print_status("DriftGenerator import", 'success')
        
        from monitoring.drift_detectors import DriftDetector
        print_status("DriftDetector import", 'success')
        
        from models.model_manager import ModelManager
        print_status("ModelManager import", 'success')
        
        print_status("All key components import successfully", 'success')
        return True
        
    except Exception as e:
        print_status(f"Import error: {str(e)}", 'error')
        return False

def print_next_steps(all_checks_passed):
    """Print next steps based on verification results."""
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60 + "\n")
    
    if all_checks_passed:
        print_status("All checks passed! 🎉", 'success')
        print("\nYou can now:")
        print(f"  {Colors.BLUE}1.{Colors.END} Launch dashboard:")
        print(f"     {Colors.GREEN}streamlit run dashboard/enhanced_app.py{Colors.END}")
        print(f"\n  {Colors.BLUE}2.{Colors.END} Train models (optional):")
        print(f"     {Colors.GREEN}python train_all_models.py --quick-test{Colors.END}")
    else:
        print_status("Some checks failed", 'warning')
        print("\nTo fix:")
        print(f"  {Colors.BLUE}1.{Colors.END} Install missing dependencies:")
        print(f"     {Colors.GREEN}pip install -r requirements.txt{Colors.END}")
        print(f"\n  {Colors.BLUE}2.{Colors.END} Make sure you're in the SimDrift directory")

def main():
    """Run all verification checks."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("SimDrift Setup Verification")
    print(f"{'='*60}{Colors.END}")
    
    # Run checks
    deps_ok = check_imports()
    structure_ok = check_project_structure()
    components_ok = test_imports_quick()
    models_exist = check_model_zoo()
    
    all_checks_passed = deps_ok and structure_ok and components_ok
    
    # Print summary
    print_next_steps(all_checks_passed)
    
    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())
