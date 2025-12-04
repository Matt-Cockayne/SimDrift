"""
SimDrift - Simulation-Based Model Drift Detection Dashboard

A comprehensive educational tool for understanding and detecting
data drift, concept drift, and model performance degradation.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

from .data import MedMNISTLoader, DriftGenerator
from .monitoring import DriftDetector, PerformanceTracker, AlertSystem
from .simulations import ScenarioLibrary, ScenarioRunner
from .models import SimpleCNN

__all__ = [
    'MedMNISTLoader',
    'DriftGenerator',
    'DriftDetector',
    'PerformanceTracker',
    'AlertSystem',
    'ScenarioLibrary',
    'ScenarioRunner',
    'SimpleCNN'
]
