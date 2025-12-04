"""Lightweight CNN models for MedMNIST classification."""

from .simple_classifier import SimpleCNN, train_model, evaluate_model

__all__ = ['SimpleCNN', 'train_model', 'evaluate_model']
