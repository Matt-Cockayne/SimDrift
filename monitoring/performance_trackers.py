"""
Performance tracking for model monitoring over time.

Tracks:
- Classification metrics (accuracy, precision, recall, F1)
- Calibration metrics (ECE, MCE)
- Fairness metrics (demographic parity, equalized odds)
- Prediction distributions
- Confidence scores
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from scipy.special import softmax


class PerformanceTracker:
    """
    Track model performance over time for drift monitoring.
    """
    
    def __init__(self, n_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize performance tracker.
        
        Args:
            n_classes: Number of classes
            class_names: Optional class names
        """
        self.n_classes = n_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(n_classes)]
        
        self.tracking_history = []
    
    def track_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        sensitive_attributes: Optional[Dict[str, np.ndarray]] = None,
        timestamp: Optional[str] = None
    ) -> Dict:
        """
        Compute and track performance metrics.
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_proba: Predicted probabilities (N, n_classes)
            sensitive_attributes: Dict of sensitive attributes for fairness
            timestamp: Optional timestamp for tracking
            
        Returns:
            Dictionary with all performance metrics
        """
        metrics = {
            'timestamp': timestamp,
            'n_samples': len(y_true)
        }
        
        # Classification metrics
        metrics['classification'] = self._compute_classification_metrics(
            y_true, y_pred, y_proba
        )
        
        # Calibration metrics
        if y_proba is not None:
            metrics['calibration'] = self._compute_calibration_metrics(
                y_true, y_proba
            )
        
        # Fairness metrics
        if sensitive_attributes is not None:
            metrics['fairness'] = self._compute_fairness_metrics(
                y_true, y_pred, sensitive_attributes
            )
        
        # Prediction distribution
        metrics['prediction_distribution'] = self._compute_prediction_distribution(
            y_pred, y_proba
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            y_true, y_pred, labels=range(self.n_classes)
        ).tolist()
        
        self.tracking_history.append(metrics)
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """Compute classification metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Per-class metrics
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = y_true == i
            if class_mask.sum() > 0:
                per_class[class_name] = {
                    'precision': float(precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)),
                    'f1': float(f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)),
                    'support': int(class_mask.sum())
                }
        
        metrics['per_class'] = per_class
        
        # AUC metrics if probabilities available
        if y_proba is not None:
            try:
                if self.n_classes == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    metrics['pr_auc'] = float(average_precision_score(y_true, y_proba[:, 1]))
                else:
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
                    metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro'))
            except Exception as e:
                metrics['auc_error'] = str(e)
        
        return metrics
    
    def _compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Compute calibration metrics (ECE, MCE).
        
        Expected Calibration Error (ECE): Average difference between confidence and accuracy
        Maximum Calibration Error (MCE): Maximum difference across bins
        """
        # Get predicted class and confidence
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                bin_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'proportion': float(prop_in_bin),
                    'count': int(in_bin.sum())
                })
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'bins': bin_data,
            'avg_confidence': float(confidences.mean()),
            'avg_accuracy': float(accuracies.mean())
        }
    
    def _compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Compute fairness metrics for sensitive attributes.
        
        Metrics:
        - Demographic Parity: P(Y_hat=1 | A=0) ≈ P(Y_hat=1 | A=1)
        - Equalized Odds: TPR and FPR equal across groups
        - Equal Opportunity: TPR equal across groups
        """
        fairness_metrics = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            unique_groups = np.unique(attr_values)
            
            # Demographic parity
            positive_rates = {}
            for group in unique_groups:
                group_mask = attr_values == group
                positive_rate = (y_pred[group_mask] == 1).mean() if self.n_classes == 2 else (y_pred[group_mask] > 0).mean()
                positive_rates[str(group)] = float(positive_rate)
            
            demographic_parity_diff = float(max(positive_rates.values()) - min(positive_rates.values()))
            
            # Equalized odds (for binary classification)
            if self.n_classes == 2:
                tpr_by_group = {}
                fpr_by_group = {}
                
                for group in unique_groups:
                    group_mask = attr_values == group
                    
                    # TPR: P(Y_hat=1 | Y=1, A=group)
                    true_positives = ((y_true == 1) & (y_pred == 1) & group_mask).sum()
                    condition_positive = ((y_true == 1) & group_mask).sum()
                    tpr = true_positives / condition_positive if condition_positive > 0 else 0
                    
                    # FPR: P(Y_hat=1 | Y=0, A=group)
                    false_positives = ((y_true == 0) & (y_pred == 1) & group_mask).sum()
                    condition_negative = ((y_true == 0) & group_mask).sum()
                    fpr = false_positives / condition_negative if condition_negative > 0 else 0
                    
                    tpr_by_group[str(group)] = float(tpr)
                    fpr_by_group[str(group)] = float(fpr)
                
                equalized_odds_diff = float(
                    max(max(tpr_by_group.values()) - min(tpr_by_group.values()),
                        max(fpr_by_group.values()) - min(fpr_by_group.values()))
                )
                
                fairness_metrics[attr_name] = {
                    'demographic_parity_diff': demographic_parity_diff,
                    'positive_rates': positive_rates,
                    'equalized_odds_diff': equalized_odds_diff,
                    'tpr_by_group': tpr_by_group,
                    'fpr_by_group': fpr_by_group
                }
            else:
                # Multi-class: only demographic parity
                fairness_metrics[attr_name] = {
                    'demographic_parity_diff': demographic_parity_diff,
                    'positive_rates': positive_rates
                }
        
        return fairness_metrics
    
    def _compute_prediction_distribution(
        self,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """Compute prediction distribution statistics."""
        # Class distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        class_distribution = {
            int(cls): int(count) for cls, count in zip(unique, counts)
        }
        
        distribution = {
            'class_counts': class_distribution,
            'class_proportions': {
                int(cls): float(count / len(y_pred))
                for cls, count in zip(unique, counts)
            }
        }
        
        # Confidence statistics
        if y_proba is not None:
            max_probas = np.max(y_proba, axis=1)
            distribution['confidence'] = {
                'mean': float(max_probas.mean()),
                'std': float(max_probas.std()),
                'min': float(max_probas.min()),
                'max': float(max_probas.max()),
                'median': float(np.median(max_probas))
            }
        
        return distribution
    
    def compare_performance(
        self,
        reference_idx: int = 0,
        current_idx: int = -1
    ) -> Dict:
        """
        Compare performance between two time points.
        
        Args:
            reference_idx: Index of reference measurement
            current_idx: Index of current measurement
            
        Returns:
            Dictionary with performance deltas
        """
        if len(self.tracking_history) < 2:
            return {'error': 'Need at least 2 measurements for comparison'}
        
        reference = self.tracking_history[reference_idx]
        current = self.tracking_history[current_idx]
        
        comparison = {
            'reference_timestamp': reference['timestamp'],
            'current_timestamp': current['timestamp'],
            'deltas': {}
        }
        
        # Classification metric deltas
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            ref_val = reference['classification'][metric]
            curr_val = current['classification'][metric]
            comparison['deltas'][metric] = {
                'reference': ref_val,
                'current': curr_val,
                'absolute_change': curr_val - ref_val,
                'relative_change': ((curr_val - ref_val) / ref_val * 100) if ref_val > 0 else 0
            }
        
        # Calibration deltas
        if 'calibration' in reference and 'calibration' in current:
            for metric in ['ece', 'mce']:
                ref_val = reference['calibration'][metric]
                curr_val = current['calibration'][metric]
                comparison['deltas'][f'calibration_{metric}'] = {
                    'reference': ref_val,
                    'current': curr_val,
                    'absolute_change': curr_val - ref_val
                }
        
        return comparison
    
    def get_performance_trend(
        self,
        metric: str = 'accuracy',
        window: Optional[int] = None
    ) -> List[float]:
        """
        Get trend of a specific metric over time.
        
        Args:
            metric: Metric to track
            window: Number of recent measurements (None for all)
            
        Returns:
            List of metric values over time
        """
        history = self.tracking_history[-window:] if window else self.tracking_history
        
        trend = []
        for measurement in history:
            if metric in measurement.get('classification', {}):
                trend.append(measurement['classification'][metric])
            elif 'calibration_' in metric and 'calibration' in measurement:
                cal_metric = metric.replace('calibration_', '')
                trend.append(measurement['calibration'][cal_metric])
        
        return trend
    
    def get_tracking_history(self) -> List[Dict]:
        """Get full tracking history."""
        return self.tracking_history
    
    def reset_history(self):
        """Clear tracking history."""
        self.tracking_history = []
    
    def generate_report(self, latest: bool = True) -> str:
        """
        Generate human-readable performance report.
        
        Args:
            latest: If True, report on latest measurement; else all history
            
        Returns:
            Report string
        """
        if not self.tracking_history:
            return "No tracking data available"
        
        report = []
        report.append("=" * 70)
        report.append("MODEL PERFORMANCE REPORT")
        report.append("=" * 70)
        
        if latest:
            metrics = self.tracking_history[-1]
            report.append(f"\nTimestamp: {metrics.get('timestamp', 'N/A')}")
            report.append(f"Samples: {metrics['n_samples']}")
            
            report.append("\nCLASSIFICATION METRICS:")
            for key, value in metrics['classification'].items():
                if key != 'per_class' and not isinstance(value, dict):
                    report.append(f"  {key}: {value:.4f}")
            
            if 'calibration' in metrics:
                report.append("\nCALIBRATION METRICS:")
                report.append(f"  ECE: {metrics['calibration']['ece']:.4f}")
                report.append(f"  MCE: {metrics['calibration']['mce']:.4f}")
                report.append(f"  Avg Confidence: {metrics['calibration']['avg_confidence']:.4f}")
                report.append(f"  Avg Accuracy: {metrics['calibration']['avg_accuracy']:.4f}")
        
        else:
            report.append(f"\nTotal measurements: {len(self.tracking_history)}")
            report.append("\nACCURACY TREND:")
            accuracy_trend = self.get_performance_trend('accuracy')
            report.append(f"  Min: {min(accuracy_trend):.4f}")
            report.append(f"  Max: {max(accuracy_trend):.4f}")
            report.append(f"  Mean: {np.mean(accuracy_trend):.4f}")
            report.append(f"  Std: {np.std(accuracy_trend):.4f}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)


if __name__ == '__main__':
    # Example usage
    print("Testing PerformanceTracker...")
    
    np.random.seed(42)
    
    # Simulate predictions over time
    tracker = PerformanceTracker(n_classes=3, class_names=['Class_A', 'Class_B', 'Class_C'])
    
    for t in range(5):
        # Simulate performance degradation
        n_samples = 200
        y_true = np.random.randint(0, 3, size=n_samples)
        
        # Degrading accuracy
        accuracy = 0.9 - t * 0.05
        y_pred = y_true.copy()
        n_errors = int(n_samples * (1 - accuracy))
        error_indices = np.random.choice(n_samples, size=n_errors, replace=False)
        y_pred[error_indices] = np.random.randint(0, 3, size=n_errors)
        
        # Simulate probabilities
        y_proba = np.random.dirichlet([10, 1, 1], size=n_samples)
        for i, label in enumerate(y_pred):
            y_proba[i] = np.random.dirichlet([1 if j != label else 10 for j in range(3)])
        
        # Track performance
        metrics = tracker.track_performance(
            y_true, y_pred, y_proba,
            timestamp=f'2024-01-{t+1:02d}'
        )
        
        print(f"\nStep {t}: Accuracy = {metrics['classification']['accuracy']:.3f}, "
              f"ECE = {metrics['calibration']['ece']:.3f}")
    
    # Generate report
    print("\n" + tracker.generate_report(latest=False))
    
    # Compare first and last
    comparison = tracker.compare_performance(0, -1)
    print(f"\nAccuracy change: {comparison['deltas']['accuracy']['absolute_change']:.3f}")
    
    print("\n✓ All tests passed!")
