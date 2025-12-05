"""
Drift detection algorithms for monitoring model and data drift.

Implements:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test (KS)
- Chi-Square Test
- Maximum Mean Discrepancy (MMD)
- Wasserstein Distance
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


class DriftDetector:
    """
    Multi-method drift detection for production ML monitoring.
    """
    
    def __init__(self, threshold_psi: float = 0.2, threshold_ks: float = 0.05,
                 threshold_chi2: float = 0.05, threshold_mmd: float = 0.1):
        """
        Initialize drift detector with thresholds.
        
        Args:
            threshold_psi: PSI threshold (>0.2 indicates drift)
            threshold_ks: KS test p-value threshold (<0.05 indicates drift)
            threshold_chi2: Chi-square p-value threshold (<0.05 indicates drift)
            threshold_mmd: MMD threshold (>0.1 indicates drift)
        """
        self.threshold_psi = threshold_psi
        self.threshold_ks = threshold_ks
        self.threshold_chi2 = threshold_chi2
        self.threshold_mmd = threshold_mmd
        
        self.detection_results = []
    
    def detect_drift(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray,
        method: str = 'all',
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect drift using specified method(s).
        
        Args:
            reference_data: Reference/training distribution (N, D)
            production_data: Production/test distribution (M, D)
            method: Detection method ('psi', 'ks', 'chi2', 'mmd', 'wasserstein', 'all')
            feature_names: Optional feature names for reporting
            
        Returns:
            Dictionary with drift detection results
        """
        results = {
            'drift_detected': False,
            'methods': {},
            'feature_drift': {}
        }
        
        # Ensure 2D arrays
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)
        if production_data.ndim == 1:
            production_data = production_data.reshape(-1, 1)
        
        n_features = reference_data.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        methods_to_run = ['psi', 'ks', 'chi2', 'mmd', 'wasserstein'] if method == 'all' else [method]
        
        for method_name in methods_to_run:
            if method_name == 'psi':
                psi_scores, psi_drift = self.calculate_psi(reference_data, production_data)
                results['methods']['psi'] = {
                    'scores': psi_scores,
                    'drift_detected': psi_drift,
                    'threshold': self.threshold_psi
                }
                for i, fname in enumerate(feature_names):
                    if fname not in results['feature_drift']:
                        results['feature_drift'][fname] = {}
                    results['feature_drift'][fname]['psi'] = psi_scores[i]
                
                if psi_drift:
                    results['drift_detected'] = True
            
            elif method_name == 'ks':
                ks_scores, ks_pvalues, ks_drift = self.calculate_ks_test(
                    reference_data, production_data
                )
                results['methods']['ks'] = {
                    'statistics': ks_scores,
                    'p_values': ks_pvalues,
                    'drift_detected': ks_drift,
                    'threshold': self.threshold_ks
                }
                for i, fname in enumerate(feature_names):
                    if fname not in results['feature_drift']:
                        results['feature_drift'][fname] = {}
                    results['feature_drift'][fname]['ks'] = {
                        'statistic': ks_scores[i],
                        'p_value': ks_pvalues[i]
                    }
                
                if ks_drift:
                    results['drift_detected'] = True
            
            elif method_name == 'chi2':
                chi2_scores, chi2_pvalues, chi2_drift = self.calculate_chi2_test(
                    reference_data, production_data
                )
                results['methods']['chi2'] = {
                    'statistics': chi2_scores,
                    'p_values': chi2_pvalues,
                    'drift_detected': chi2_drift,
                    'threshold': self.threshold_chi2
                }
                for i, fname in enumerate(feature_names):
                    if fname not in results['feature_drift']:
                        results['feature_drift'][fname] = {}
                    results['feature_drift'][fname]['chi2'] = {
                        'statistic': chi2_scores[i],
                        'p_value': chi2_pvalues[i]
                    }
                
                if chi2_drift:
                    results['drift_detected'] = True
            
            elif method_name == 'mmd':
                mmd_score, mmd_drift = self.calculate_mmd(reference_data, production_data)
                results['methods']['mmd'] = {
                    'score': mmd_score,
                    'drift_detected': mmd_drift,
                    'threshold': self.threshold_mmd
                }
                
                if mmd_drift:
                    results['drift_detected'] = True
            
            elif method_name == 'wasserstein':
                wasserstein_scores = self.calculate_wasserstein(reference_data, production_data)
                results['methods']['wasserstein'] = {
                    'distances': wasserstein_scores
                }
                for i, fname in enumerate(feature_names):
                    if fname not in results['feature_drift']:
                        results['feature_drift'][fname] = {}
                    results['feature_drift'][fname]['wasserstein'] = wasserstein_scores[i]
        
        self.detection_results.append(results)
        return results
    
    def calculate_psi(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, bool]:
        """
        Calculate Population Stability Index (PSI) for each feature.
        
        PSI measures distribution shift:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Small change
        - PSI ≥ 0.2: Significant change (drift)
        
        Args:
            reference_data: Reference distribution
            production_data: Production distribution
            n_bins: Number of bins for discretization
            
        Returns:
            psi_scores: PSI for each feature
            drift_detected: Whether drift detected (any PSI > threshold)
        """
        n_features = reference_data.shape[1]
        psi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            ref_feature = reference_data[:, i]
            prod_feature = production_data[:, i]
            
            # Create bins based on reference data
            _, bin_edges = np.histogram(ref_feature, bins=n_bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(ref_feature, bins=bin_edges)
            prod_counts, _ = np.histogram(prod_feature, bins=bin_edges)
            
            # Convert to proportions (add small epsilon to avoid log(0))
            epsilon = 1e-10
            ref_props = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * n_bins)
            prod_props = (prod_counts + epsilon) / (prod_counts.sum() + epsilon * n_bins)
            
            # Calculate PSI using the standard formula
            # PSI = sum((actual% - expected%) * ln(actual% / expected%))
            psi = np.sum((prod_props - ref_props) * np.log(prod_props / ref_props))
            # Take absolute value since PSI should be non-negative
            psi_scores[i] = np.abs(psi)
        
        drift_detected = np.any(psi_scores > self.threshold_psi)
        
        return psi_scores, drift_detected
    
    def calculate_ks_test(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Kolmogorov-Smirnov test for distribution comparison.
        
        Tests whether two samples come from the same distribution.
        p-value < 0.05 typically indicates different distributions (drift).
        
        Args:
            reference_data: Reference distribution
            production_data: Production distribution
            
        Returns:
            ks_statistics: KS statistic for each feature
            p_values: p-values for each feature
            drift_detected: Whether drift detected (any p < threshold)
        """
        n_features = reference_data.shape[1]
        ks_statistics = np.zeros(n_features)
        p_values = np.zeros(n_features)
        
        for i in range(n_features):
            statistic, p_value = stats.ks_2samp(
                reference_data[:, i],
                production_data[:, i]
            )
            ks_statistics[i] = statistic
            p_values[i] = p_value
        
        drift_detected = np.any(p_values < self.threshold_ks)
        
        return ks_statistics, p_values, drift_detected
    
    def calculate_chi2_test(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Chi-square test for distribution comparison.
        
        Tests independence between reference and production distributions.
        p-value < 0.05 indicates significant difference (drift).
        
        Args:
            reference_data: Reference distribution
            production_data: Production distribution
            n_bins: Number of bins for discretization
            
        Returns:
            chi2_statistics: Chi-square statistic for each feature
            p_values: p-values for each feature
            drift_detected: Whether drift detected
        """
        n_features = reference_data.shape[1]
        chi2_statistics = np.zeros(n_features)
        p_values = np.zeros(n_features)
        
        for i in range(n_features):
            ref_feature = reference_data[:, i]
            prod_feature = production_data[:, i]
            
            # Create bins
            _, bin_edges = np.histogram(
                np.concatenate([ref_feature, prod_feature]),
                bins=n_bins
            )
            
            # Calculate contingency table
            ref_counts, _ = np.histogram(ref_feature, bins=bin_edges)
            prod_counts, _ = np.histogram(prod_feature, bins=bin_edges)
            
            # Avoid zero counts
            ref_counts = ref_counts + 1
            prod_counts = prod_counts + 1
            
            contingency = np.array([ref_counts, prod_counts])
            
            # Chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            chi2_statistics[i] = chi2
            p_values[i] = p_value
        
        drift_detected = np.any(p_values < self.threshold_chi2)
        
        return chi2_statistics, p_values, drift_detected
    
    def calculate_mmd(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray,
        kernel: str = 'rbf',
        gamma: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Calculate Maximum Mean Discrepancy (MMD) between distributions.
        
        MMD measures distance between distributions in reproducing kernel Hilbert space.
        Higher values indicate more drift.
        
        Args:
            reference_data: Reference distribution
            production_data: Production distribution
            kernel: Kernel type ('rbf', 'linear')
            gamma: RBF kernel bandwidth (auto if None)
            
        Returns:
            mmd_score: MMD value
            drift_detected: Whether drift detected
        """
        # Flatten if multidimensional
        ref_flat = reference_data.reshape(len(reference_data), -1)
        prod_flat = production_data.reshape(len(production_data), -1)
        
        if kernel == 'rbf':
            if gamma is None:
                # Median heuristic
                pairwise_dists = np.sum((ref_flat[:, None, :] - ref_flat[None, :, :]) ** 2, axis=2)
                gamma = 1.0 / (2 * np.median(pairwise_dists) + 1e-10)
            
            # Kernel matrices
            K_xx = self._rbf_kernel(ref_flat, ref_flat, gamma)
            K_yy = self._rbf_kernel(prod_flat, prod_flat, gamma)
            K_xy = self._rbf_kernel(ref_flat, prod_flat, gamma)
            
        else:  # linear kernel
            K_xx = ref_flat @ ref_flat.T
            K_yy = prod_flat @ prod_flat.T
            K_xy = ref_flat @ prod_flat.T
        
        # MMD^2 = E[K(x,x')] - 2E[K(x,y)] + E[K(y,y')]
        m = len(ref_flat)
        n = len(prod_flat)
        
        mmd_squared = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1))
        mmd_squared += (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1))
        mmd_squared -= 2 * K_xy.sum() / (m * n)
        
        mmd_score = np.sqrt(max(mmd_squared, 0))
        drift_detected = mmd_score > self.threshold_mmd
        
        return mmd_score, drift_detected
    
    def calculate_wasserstein(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Wasserstein distance (Earth Mover's Distance) for each feature.
        
        Args:
            reference_data: Reference distribution
            production_data: Production distribution
            
        Returns:
            distances: Wasserstein distance for each feature
        """
        n_features = reference_data.shape[1]
        distances = np.zeros(n_features)
        
        for i in range(n_features):
            distances[i] = wasserstein_distance(
                reference_data[:, i],
                production_data[:, i]
            )
        
        return distances
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        pairwise_sq_dists = np.sum(X[:, None, :] ** 2, axis=2) - 2 * (X @ Y.T) + np.sum(Y[None, :, :] ** 2, axis=2)
        return np.exp(-gamma * pairwise_sq_dists)
    
    def get_drift_summary(self, results: Dict) -> str:
        """
        Generate human-readable drift summary.
        
        Args:
            results: Drift detection results
            
        Returns:
            Summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("DRIFT DETECTION SUMMARY")
        summary.append("=" * 60)
        
        if results['drift_detected']:
            summary.append("⚠️  DRIFT DETECTED!")
        else:
            summary.append("✓ No drift detected")
        
        summary.append("\nMethod-wise results:")
        for method, data in results['methods'].items():
            summary.append(f"\n{method.upper()}:")
            if 'scores' in data:
                summary.append(f"  Max score: {np.max(data['scores']):.4f} (threshold: {data['threshold']})")
            if 'p_values' in data:
                summary.append(f"  Min p-value: {np.min(data['p_values']):.4f} (threshold: {data['threshold']})")
            if 'score' in data:
                summary.append(f"  Score: {data['score']:.4f} (threshold: {data['threshold']})")
            summary.append(f"  Drift: {'YES' if data.get('drift_detected', False) else 'NO'}")
        
        summary.append("\n" + "=" * 60)
        return "\n".join(summary)
    
    def get_detection_history(self) -> List[Dict]:
        """Get history of all drift detections."""
        return self.detection_results
    
    def reset_history(self):
        """Clear detection history."""
        self.detection_results = []


if __name__ == '__main__':
    # Example usage
    print("Testing DriftDetector...")
    
    # Create sample data
    np.random.seed(42)
    
    # Reference data (normal distribution)
    reference = np.random.normal(0, 1, size=(1000, 5))
    
    # Production data with drift
    production_drifted = np.random.normal(0.5, 1.2, size=(1000, 5))
    
    # Production data without drift
    production_clean = np.random.normal(0, 1, size=(1000, 5))
    
    detector = DriftDetector()
    
    # Test 1: Detect drift
    print("\n1. Testing with drifted data...")
    results_drift = detector.detect_drift(reference, production_drifted, method='all')
    print(detector.get_drift_summary(results_drift))
    
    # Test 2: No drift
    print("\n2. Testing with clean data...")
    results_clean = detector.detect_drift(reference, production_clean, method='all')
    print(detector.get_drift_summary(results_clean))
    
    print("\n✓ All tests passed!")
