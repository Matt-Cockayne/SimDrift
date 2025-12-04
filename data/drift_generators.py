"""
Drift generation utilities for simulating realistic data and concept drift scenarios.

Supports:
- Gradual drift (equipment aging, demographic shifts)
- Abrupt drift (scanner replacement, new hospital)
- Seasonal drift (temporal variations)
- Covariate shift (feature distribution changes)
- Label shift (class prevalence changes)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable, List
from scipy import ndimage
from sklearn.utils import resample
import cv2


class DriftGenerator:
    """
    Educational drift simulation for medical imaging models.
    Generates realistic drift scenarios to demonstrate monitoring concepts.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.drift_history = []
    
    def simulate_gradual_drift(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        drift_type: str = 'brightness',
        n_steps: int = 100,
        max_severity: float = 0.5,
        return_steps: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate gradual drift over time (e.g., equipment aging).
        
        Args:
            images: Original images (N, H, W, C)
            labels: Original labels
            drift_type: Type of gradual drift
                - 'brightness': Gradual brightness increase (aging sensor)
                - 'contrast': Gradual contrast reduction
                - 'blur': Gradual blurring (lens degradation)
                - 'noise': Gradual noise increase
                - 'demographic': Age/skin type distribution shift
            n_steps: Number of temporal steps
            max_severity: Maximum drift severity (0-1)
            return_steps: If True, return all temporal steps; else only final
            
        Returns:
            drifted_images: Images with applied drift
            drifted_labels: Corresponding labels
            metadata: Drift metadata and severity tracking
        """
        n_samples = len(images)
        samples_per_step = n_samples // n_steps
        
        drifted_images_list = []
        drifted_labels_list = []
        severity_list = []
        
        for step in range(n_steps):
            # Linear severity increase
            severity = (step / (n_steps - 1)) * max_severity
            severity_list.append(severity)
            
            # Get batch for this step
            start_idx = step * samples_per_step
            end_idx = start_idx + samples_per_step if step < n_steps - 1 else n_samples
            batch_images = images[start_idx:end_idx].copy()
            batch_labels = labels[start_idx:end_idx]
            
            # Apply drift based on type
            if drift_type == 'brightness':
                # Simulate sensor aging → gradual brightness increase
                batch_images = self._apply_brightness_shift(batch_images, severity)
                
            elif drift_type == 'contrast':
                # Simulate contrast degradation
                batch_images = self._apply_contrast_reduction(batch_images, severity)
                
            elif drift_type == 'blur':
                # Simulate lens/optics degradation
                batch_images = self._apply_blur(batch_images, severity)
                
            elif drift_type == 'noise':
                # Simulate electronic noise increase
                batch_images = self._apply_noise(batch_images, severity)
                
            elif drift_type == 'demographic':
                # Simulate demographic distribution shift
                batch_images, batch_labels = self._apply_demographic_shift(
                    batch_images, batch_labels, severity
                )
            
            # Enhanced visual drift types
            elif drift_type == 'motion_blur':
                batch_images = self._apply_motion_blur(batch_images, severity)
                
            elif drift_type == 'jpeg_compression':
                batch_images = self._apply_jpeg_compression(batch_images, severity)
                
            elif drift_type == 'occlusion':
                batch_images = self._apply_occlusion(batch_images, severity)
                
            elif drift_type == 'zoom':
                batch_images = self._apply_zoom(batch_images, severity)
                
            elif drift_type == 'vignette':
                batch_images = self._apply_vignette(batch_images, severity)
                
            elif drift_type == 'color_temperature':
                batch_images = self._apply_color_temperature(batch_images, severity)
                
            elif drift_type == 'saturation':
                batch_images = self._apply_saturation(batch_images, severity)
            
            else:
                raise ValueError(f"Unknown drift type: {drift_type}")

            
            drifted_images_list.append(batch_images)
            drifted_labels_list.append(batch_labels)
        
        # Concatenate all steps
        drifted_images = np.concatenate(drifted_images_list, axis=0)
        drifted_labels = np.concatenate(drifted_labels_list, axis=0)
        
        metadata = {
            'drift_type': f'gradual_{drift_type}',
            'n_steps': n_steps,
            'max_severity': max_severity,
            'severity_per_step': severity_list,
            'samples_per_step': samples_per_step
        }
        
        self.drift_history.append(metadata)
        
        if return_steps:
            # Return with temporal structure
            drifted_images = np.array(drifted_images_list)
            drifted_labels = np.array(drifted_labels_list)
        
        return drifted_images, drifted_labels, metadata
    
    def simulate_abrupt_drift(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        drift_type: str = 'new_scanner',
        severity: float = 0.5,
        drift_point: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate abrupt drift at a specific point (e.g., scanner replacement).
        
        Args:
            images: Original images
            labels: Original labels
            drift_type: Type of abrupt drift
                - 'new_scanner': Different scanner characteristics
                - 'new_hospital': Different hospital/population
                - 'protocol_change': Imaging protocol change
                - 'preprocessing_change': Different preprocessing pipeline
            severity: Drift severity (0-1)
            drift_point: When drift occurs (0-1, fraction of samples)
            
        Returns:
            drifted_images: Images with abrupt drift
            drifted_labels: Corresponding labels
            metadata: Drift metadata
        """
        n_samples = len(images)
        drift_idx = int(n_samples * drift_point)
        
        drifted_images = images.copy()
        drifted_labels = labels.copy()
        
        # Apply drift only after drift_point
        if drift_type == 'new_scanner':
            # Simulate different scanner: color shift + different noise profile
            drifted_images[drift_idx:] = self._apply_scanner_artifact(
                drifted_images[drift_idx:], severity
            )
            
        elif drift_type == 'new_hospital':
            # Simulate different population + equipment
            drifted_images[drift_idx:] = self._apply_hospital_artifacts(
                drifted_images[drift_idx:], severity
            )
            
        elif drift_type == 'protocol_change':
            # Simulate protocol change (zoom, rotation, different region)
            drifted_images[drift_idx:] = self._apply_protocol_change(
                drifted_images[drift_idx:], severity
            )
            
        elif drift_type == 'preprocessing_change':
            # Simulate different preprocessing (normalization, filtering)
            drifted_images[drift_idx:] = self._apply_preprocessing_change(
                drifted_images[drift_idx:], severity
            )
        
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")
        
        metadata = {
            'drift_type': f'abrupt_{drift_type}',
            'severity': severity,
            'drift_point': drift_point,
            'drift_idx': drift_idx,
            'affected_samples': n_samples - drift_idx
        }
        
        self.drift_history.append(metadata)
        
        return drifted_images, drifted_labels, metadata
    
    def simulate_seasonal_drift(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_cycles: int = 4,
        amplitude: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate seasonal/periodic drift (e.g., patient demographics over year).
        
        Args:
            images: Original images
            labels: Original labels
            n_cycles: Number of seasonal cycles
            amplitude: Amplitude of seasonal variation
            
        Returns:
            drifted_images: Images with seasonal drift
            drifted_labels: Corresponding labels
            metadata: Drift metadata
        """
        n_samples = len(images)
        drifted_images = images.copy()
        
        # Create sinusoidal drift pattern
        for i in range(n_samples):
            phase = (i / n_samples) * 2 * np.pi * n_cycles
            severity = amplitude * (0.5 + 0.5 * np.sin(phase))
            
            # Apply brightness variation (e.g., seasonal lighting changes)
            drifted_images[i] = self._apply_brightness_shift(
                drifted_images[i:i+1], severity
            )[0]
        
        metadata = {
            'drift_type': 'seasonal',
            'n_cycles': n_cycles,
            'amplitude': amplitude
        }
        
        self.drift_history.append(metadata)
        
        return drifted_images, labels, metadata
    
    def simulate_covariate_shift(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        shift_type: str = 'brightness',
        severity: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate covariate shift (P(X) changes, P(Y|X) stays same).
        
        Args:
            images: Original images
            labels: Original labels  
            shift_type: Type of feature shift
            severity: Shift severity
            
        Returns:
            shifted_images: Images with covariate shift
            labels: Original labels (unchanged)
            metadata: Shift metadata
        """
        shifted_images = images.copy()
        
        if shift_type == 'brightness':
            shifted_images = self._apply_brightness_shift(shifted_images, severity)
        elif shift_type == 'contrast':
            shifted_images = self._apply_contrast_reduction(shifted_images, severity)
        elif shift_type == 'color':
            shifted_images = self._apply_color_shift(shifted_images, severity)
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")
        
        metadata = {
            'drift_type': 'covariate_shift',
            'shift_type': shift_type,
            'severity': severity
        }
        
        self.drift_history.append(metadata)
        
        return shifted_images, labels, metadata
    
    def simulate_label_shift(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        target_distribution: Optional[Dict[int, float]] = None,
        severity: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate label shift (P(Y) changes, P(X|Y) stays same).
        
        Args:
            images: Original images
            labels: Original labels
            target_distribution: Target class distribution {class: proportion}
            severity: How much to shift toward target distribution
            
        Returns:
            shifted_images: Resampled images
            shifted_labels: Resampled labels with new distribution
            metadata: Shift metadata
        """
        unique_classes = np.unique(labels.flatten())
        n_samples = len(images)
        
        # Default: favor majority class
        if target_distribution is None:
            majority_class = np.argmax(np.bincount(labels.flatten()))
            target_distribution = {
                cls: 0.7 if cls == majority_class else 0.3 / (len(unique_classes) - 1)
                for cls in unique_classes
            }
        
        # Calculate current distribution
        current_dist = {
            cls: np.sum(labels == cls) / n_samples
            for cls in unique_classes
        }
        
        # Interpolate between current and target
        new_dist = {
            cls: current_dist[cls] * (1 - severity) + target_distribution[cls] * severity
            for cls in unique_classes
        }
        
        # Resample to achieve new distribution
        new_samples_per_class = {
            cls: int(n_samples * new_dist[cls])
            for cls in unique_classes
        }
        
        resampled_images = []
        resampled_labels = []
        
        for cls in unique_classes:
            class_mask = labels.flatten() == cls
            class_images = images[class_mask]
            class_labels = labels[class_mask]
            
            n_needed = new_samples_per_class[cls]
            if n_needed > 0:
                resampled_cls_images, resampled_cls_labels = resample(
                    class_images, class_labels,
                    n_samples=n_needed,
                    replace=True,
                    random_state=self.seed
                )
                resampled_images.append(resampled_cls_images)
                resampled_labels.append(resampled_cls_labels)
        
        shifted_images = np.concatenate(resampled_images, axis=0)
        shifted_labels = np.concatenate(resampled_labels, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(shifted_images))
        shifted_images = shifted_images[indices]
        shifted_labels = shifted_labels[indices]
        
        metadata = {
            'drift_type': 'label_shift',
            'original_distribution': current_dist,
            'target_distribution': target_distribution,
            'new_distribution': new_dist,
            'severity': severity
        }
        
        self.drift_history.append(metadata)
        
        return shifted_images, shifted_labels, metadata
    
    # ==================== Private helper methods ====================
    
    def _apply_brightness_shift(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Add brightness shift."""
        shift = severity * 50  # Pixel value shift
        return np.clip(images + shift, 0, 255).astype(images.dtype)
    
    def _apply_contrast_reduction(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Reduce contrast."""
        factor = 1.0 - severity * 0.5
        mean = images.mean(axis=(1, 2, 3), keepdims=True)
        return np.clip((images - mean) * factor + mean, 0, 255).astype(images.dtype)
    
    def _apply_blur(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply Gaussian blur."""
        sigma = severity * 3.0
        blurred = np.zeros_like(images)
        for i in range(len(images)):
            for c in range(images.shape[-1]):
                blurred[i, :, :, c] = ndimage.gaussian_filter(
                    images[i, :, :, c], sigma=sigma
                )
        return blurred.astype(images.dtype)
    
    def _apply_noise(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise_std = severity * 30
        noise = np.random.normal(0, noise_std, images.shape)
        return np.clip(images + noise, 0, 255).astype(images.dtype)
    
    def _apply_color_shift(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply color shift (if RGB)."""
        if images.shape[-1] == 3:
            # Shift each channel differently
            shifts = np.random.uniform(-severity * 30, severity * 30, size=(1, 1, 1, 3))
            return np.clip(images + shifts, 0, 255).astype(images.dtype)
        return images
    
    def _apply_demographic_shift(
        self, images: np.ndarray, labels: np.ndarray, severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate demographic distribution shift via resampling."""
        # Simple implementation: oversample certain classes
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            target_class = unique_labels[0]
            target_weight = 1.0 + severity
            
            weights = np.where(labels.flatten() == target_class, target_weight, 1.0)
            weights = weights / weights.sum()
            
            indices = np.random.choice(
                len(images), size=len(images), replace=True, p=weights
            )
            return images[indices], labels[indices]
        return images, labels
    
    def _apply_scanner_artifact(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Simulate different scanner artifacts."""
        # Combine brightness shift and different noise pattern
        images = self._apply_brightness_shift(images, severity * 0.3)
        images = self._apply_noise(images, severity * 0.5)
        return images
    
    def _apply_hospital_artifacts(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Simulate different hospital artifacts."""
        # Multiple transformations
        images = self._apply_color_shift(images, severity * 0.4)
        images = self._apply_contrast_reduction(images, severity * 0.3)
        return images
    
    def _apply_protocol_change(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Simulate imaging protocol change."""
        # Crop/zoom simulation
        crop_amount = int(severity * images.shape[1] * 0.1)
        if crop_amount > 0:
            images = images[:, crop_amount:-crop_amount, crop_amount:-crop_amount, :]
            # Resize back to original (simple repeat for now)
            from scipy.ndimage import zoom
            zoom_factor = images.shape[1] / (images.shape[1] - 2 * crop_amount)
            images = zoom(images, (1, zoom_factor, zoom_factor, 1), order=1)
        return images.astype(np.uint8)
    
    def _apply_preprocessing_change(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Simulate different preprocessing."""
        # Different normalization approach
        new_mean = 128 + severity * 30
        new_std = 50 - severity * 20
        
        normalized = (images - images.mean()) / (images.std() + 1e-8)
        denormalized = normalized * new_std + new_mean
        
        return np.clip(denormalized, 0, 255).astype(images.dtype)
    
    # ==================== Enhanced visual drift methods ====================
    
    def _apply_motion_blur(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply motion blur (simulates patient movement)."""
        kernel_size = max(3, int(severity * 15))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        blurred = np.zeros_like(images)
        for i in range(len(images)):
            for c in range(images.shape[-1]):
                blurred[i, :, :, c] = cv2.filter2D(
                    images[i, :, :, c], -1, kernel
                )
        return blurred.astype(images.dtype)
    
    def _apply_jpeg_compression(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        quality = int(100 - severity * 80)  # Lower quality = more compression
        quality = max(1, quality)
        
        compressed = np.zeros_like(images)
        for i in range(len(images)):
            # Encode and decode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', images[i], encode_param)
            compressed[i] = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        
        return compressed.astype(images.dtype)
    
    def _apply_occlusion(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply random occlusions (simulates obstructions)."""
        occluded = images.copy()
        h, w = images.shape[1:3]
        
        # Size of occlusion based on severity
        occ_size = int(severity * min(h, w) * 0.3)
        
        if occ_size > 0:
            for i in range(len(images)):
                # Random position
                x = np.random.randint(0, w - occ_size + 1)
                y = np.random.randint(0, h - occ_size + 1)
                
                # Black rectangle occlusion
                occluded[i, y:y+occ_size, x:x+occ_size, :] = 0
        
        return occluded
    
    def _apply_zoom(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply zoom/crop (simulates field of view changes)."""
        h, w = images.shape[1:3]
        
        # Zoom in by cropping and resizing
        crop_percent = severity * 0.3  # Up to 30% crop
        crop_h = int(h * crop_percent / 2)
        crop_w = int(w * crop_percent / 2)
        
        if crop_h > 0 and crop_w > 0:
            zoomed = np.zeros_like(images)
            for i in range(len(images)):
                cropped = images[i, crop_h:-crop_h, crop_w:-crop_w, :]
                zoomed[i] = cv2.resize(cropped, (w, h))
            return zoomed.astype(images.dtype)
        
        return images
    
    def _apply_color_temperature(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply color temperature shift (warm to cool)."""
        if images.shape[-1] != 3:
            return images  # Only for RGB
        
        shifted = images.copy().astype(float)
        
        # Shift towards warm (positive) or cool (negative)
        shift = (severity - 0.5) * 2  # Map [0,1] to [-1,1]
        
        if shift > 0:  # Warm (more red/yellow)
            shifted[:, :, :, 0] += shift * 30  # Red channel
            shifted[:, :, :, 2] -= shift * 15  # Blue channel
        else:  # Cool (more blue)
            shifted[:, :, :, 2] -= shift * 30  # Blue channel
            shifted[:, :, :, 0] += shift * 15  # Red channel
        
        return np.clip(shifted, 0, 255).astype(images.dtype)
    
    def _apply_saturation(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply saturation change."""
        if images.shape[-1] != 3:
            return images
        
        adjusted = np.zeros_like(images)
        factor = 1.0 + (severity - 0.5) * 2  # Map [0,1] to [0,2]
        
        for i in range(len(images)):
            # Convert to HSV, adjust saturation, convert back
            hsv = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV).astype(float)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            adjusted[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return adjusted.astype(images.dtype)
    
    def _apply_vignette(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Apply vignette effect (darkening at edges)."""
        h, w = images.shape[1:3]
        
        # Create radial gradient
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Vignette mask (1 at center, decreases towards edges)
        vignette_mask = 1 - (dist / max_dist) ** 2 * severity
        vignette_mask = np.clip(vignette_mask, 0, 1)
        
        # Apply vignette
        vignetted = images * vignette_mask[np.newaxis, :, :, np.newaxis]
        
        return np.clip(vignetted, 0, 255).astype(images.dtype)
    
    def simulate_gradual_drift_with_steps(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        drift_type: str = 'brightness',
        n_steps: int = 50,
        max_severity: float = 0.5
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Simulate gradual drift and return all intermediate steps.
        
        This is useful for creating animations or real-time drift visualization.
        
        Args:
            images: Original images
            labels: Original labels
            drift_type: Type of drift to apply
            n_steps: Number of steps in drift progression
            max_severity: Maximum severity at final step
            
        Returns:
            List of image arrays (one per step)
            List of label arrays (one per step)
            List of severity values (one per step)
        """
        all_images = []
        all_labels = []
        all_severities = []
        
        for step in range(n_steps):
            severity = (step / (n_steps - 1)) * max_severity
            all_severities.append(severity)
            
            # Apply drift to entire dataset
            if drift_type == 'brightness':
                drifted = self._apply_brightness_shift(images.copy(), severity)
            elif drift_type == 'contrast':
                drifted = self._apply_contrast_reduction(images.copy(), severity)
            elif drift_type == 'blur':
                drifted = self._apply_blur(images.copy(), severity)
            elif drift_type == 'noise':
                drifted = self._apply_noise(images.copy(), severity)
            elif drift_type == 'motion_blur':
                drifted = self._apply_motion_blur(images.copy(), severity)
            elif drift_type == 'jpeg_compression':
                drifted = self._apply_jpeg_compression(images.copy(), severity)
            elif drift_type == 'occlusion':
                drifted = self._apply_occlusion(images.copy(), severity)
            elif drift_type == 'zoom':
                drifted = self._apply_zoom(images.copy(), severity)
            elif drift_type == 'vignette':
                drifted = self._apply_vignette(images.copy(), severity)
            elif drift_type == 'color_temperature':
                drifted = self._apply_color_temperature(images.copy(), severity)
            elif drift_type == 'saturation':
                drifted = self._apply_saturation(images.copy(), severity)
            else:
                drifted = images.copy()
            
            all_images.append(drifted)
            all_labels.append(labels.copy())
        
        return all_images, all_labels, all_severities
    
    def get_available_drift_types(self) -> Dict[str, str]:
        """
        Get all available drift types with descriptions.
        
        Returns:
            Dictionary mapping drift type names to descriptions
        """
        return {
            # Original drift types
            'brightness': 'Gradual brightness increase (aging sensor)',
            'contrast': 'Gradual contrast reduction',
            'blur': 'Gaussian blur (lens degradation)',
            'noise': 'Gaussian noise increase (electronic noise)',
            'demographic': 'Age/population distribution shift',
            
            # Enhanced visual drift types
            'motion_blur': 'Motion blur (patient movement)',
            'jpeg_compression': 'JPEG compression artifacts',
            'occlusion': 'Random occlusions (obstructions)',
            'zoom': 'Field of view changes (zoom/crop)',
            'vignette': 'Edge darkening (vignette effect)',
            'color_temperature': 'Color temperature shift (warm/cool)',
            'saturation': 'Color saturation changes',
            
            # Combined drift types
            'scanner_aging': 'Combined brightness + noise (equipment aging)',
            'new_scanner': 'Abrupt scanner replacement',
            'hospital_transfer': 'Hospital-specific artifacts',
            'protocol_change': 'Imaging protocol update'
        }
    
    def get_drift_history(self) -> list:
        """Get history of all drift operations."""
        return self.drift_history
    
    def reset_history(self):
        """Clear drift history."""
        self.drift_history = []


if __name__ == '__main__':
    # Example usage
    print("Testing DriftGenerator...")
    
    # Create dummy data
    np.random.seed(42)
    images = np.random.randint(0, 255, size=(100, 28, 28, 3), dtype=np.uint8)
    labels = np.random.randint(0, 5, size=(100, 1))
    
    generator = DriftGenerator(seed=42)
    
    # Test gradual drift
    print("\n1. Gradual brightness drift...")
    drifted_img, drifted_lbl, meta = generator.simulate_gradual_drift(
        images, labels, drift_type='brightness', n_steps=10, max_severity=0.5, return_steps=False
    )
    print(f"   Shape: {drifted_img.shape}, Severity: {meta['max_severity']}")
    
    # Test abrupt drift
    print("\n2. Abrupt scanner replacement...")
    drifted_img, drifted_lbl, meta = generator.simulate_abrupt_drift(
        images, labels, drift_type='new_scanner', severity=0.6
    )
    print(f"   Drift at sample {meta['drift_idx']}, affected: {meta['affected_samples']}")
    
    # Test label shift
    print("\n3. Label distribution shift...")
    drifted_img, drifted_lbl, meta = generator.simulate_label_shift(
        images, labels, severity=0.7
    )
    print(f"   New distribution: {meta['new_distribution']}")
    
    print("\n✓ All drift simulations working!")
    print(f"\nDrift history: {len(generator.get_drift_history())} operations")
