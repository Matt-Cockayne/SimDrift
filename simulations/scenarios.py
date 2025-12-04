"""
Pre-configured realistic drift scenarios for medical imaging.

Scenarios simulate real-world drift situations:
- Equipment aging
- Scanner replacement
- Hospital transfer
- Demographic shifts
- Seasonal variations
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.drift_generators import DriftGenerator


@dataclass
class DriftScenario:
    """
    Configuration for a drift scenario.
    """
    name: str
    description: str
    drift_type: str  # 'gradual', 'abrupt', 'seasonal', 'covariate', 'label'
    parameters: Dict
    expected_detection_methods: list  # Which methods should detect this
    severity: str  # 'mild', 'moderate', 'severe'
    category: str  # 'data_drift', 'concept_drift', 'combined'
    
    def __str__(self):
        return f"{self.name} ({self.severity}): {self.description}"


class ScenarioLibrary:
    """
    Library of pre-configured realistic drift scenarios.
    """
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, DriftScenario]:
        """Get all available scenarios."""
        return {
            # ============ GRADUAL DRIFT SCENARIOS ============
            'equipment_aging': DriftScenario(
                name="Equipment Aging",
                description="Medical imaging equipment gradually degrades over time, causing subtle brightness and contrast changes. Simulates sensor aging over 6 months.",
                drift_type='gradual',
                parameters={
                    'drift_type': 'brightness',
                    'n_steps': 50,
                    'max_severity': 0.4,
                    'return_steps': False
                },
                expected_detection_methods=['psi', 'ks', 'wasserstein'],
                severity='moderate',
                category='data_drift'
            ),
            
            'lens_degradation': DriftScenario(
                name="Lens/Optics Degradation",
                description="Optical components degrade, causing gradual image blur. Common in dermatology cameras and fundus imaging.",
                drift_type='gradual',
                parameters={
                    'drift_type': 'blur',
                    'n_steps': 40,
                    'max_severity': 0.5,
                    'return_steps': False
                },
                expected_detection_methods=['psi', 'ks', 'mmd'],
                severity='moderate',
                category='data_drift'
            ),
            
            'demographic_shift': DriftScenario(
                name="Demographic Shift",
                description="Patient population gradually changes (age, ethnicity, skin type). Common when expanding to new geographic regions.",
                drift_type='gradual',
                parameters={
                    'drift_type': 'demographic',
                    'n_steps': 30,
                    'max_severity': 0.6,
                    'return_steps': False
                },
                expected_detection_methods=['chi2', 'psi', 'mmd'],
                severity='severe',
                category='combined'  # Both data and concept drift
            ),
            
            'noise_increase': DriftScenario(
                name="Electronic Noise Increase",
                description="Electronic noise gradually increases due to equipment wear or interference. Affects image quality over time.",
                drift_type='gradual',
                parameters={
                    'drift_type': 'noise',
                    'n_steps': 45,
                    'max_severity': 0.35,
                    'return_steps': False
                },
                expected_detection_methods=['psi', 'ks', 'wasserstein'],
                severity='mild',
                category='data_drift'
            ),
            
            # ============ ABRUPT DRIFT SCENARIOS ============
            'scanner_replacement': DriftScenario(
                name="Scanner Replacement",
                description="Hospital replaces imaging equipment with a different model/manufacturer. Immediate distribution shift at deployment.",
                drift_type='abrupt',
                parameters={
                    'drift_type': 'new_scanner',
                    'severity': 0.6,
                    'drift_point': 0.5
                },
                expected_detection_methods=['psi', 'ks', 'chi2', 'mmd'],
                severity='severe',
                category='data_drift'
            ),
            
            'hospital_transfer': DriftScenario(
                name="Hospital Transfer",
                description="Model deployed to new hospital with different equipment, protocols, and patient demographics.",
                drift_type='abrupt',
                parameters={
                    'drift_type': 'new_hospital',
                    'severity': 0.7,
                    'drift_point': 0.4
                },
                expected_detection_methods=['psi', 'ks', 'chi2', 'mmd'],
                severity='severe',
                category='combined'
            ),
            
            'protocol_change': DriftScenario(
                name="Imaging Protocol Change",
                description="Hospital updates imaging protocol (zoom level, region of interest, preprocessing). Common after guideline updates.",
                drift_type='abrupt',
                parameters={
                    'drift_type': 'protocol_change',
                    'severity': 0.5,
                    'drift_point': 0.6
                },
                expected_detection_methods=['psi', 'mmd', 'wasserstein'],
                severity='moderate',
                category='data_drift'
            ),
            
            'preprocessing_update': DriftScenario(
                name="Preprocessing Pipeline Update",
                description="Software update changes image preprocessing (normalization, color correction). Often undocumented.",
                drift_type='abrupt',
                parameters={
                    'drift_type': 'preprocessing_change',
                    'severity': 0.45,
                    'drift_point': 0.5
                },
                expected_detection_methods=['psi', 'ks', 'mmd'],
                severity='moderate',
                category='data_drift'
            ),
            
            # ============ SEASONAL DRIFT SCENARIOS ============
            'seasonal_patient_flow': DriftScenario(
                name="Seasonal Patient Demographics",
                description="Patient demographics vary seasonally (e.g., more elderly patients in winter, more outdoor workers in summer).",
                drift_type='seasonal',
                parameters={
                    'n_cycles': 3,
                    'amplitude': 0.4
                },
                expected_detection_methods=['psi', 'chi2'],
                severity='mild',
                category='combined'
            ),
            
            'seasonal_lighting': DriftScenario(
                name="Seasonal Lighting Variation",
                description="Ambient lighting affects image capture (window light, seasonal sun angle). Common in dermatology clinics.",
                drift_type='seasonal',
                parameters={
                    'n_cycles': 4,
                    'amplitude': 0.3
                },
                expected_detection_methods=['psi', 'wasserstein'],
                severity='mild',
                category='data_drift'
            ),
            
            # ============ COVARIATE SHIFT SCENARIOS ============
            'brightness_shift': DriftScenario(
                name="Brightness Covariate Shift",
                description="Systematic brightness shift (P(X) changes, P(Y|X) constant). Tests feature-level drift detection.",
                drift_type='covariate',
                parameters={
                    'shift_type': 'brightness',
                    'severity': 0.5
                },
                expected_detection_methods=['psi', 'ks', 'wasserstein'],
                severity='moderate',
                category='data_drift'
            ),
            
            'contrast_shift': DriftScenario(
                name="Contrast Covariate Shift",
                description="Systematic contrast change without affecting diagnosis. Tests pure covariate shift.",
                drift_type='covariate',
                parameters={
                    'shift_type': 'contrast',
                    'severity': 0.4
                },
                expected_detection_methods=['psi', 'ks'],
                severity='mild',
                category='data_drift'
            ),
            
            'color_shift': DriftScenario(
                name="Color Calibration Shift",
                description="Color calibration changes (white balance, color temperature). Common in RGB medical imaging.",
                drift_type='covariate',
                parameters={
                    'shift_type': 'color',
                    'severity': 0.5
                },
                expected_detection_methods=['psi', 'ks', 'mmd'],
                severity='moderate',
                category='data_drift'
            ),
            
            # ============ LABEL SHIFT SCENARIOS ============
            'prevalence_shift_mild': DriftScenario(
                name="Disease Prevalence Shift (Mild)",
                description="Disease prevalence changes slightly (e.g., outbreak in community). P(Y) changes, P(X|Y) constant.",
                drift_type='label',
                parameters={
                    'target_distribution': None,  # Will be set dynamically
                    'severity': 0.3
                },
                expected_detection_methods=['chi2', 'psi'],
                severity='mild',
                category='concept_drift'
            ),
            
            'prevalence_shift_severe': DriftScenario(
                name="Disease Prevalence Shift (Severe)",
                description="Major prevalence change (epidemic, screening program targeting high-risk). Significant class imbalance shift.",
                drift_type='label',
                parameters={
                    'target_distribution': None,
                    'severity': 0.7
                },
                expected_detection_methods=['chi2', 'psi'],
                severity='severe',
                category='concept_drift'
            ),
        }
    
    @staticmethod
    def get_scenario(scenario_name: str) -> Optional[DriftScenario]:
        """Get a specific scenario by name."""
        scenarios = ScenarioLibrary.get_all_scenarios()
        return scenarios.get(scenario_name)
    
    @staticmethod
    def list_scenarios_by_category(category: str = None) -> Dict[str, DriftScenario]:
        """List scenarios by category."""
        scenarios = ScenarioLibrary.get_all_scenarios()
        
        if category is None:
            return scenarios
        
        return {
            name: scenario
            for name, scenario in scenarios.items()
            if scenario.category == category
        }
    
    @staticmethod
    def list_scenarios_by_severity(severity: str) -> Dict[str, DriftScenario]:
        """List scenarios by severity."""
        scenarios = ScenarioLibrary.get_all_scenarios()
        
        return {
            name: scenario
            for name, scenario in scenarios.items()
            if scenario.severity == severity
        }


class ScenarioRunner:
    """
    Execute drift scenarios on data.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize scenario runner.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.drift_generator = DriftGenerator(seed=seed)
        self.scenario_results = []
    
    def run_scenario(
        self,
        scenario: DriftScenario,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Execute a drift scenario.
        
        Args:
            scenario: DriftScenario configuration
            images: Original images
            labels: Original labels
            
        Returns:
            drifted_images: Images with applied drift
            drifted_labels: Corresponding labels
            metadata: Scenario execution metadata
        """
        # Select appropriate drift generation method
        if scenario.drift_type == 'gradual':
            drifted_images, drifted_labels, drift_meta = self.drift_generator.simulate_gradual_drift(
                images, labels, **scenario.parameters
            )
        
        elif scenario.drift_type == 'abrupt':
            drifted_images, drifted_labels, drift_meta = self.drift_generator.simulate_abrupt_drift(
                images, labels, **scenario.parameters
            )
        
        elif scenario.drift_type == 'seasonal':
            drifted_images, drifted_labels, drift_meta = self.drift_generator.simulate_seasonal_drift(
                images, labels, **scenario.parameters
            )
        
        elif scenario.drift_type == 'covariate':
            drifted_images, drifted_labels, drift_meta = self.drift_generator.simulate_covariate_shift(
                images, labels, **scenario.parameters
            )
        
        elif scenario.drift_type == 'label':
            drifted_images, drifted_labels, drift_meta = self.drift_generator.simulate_label_shift(
                images, labels, **scenario.parameters
            )
        
        else:
            raise ValueError(f"Unknown drift type: {scenario.drift_type}")
        
        # Add scenario info to metadata
        metadata = {
            'scenario_name': scenario.name,
            'scenario_description': scenario.description,
            'severity': scenario.severity,
            'category': scenario.category,
            'expected_detection_methods': scenario.expected_detection_methods,
            'drift_metadata': drift_meta
        }
        
        self.scenario_results.append(metadata)
        
        return drifted_images, drifted_labels, metadata
    
    def run_multiple_scenarios(
        self,
        scenario_names: list,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Run multiple scenarios on the same data.
        
        Args:
            scenario_names: List of scenario names to run
            images: Original images
            labels: Original labels
            
        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        
        for scenario_name in scenario_names:
            scenario = ScenarioLibrary.get_scenario(scenario_name)
            if scenario is None:
                print(f"Warning: Scenario '{scenario_name}' not found")
                continue
            
            print(f"Running scenario: {scenario.name}...")
            drifted_images, drifted_labels, metadata = self.run_scenario(
                scenario, images.copy(), labels.copy()
            )
            
            results[scenario_name] = (drifted_images, drifted_labels, metadata)
        
        return results
    
    def get_scenario_results(self) -> list:
        """Get history of scenario executions."""
        return self.scenario_results
    
    def reset_results(self):
        """Clear scenario results."""
        self.scenario_results = []


def print_scenario_catalog():
    """Print formatted catalog of all scenarios."""
    scenarios = ScenarioLibrary.get_all_scenarios()
    
    print("=" * 80)
    print("DRIFT SCENARIO CATALOG")
    print("=" * 80)
    
    by_category = {
        'data_drift': [],
        'concept_drift': [],
        'combined': []
    }
    
    for name, scenario in scenarios.items():
        by_category[scenario.category].append((name, scenario))
    
    for category in ['data_drift', 'concept_drift', 'combined']:
        print(f"\n{category.replace('_', ' ').upper()}:")
        print("-" * 80)
        
        for name, scenario in by_category[category]:
            severity_emoji = {
                'mild': '🟢',
                'moderate': '🟡',
                'severe': '🔴'
            }[scenario.severity]
            
            print(f"\n{severity_emoji} {scenario.name} [{scenario.severity}]")
            print(f"   {scenario.description}")
            print(f"   Type: {scenario.drift_type}")
            print(f"   Detection methods: {', '.join(scenario.expected_detection_methods)}")
    
    print("\n" + "=" * 80)
    print(f"Total scenarios: {len(scenarios)}")


if __name__ == '__main__':
    # Example usage
    print("Testing ScenarioLibrary...")
    
    # Print catalog
    print_scenario_catalog()
    
    # Test scenario execution
    print("\n\nTesting scenario execution...")
    
    # Create dummy data
    np.random.seed(42)
    images = np.random.randint(0, 255, size=(100, 28, 28, 3), dtype=np.uint8)
    labels = np.random.randint(0, 3, size=(100, 1))
    
    runner = ScenarioRunner(seed=42)
    
    # Run a few scenarios
    test_scenarios = ['equipment_aging', 'scanner_replacement', 'seasonal_patient_flow']
    
    results = runner.run_multiple_scenarios(test_scenarios, images, labels)
    
    print(f"\n✓ Successfully ran {len(results)} scenarios")
    
    for scenario_name, (drifted_img, drifted_lbl, metadata) in results.items():
        print(f"  - {scenario_name}: {drifted_img.shape}, severity={metadata['severity']}")
    
    print("\n✓ All tests passed!")
