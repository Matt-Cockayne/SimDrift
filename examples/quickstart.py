# Example: Quick Start with SimDrift

from data.medmnist_loader import MedMNISTLoader
from data.drift_generators import DriftGenerator
from monitoring.drift_detectors import DriftDetector
from monitoring.performance_trackers import PerformanceTracker
from monitoring.alerting import AlertSystem
from simulations.scenarios import ScenarioLibrary, ScenarioRunner

# 1. Load Medical Imaging Dataset
print("Loading PathMNIST dataset...")
loader = MedMNISTLoader('pathmnist')
test_images, test_labels = loader.get_numpy_data('test')
print(f"Loaded {len(test_images)} test images")

# 2. Run a Drift Scenario
print("\nRunning 'Equipment Aging' scenario...")
runner = ScenarioRunner(seed=42)
scenario = ScenarioLibrary.get_scenario('equipment_aging')
drifted_images, drifted_labels, metadata = runner.run_scenario(
    scenario, test_images[:500], test_labels[:500]
)
print(f"Scenario: {metadata['scenario_name']}")
print(f"Severity: {metadata['severity']}")

# 3. Detect Drift
print("\nDetecting drift...")
detector = DriftDetector()

# Flatten images for detection
ref_flat = test_images[:250].reshape(250, -1).astype(float) / 255.0
drift_flat = drifted_images[:250].reshape(250, -1).astype(float) / 255.0

# Sample features for efficiency
import numpy as np
n_features = 100
feature_indices = np.random.choice(ref_flat.shape[1], n_features, replace=False)

results = detector.detect_drift(
    ref_flat[:, feature_indices],
    drift_flat[:, feature_indices],
    method='all'
)

print(detector.get_drift_summary(results))

# 4. Generate Alerts
print("\nGenerating alerts...")
alert_system = AlertSystem()
alerts = alert_system.check_drift_alert(results)

for alert in alerts:
    print(f"\n⚠️  {alert['message']}")
    print("Recommendations:")
    for rec in alert['recommendations'][:3]:
        print(f"  {rec}")

print("\n✅ Quick start complete!")
print("Next steps:")
print("  - Run the dashboard: streamlit run dashboard/app.py")
print("  - Explore notebooks/ for detailed tutorials")
print("  - Try different scenarios from ScenarioLibrary")
