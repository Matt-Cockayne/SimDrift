# SimDrift 📈

**Simulation-Based Model Drift Detection Dashboard**

SimDrift is a stunning, interactive platform for visualizing and understanding ML model drift in medical imaging. It features **24+ pre-trained models** across **8 medical datasets**, **real-time drift simulation** with **15+ degradation types**, and a **visual dashboard** that makes complex MLOps concepts instantly understandable. Perfect for education, research, and demonstrating production ML monitoring without needing actual production access.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

SimDrift demonstrates production ML monitoring challenges without requiring actual production systems. It simulates realistic drift scenarios on medical imaging datasets, detecting distribution shifts using statistical methods and tracking model performance degradation.

**Key Features:**
- **15+ Drift Scenarios**: Equipment aging, motion blur, compression artifacts, demographic shifts, protocol changes
- **5 Detection Methods**: PSI, KS test, Chi-square, MMD, Wasserstein distance  
- **Interactive Dashboard**: Real-time visualization with side-by-side image comparison and drift metrics
- **Model Training**: Support for SimpleCNN, ResNet18/34, EfficientNet, ViT on 8 medical datasets
- **Smart Alerts**: Severity-based alerting with actionable recommendations
- **Educational**: Designed for teaching MLOps concepts through hands-on simulation

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Matt-Cockayne/SimDrift.git
cd SimDrift

# Run setup script (creates conda environment)
./setup.sh

# Activate the environment
conda activate SimDrift
```

**Alternative: Manual Installation**

```bash
# Create conda environment
conda create -n SimDrift python=3.10 -y
conda activate SimDrift

# Install PyTorch (CPU version)
conda install pytorch torchvision cpuonly -c pytorch -y

# Install dependencies
pip install -r requirements.txt
```

### Train Models (Optional)

SimDrift works out-of-the-box with simulated models, but for best results, train real models:

```bash
# Quick test (1 model, 3 epochs)
python train_all_models.py --quick-test

# Train all models (recommended for full experience)
python train_all_models.py

# Train specific dataset
python train_all_models.py --datasets pathmnist dermamnist

# Train specific architecture
python train_all_models.py --architectures simple_cnn resnet18
```

Training creates checkpoints in `model_zoo/{dataset}/{architecture}_best.pth` with metadata and visualizations.

### Launch Dashboard

```bash
streamlit run dashboard/Home.py --server.port 8503
```

The dashboard opens at `http://localhost:8503` with navigation to:
- **Home**: Main drift simulation interface  
- **Model Zoo**: Compare trained models
- **Drift Lab**: Create custom scenarios
- **Analytics**: Deep metrics analysis
- **Tutorial**: Interactive learning modules
- **Settings**: Configure detection and alerts

---

## 📊 Available Datasets

| Dataset | Description | Classes | Image Size | Samples |
|---------|-------------|---------|------------|---------|
| **PathMNIST** | Colon pathology | 9 | 28×28 RGB | 107,180 |
| **DermaMNIST** | Dermatology | 7 | 28×28 RGB | 10,015 |
| **RetinaMNIST** | Retina OCT | 5 | 28×28 RGB | 1,600 |
| **BloodMNIST** | Blood cell | 8 | 28×28 RGB | 17,092 |
| **PneumoniaMNIST** | Chest X-ray | 2 | 28×28 Gray | 5,856 |
| **BreastMNIST** | Breast ultrasound | 2 | 28×28 Gray | 780 |
| **OCTMNIST** | Retinal OCT | 4 | 28×28 Gray | 109,309 |
| **TissueMNIST** | Kidney tissue | 8 | 28×28 Gray | 236,386 |

---

## Drift Types

**Visual Drift:**
- Brightness, Contrast, Blur, Motion Blur, JPEG Compression, Noise
- Occlusion, Zoom, Vignette, Color Temperature, Saturation

**Concept Drift:**
- Demographic shifts, Prevalence changes, Label shift

**Combined Scenarios:**
- Equipment aging, Scanner replacement, Hospital transfer, Protocol changes

---

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **PSI** | Univariate drift | Fast, interpretable | Requires binning |
| **KS Test** | Distribution comparison | Statistical rigor | Univariate only |
| **Chi-Square** | Categorical features | Handles discrete data | Requires sufficient samples |
| **MMD** | Multivariate drift | Captures complex patterns | Computationally intensive |
| **Wasserstein** | Distribution distance | Geometric interpretation | Slower for high dimensions |

## 📈 Performance Metrics

- **Classification**: Accuracy, precision, recall, F1 (macro/weighted)
- **Calibration**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
- **Fairness**: Demographic parity, equalized odds (by sensitive attributes)
- **Confidence**: Prediction distribution statistics
## Detection Methods

| Method | Pros | Cons |
|--------|------|------|
| **PSI** | Fast, interpretable | Univariate, requires binning |
| **KS Test** | Statistical rigor | Univariate only |
| **Chi-Square** | Handles discrete data | Needs sufficient samples |
| **MMD** | Captures complex patterns | Computationally intensive |
| **Wasserstein** | Geometric interpretation | Slower for high dimensions |

## Usage

### Python API

```python
from data.medmnist_loader import MedMNISTLoader
from data.drift_generators import DriftGenerator
## Project Structure

```
SimDrift/
├── data/                    # Data layer
│   ├── medmnist_loader.py  # Dataset loader
│   └── drift_generators.py # Drift simulation
├── models/                  # Model layer
│   ├── model_manager.py    # Model zoo management
│   └── training/           # Training infrastructure
├── monitoring/              # Monitoring layer
│   ├── drift_detectors.py  # Detection methods
│   ├── performance_trackers.py
│   └── alerting.py         # Alert system
├── simulations/            # Scenario library
├── dashboard/              # Interactive UI
│   ├── Home.py            # Main dashboard
│   └── pages/             # Multi-page app
├── train_all_models.py    # Training script
└── requirements.txt
```

## Contributing

Areas for enhancement:
- Additional drift types (adversarial, feedback loops)
- More detection methods (ADWIN, DDM, KSWIN)
- Enhanced visualizations
- Jupyter tutorials
- REST API for monitoring
   - 🚨 URGENT: Consider model retraining immediately
   - ⚠️ Evaluate current model predictions for reliability
   - 🔄 Implement temporary fallback mechanisms
```

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Additional drift scenarios
- New detection methods
- More medical imaging datasets
- Enhanced visualizations
- Performance optimizations

## 📝 Citation

If you use SimDrift in your research or teaching, please cite:

```bibtex
@software{simdrift2024,
  title={SimDrift: Simulation-Based Model Drift Detection},
  author={Cockayne, Matthew J.},
  year={2024},
  url={https://github.com/Matt-Cockayne/SimDrift}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **MedMNIST**: Jiancheng Yang et al. for the excellent medical imaging benchmark
- **Medical AI Community**: For highlighting the importance of monitoring in healthcare ML
- **MLOps Best Practices**: Inspired by production ML monitoring systems

## 📧 Contact

- **Author**: Matthew J. Cockayne
- **Email**: m.j.cockayne@keele.ac.uk
- **GitHub**: https://github.com/Matt-Cockayne

---

**⭐ Star this repo if you find it useful for learning about ML monitoring!**
