#!/bin/bash

# SimDrift Environment Setup Script
# Creates a conda environment with all required dependencies

set -e  # Exit on error

echo "========================================="
echo "SimDrift Environment Setup"
echo "========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="SimDrift"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Warning: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborting setup."
        exit 0
    fi
fi

echo ""
echo "Creating conda environment: ${ENV_NAME}"
echo "This may take several minutes..."
echo ""

# Create conda environment with Python 3.10
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "Activating environment..."

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo ""
echo "Installing core dependencies..."

# Install PyTorch GPU version
# For GPU support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "Installing additional packages from requirements.txt..."

# Disable user site-packages to ensure clean environment
export PYTHONNOUSERSITE=1

# Install remaining packages via pip (with --no-user flag)
pip install --upgrade pip

# Core scientific packages (prefer conda when possible)
conda install numpy scipy scikit-learn pandas -y

# Fallback to pip if conda doesn't have the right version
pip install --no-user "numpy>=1.21.0"
pip install --no-user "scipy>=1.7.0"
pip install --no-user "scikit-learn>=1.0.0"
pip install --no-user "pandas>=1.3.0"

# Medical imaging dataset
pip install --no-user "medmnist>=2.2.0"

# Visualization (use conda for matplotlib)
conda install matplotlib seaborn -y
pip install --no-user "plotly>=5.0.0"

# Dashboard
pip install --no-user "streamlit>=1.25.0"

# Utilities
pip install --no-user "tqdm>=4.62.0"

# Development tools (optional)
read -p "Install development tools (pytest, jupyter, etc.)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development tools..."
    conda install pytest jupyter ipykernel -y
    pip install --no-user "pytest-cov>=3.0.0"
    pip install --no-user "black>=22.0.0"
    pip install --no-user "flake8>=4.0.0"
    
    # Register kernel for Jupyter
    python -m ipykernel install --user --name=${ENV_NAME} --display-name "Python (${ENV_NAME})"
fi

echo ""
echo "Configuring environment to ignore user site-packages..."

# Create activation script to set PYTHONNOUSERSITE
mkdir -p ~/miniconda3/envs/${ENV_NAME}/etc/conda/activate.d
cat > ~/miniconda3/envs/${ENV_NAME}/etc/conda/activate.d/env_vars.sh << 'ENVEOF'
#!/bin/bash
export PYTHONNOUSERSITE=1
ENVEOF
chmod +x ~/miniconda3/envs/${ENV_NAME}/etc/conda/activate.d/env_vars.sh

echo "✓ Environment configured to use conda packages only"
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To launch the dashboard, run:"
echo "  streamlit run dashboard/app.py"
echo ""
echo "To test the installation, run:"
echo "  python examples/quickstart.py"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "========================================="
