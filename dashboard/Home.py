"""
SimDrift - Enhanced Interactive Visual Dashboard

A dashboard for ML drift demonstration.
Features real-time visualization, model comparison, and comprehensive monitoring.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os
from pathlib import Path
import torch
from PIL import Image
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.medmnist_loader import MedMNISTLoader
from data.drift_generators import DriftGenerator
from monitoring.drift_detectors import DriftDetector
from monitoring.performance_trackers import PerformanceTracker
from monitoring.alerting import AlertSystem
from simulations.scenarios import ScenarioLibrary
from models.model_manager import ModelManager

# Page configuration
st.set_page_config(
    page_title="SimDrift - Home",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme (Modern dark theme)
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Purple
    'success': '#10b981',      # Green
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'background': '#0f172a',   # Slate 900
    'surface': '#1e293b',      # Slate 800
    'text': '#f1f5f9'          # Slate 100
}

# Custom CSS with modern design
st.markdown(f"""
<style>
    /* Global styles */
    .main {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
    }}
    
    /* Hero header */
    .hero-header {{
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, 
            {COLORS['primary']} 0%, 
            {COLORS['secondary']} 50%, 
            #ec4899 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border-radius: 1rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
    }}
    
    .hero-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }}
    
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    .hero-title {{
        font-size: 3.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #ffffff 0%, #f0f9ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        color: rgba(255,255,255,0.95);
        position: relative;
        z-index: 1;
        margin-top: 0.5rem;
    }}
    
    /* Glass morphism cards */
    .glass-card {{
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}
    
    /* Metric cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['surface']}, {COLORS['background']});
        border-left: 4px solid {COLORS['primary']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {COLORS['primary']};
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {COLORS['text']};
        opacity: 0.8;
    }}
    
    /* Alert cards */
    .alert-critical {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border-left: 4px solid {COLORS['danger']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }}
    
    .alert-warning {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1));
        border-left: 4px solid {COLORS['warning']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    
    .alert-success {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border-left: 4px solid {COLORS['success']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    /* Image grid */
    .image-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.5rem;
        padding: 1rem;
    }}
    
    .image-cell {{
        border-radius: 0.5rem;
        overflow: hidden;
        transition: transform 0.2s;
    }}
    
    .image-cell:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.4);
    }}
    
    /* Progress bar */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    }}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    defaults = {
        'data_loaded': False,
        'dataset_name': 'pathmnist',
        'drift_severity': 0.5,
        'current_drift_type': 'brightness',
        'model_manager': None,
        'drift_generator': None,
        'drift_detector': None,
        'alert_system': None,
        'original_images': None,
        'original_labels': None,
        'drifted_images': None,
        'simulation_running': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Lazy-load heavy objects
    if st.session_state.drift_generator is None:
        st.session_state.drift_generator = DriftGenerator(seed=42)
    if st.session_state.drift_detector is None:
        st.session_state.drift_detector = DriftDetector()
    if st.session_state.alert_system is None:
        st.session_state.alert_system = AlertSystem()


@st.cache_resource
def load_dataset(dataset_name: str):
    """Load and cache dataset."""
    loader = MedMNISTLoader(dataset_name=dataset_name)
    return loader


@st.cache_resource
def get_model_manager():
    """Get cached model manager."""
    return ModelManager('model_zoo')


def evaluate_model_on_data(model, images, labels, device='cpu'):
    """
    Evaluate model on given images and calculate metrics.
    
    Args:
        model: PyTorch model
        images: Images to evaluate (N, H, W, C)
        labels: Ground truth labels (N,)
        device: Device to run on
        
    Returns:
        accuracy: Accuracy score
        ece: Expected Calibration Error
        predictions: Model predictions
        probabilities: Model probabilities
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    model.eval()
    
    # Prepare data
    # Convert to torch tensors and normalize to [0, 1]
    images_tensor = torch.from_numpy(images).float() / 255.0
    
    # Apply same normalization as training: (x - 0.5) / 0.5 = x * 2 - 1
    # This transforms [0, 1] to [-1, 1]
    images_tensor = (images_tensor - 0.5) / 0.5
    
    # Flatten labels if they have shape (N, 1) instead of (N,)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    
    labels_tensor = torch.from_numpy(labels).long()
    
    # Handle different input shapes
    # MedMNIST format is typically (N, H, W, C) where C=1 or C=3
    if images_tensor.ndim == 4:
        # If last dimension is channels, transpose to (N, C, H, W)
        if images_tensor.shape[-1] in [1, 3]:
            images_tensor = images_tensor.permute(0, 3, 1, 2)
    elif images_tensor.ndim == 3:
        # Grayscale without channel dim (N, H, W) -> (N, 1, H, W)
        images_tensor = images_tensor.unsqueeze(1)
    
    dataset = TensorDataset(images_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # Ensure correct shape: should be (B, C, H, W)
            # If shape is wrong, try to fix it
            if batch_images.ndim != 4:
                raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {batch_images.shape}")
            
            # Check if channels are in the right place
            # Typical image sizes are 28x28, 32x32, 64x64, etc. Channels are 1 or 3
            if batch_images.shape[1] not in [1, 3] and batch_images.shape[-1] in [1, 3]:
                # Channels are in wrong position, transpose
                batch_images = batch_images.permute(0, 3, 1, 2)
            
            outputs = model(batch_images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels_np)
    
    # Calculate ECE (Expected Calibration Error)
    ece = calculate_ece(probabilities, labels_np, n_bins=15)
    
    return accuracy, ece, predictions, probabilities


def calculate_ece(probs, labels, n_bins=15):
    """Calculate Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (bin_size / len(labels)) * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def display_hero():
    """Display hero section."""
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">SimDrift</h1>
        <p class="hero-subtitle">Interactive Medical AI Drift Simulation & Monitoring</p>
    </div>
    """, unsafe_allow_html=True)


def display_landing_page():
    """Display professional landing page before simulation."""
    
    # Main title and description
    st.markdown("## Welcome to SimDrift")
    st.markdown("""
    SimDrift is an interactive platform for demonstrating and analyzing data drift in medical AI systems.
    Simulate real-world drift scenarios, evaluate model robustness, and explore detection methods.
    """)
    
    st.markdown("---")
    
    # Feature overview with interactive cards
    st.markdown("### ▶️ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>⚡ Realistic Drift Simulation</h3>
            <p>Simulate equipment aging, scanner changes, and environmental variations with scientifically-grounded transformations.</p>
            <ul>
                <li>Brightness & Contrast shifts</li>
                <li>Blur & Noise injection</li>
                <li>Motion artifacts</li>
                <li>Compression effects</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>📈 Progressive Analysis</h3>
            <p>Watch model performance degrade in real-time as drift severity increases from 0% to 100%.</p>
            <ul>
                <li>Animated drift progression</li>
                <li>Performance degradation curves</li>
                <li>Before/after comparisons</li>
                <li>Statistical metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3>📁 Medical Datasets</h3>
            <p>Explore drift across 6 real medical imaging datasets from MedMNIST.</p>
            <ul>
                <li>PathMNIST (Pathology)</li>
                <li>DermaMNIST (Dermatology)</li>
                <li>RetinaMNIST (Retinal OCT)</li>
                <li>BloodMNIST (Blood Cells)</li>
                <li>TissueMNIST (Tissue)</li>
                <li>PneumoniaMNIST (Pneumonia)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive dataset preview
    st.markdown("### 🔍 Dataset Preview")
    st.markdown("Explore sample images from each medical imaging dataset")
    
    # Dataset selector for preview
    preview_dataset = st.selectbox(
        "Select Dataset to Preview",
        ['pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist', 'tissuemnist', 'pneumoniamnist'],
        format_func=lambda x: {
            'pathmnist': '🧬 PathMNIST - Colon Pathology (9 classes)',
            'dermamnist': '🩺 DermaMNIST - Skin Lesions (7 classes)',
            'retinamnist': '👁️ RetinaMNIST - Retinal OCT (5 classes)',
            'bloodmnist': '🩸 BloodMNIST - Blood Cells (8 classes)',
            'tissuemnist': '🧬 TissueMNIST - Kidney Tissue (8 classes)',
            'pneumoniamnist': '🫁 PneumoniaMNIST - Pneumonia (2 classes)'
        }[x],
        key="landing_dataset_preview"
    )
    
    # Load and display preview
    try:
        with st.spinner(f"Loading {preview_dataset} preview..."):
            loader = load_dataset(preview_dataset)
            # Only load the 8 images we need for display - much faster!
            test_images, test_labels = loader.get_numpy_data('test', limit=8, offset=0)
            
            # Get metadata for metrics without loading all data
            import medmnist
            info = medmnist.INFO[preview_dataset]
            
            # Display info
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Test Samples", f"{info['n_samples']['test']:,}")
            with col_info2:
                st.metric("Image Size", f"{test_images.shape[1]}×{test_images.shape[2]}")
            with col_info3:
                # Handle datasets with or without explicit channel dimension
                channels = test_images.shape[3] if test_images.ndim == 4 else 1
                st.metric("Channels", channels)
            with col_info4:
                st.metric("Classes", len(info['label']))  # Use metadata instead of loading all data
            
            # Display sample images
            st.markdown("#### Sample Images")
            cols = st.columns(8)
            
            for idx, col in enumerate(cols):
                with col:
                    img = test_images[idx]
                    label = test_labels[idx].item() if test_labels.ndim > 1 else test_labels[idx]
                    st.image(img, caption=f"Class {label}", use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dataset preview: {str(e)}")
    
    st.markdown("---")
    
    # Model architecture info
    st.markdown("### 📊 Available Models")
    st.markdown("Three neural network architectures trained on each dataset:")
    
    col_arch1, col_arch2, col_arch3 = st.columns(3)
    
    with col_arch1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">SimpleCNN</div>
            <div class="metric-value">~800K</div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.8;">
                Lightweight baseline model<br>
                Fast inference, good for testing
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ResNet-18</div>
            <div class="metric-value">~11M</div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.8;">
                Standard CNN architecture<br>
                Balanced performance and speed
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">EfficientNet-B0</div>
            <div class="metric-value">~4M</div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.8;">
                State-of-the-art efficiency<br>
                Best accuracy-to-size ratio
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    <div class="glass-card">
        <h4>Ready to simulate drift?</h4>
        <ol style="line-height: 2;">
            <li><strong>Configure</strong> → Select dataset and model in the sidebar</li>
            <li><strong>Choose Drift</strong> → Pick a drift type (brightness, blur, noise, etc.)</li>
            <li><strong>Set Severity</strong> → Adjust the drift intensity slider (0-100%)</li>
            <li><strong>Run Simulation</strong> → Click the button to start analysis</li>
        </ol>
        <p style="margin-top: 1rem; padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 0.5rem;">
            💡 <strong>Tip:</strong> Start with 50% severity on 'brightness' drift to see clear effects without overwhelming the model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technical details (collapsible)
    with st.expander("📖 Technical Details & Methodology"):
        st.markdown("""
        #### Drift Detection Methods
        
        SimDrift implements multiple statistical drift detection methods:
        
        - **PSI (Population Stability Index)**: Measures distribution shift between reference and current data
        - **Kolmogorov-Smirnov Test**: Non-parametric test for distribution differences
        - **Chi-Squared Test**: Goodness-of-fit test for categorical distributions
        - **Maximum Mean Discrepancy (MMD)**: Kernel-based distance between distributions
        - **Wasserstein Distance**: Optimal transport distance between distributions
        
        #### Drift Simulation
        
        Each drift type is calibrated to simulate real-world scenarios:
        
        - **Brightness**: Linear intensity shift simulating sensor aging (0-50 units)
        - **Contrast**: Reduction factor simulating calibration drift (0.5-1.0x)
        - **Blur**: Gaussian kernel simulating lens degradation (σ = 0-3.0)
        - **Noise**: Additive Gaussian noise simulating electronic interference (σ = 0-25)
        - **Motion Blur**: Directional blur simulating patient movement
        - **JPEG Compression**: Quality reduction simulating storage/transmission artifacts
        
        #### Model Training
        
        All models trained with:
        - **Optimizer**: Adam with weight decay
        - **Learning Rate**: 1e-4 to 1e-3 (architecture dependent)
        - **Normalization**: mean=0.5, std=0.5 ([-1, 1] range)
        - **Data**: MedMNIST v2.2.3 official splits
        - **Early Stopping**: Patience of 10 epochs on validation accuracy
        """)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>Ready to explore drift?</h3>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Configure your simulation in the sidebar and click <strong>"Run Simulation"</strong> to begin
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer with attribution
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: rgba(30, 41, 59, 0.5); border-radius: 0.5rem; margin-top: 2rem;">
        <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">
            Developed by <strong>Matthew Cockayne</strong> | PhD Researcher, Keele University
        </p>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.85rem;">
            <a href="https://matt-cockayne.github.io" target="_blank" style="color: {COLORS['primary']}; text-decoration: none; margin: 0 0.5rem;">Portfolio</a> |
            <a href="https://github.com/Matt-Cockayne" target="_blank" style="color: {COLORS['primary']}; text-decoration: none; margin: 0 0.5rem;">GitHub</a> |
            <a href="https://www.linkedin.com/in/matthew-cockayne-193659199" target="_blank" style="color: {COLORS['primary']}; text-decoration: none; margin: 0 0.5rem;">LinkedIn</a> |
            <a href="mailto:matthewcockayne2@gmail.com" style="color: {COLORS['primary']}; text-decoration: none; margin: 0 0.5rem;">Contact</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_metrics(original_acc: float, drifted_acc: float, drift_score: float, ece: float):
    """Display metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_acc = drifted_acc - original_acc
        st.metric(
            label="Accuracy",
            value=f"{drifted_acc:.2%}",
            delta=f"{delta_acc:+.2%}",
            delta_color="inverse" if delta_acc < 0 else "normal"
        )
    
    with col2:
        st.metric(
            label="Drift Score (MMD)",
            value=f"{drift_score:.4f}",
            delta="Drift Detected" if drift_score > 0.1 else "Normal",
            delta_color="inverse" if drift_score > 0.1 else "off"
        )
    
    with col3:
        st.metric(
            label="Calibration (ECE)",
            value=f"{ece:.4f}",
            delta="Poor" if ece > 0.1 else "Good",
            delta_color="inverse" if ece > 0.1 else "off"
        )
    
    with col4:
        degradation = abs(delta_acc) / original_acc * 100 if original_acc > 0 else 0
        st.metric(
            label="Degradation",
            value=f"{degradation:.1f}%",
            delta="Critical" if degradation > 20 else "Acceptable",
            delta_color="inverse" if degradation > 20 else "off"
        )


def display_image_grid(images: np.ndarray, labels: np.ndarray, 
                       predictions: np.ndarray = None,
                       title: str = "", n_display: int = 4):
    """Display image grid with predictions."""
    st.markdown(f"### {title}")
    
    # Calculate grid dimensions (2x2 grid)
    n_cols = 2
    n_rows = 2
    
    # Create grid
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            img_idx = row * n_cols + col_idx
            if img_idx >= min(n_display, len(images)):
                break
            
            with cols[col_idx]:
                # Display image
                img = images[img_idx]
                
                # Convert to PIL Image and resize for better visibility
                if img.shape[-1] == 1:
                    img_display = Image.fromarray(img.squeeze(), mode='L')
                else:
                    img_display = Image.fromarray(img.astype(np.uint8))
                
                # Upscale image for better visibility (3x)
                new_size = (img_display.width * 3, img_display.height * 3)
                img_display = img_display.resize(new_size, Image.NEAREST)
                
                st.image(img_display, use_container_width=True)
                
                # Display label and prediction
                if predictions is not None:
                    true_label = labels[img_idx]
                    pred_label = predictions[img_idx]
                    is_correct = true_label == pred_label
                    
                    color = COLORS['success'] if is_correct else COLORS['danger']
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.9rem;'>"
                        f"<span style='color:{color};'>{'✓' if is_correct else '✗'}</span> "
                        f"True: {true_label} | Pred: {pred_label}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.9rem;'>"
                        f"Label: {labels[img_idx]}"
                        f"</div>",
                        unsafe_allow_html=True
                    )


def plot_drift_progression(severities: list, accuracies: list, drift_scores: list):
    """Plot drift progression over time."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Performance Degradation', 'Drift Detection Score'),
        vertical_spacing=0.15
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=severities,
            y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor=f"rgba(99, 102, 241, 0.2)"
        ),
        row=1, col=1
    )
    
    # Drift score plot
    fig.add_trace(
        go.Scatter(
            x=severities,
            y=drift_scores,
            mode='lines+markers',
            name='Drift Score',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_hline(
        y=0.1, line_dash="dash", line_color=COLORS['warning'],
        annotation_text="Alert Threshold",
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Drift Severity", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="MMD Score", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        hovermode='x unified'
    )
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """Plot confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    )
    
    return fig


def plot_drift_heatmap(drift_results: dict):
    """Plot drift detection metrics with appropriate scales for each method."""
    # Handle nested structure from drift detector
    if 'methods' in drift_results:
        methods_data = drift_results['methods']
    else:
        methods_data = drift_results
    
    methods = list(methods_data.keys())
    scores = []
    drift_detected_flags = []
    
    # Extract scores and detection flags
    for m in methods:
        method_result = methods_data[m]
        if isinstance(method_result, dict):
            drift_detected_flags.append(method_result.get('drift_detected', False))
            # Try to get score - use 90th percentile for per-feature metrics
            if 'score' in method_result:
                scores.append(method_result['score'])
            elif 'scores' in method_result:
                # PSI per feature - use 90th percentile to reduce noise
                scores.append(np.percentile(method_result['scores'], 90))
            elif 'statistics' in method_result:
                # KS/Chi2 per feature - use 90th percentile
                scores.append(np.percentile(method_result['statistics'], 90))
            elif 'distance' in method_result:
                scores.append(method_result['distance'])
            elif 'distances' in method_result:
                scores.append(np.mean(method_result['distances']))
            else:
                scores.append(0)
        else:
            scores.append(0)
            drift_detected_flags.append(False)
    
    # Define appropriate thresholds and display ranges for each metric
    thresholds = {
        'psi': {'warning': 0.1, 'critical': 0.25, 'max_auto': True},
        'ks': {'warning': 0.1, 'critical': 0.3, 'max_auto': True},
        'chi2': {'warning': 50, 'critical': 100, 'max_auto': True},
        'mmd': {'warning': 0.05, 'critical': 0.1, 'max_auto': True},
        'wasserstein': {'warning': 0.05, 'critical': 0.1, 'max_auto': True}
    }
    
    # Create subplots - one for each metric
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=len(methods),
        subplot_titles=[m.upper() for m in methods],
        horizontal_spacing=0.12
    )
    
    for idx, (method, score, detected) in enumerate(zip(methods, scores, drift_detected_flags), 1):
        threshold_info = thresholds.get(method, {'warning': 0.1, 'critical': 0.2, 'max_auto': True})
        
        # Determine color based on detection
        if detected:
            color = '#ef4444'  # Bright red
        elif score > threshold_info['warning']:
            color = '#f59e0b'  # Bright orange
        else:
            color = '#10b981'  # Bright green
        
        # Auto-scale y-axis based on actual score
        max_y = max(score * 1.3, threshold_info['critical'] * 2)
        
        # Add bar for this metric
        fig.add_trace(
            go.Bar(
                x=[method],
                y=[score],
                marker_color=color,
                marker_line_color='rgba(255,255,255,0.3)',
                marker_line_width=1,
                text=[f"{score:.3f}"],
                textposition='outside',
                textfont=dict(size=12, color='white'),
                showlegend=False,
                hovertemplate=f"<b>{method.upper()}</b><br>Score: {score:.4f}<br>Status: {'DRIFT' if detected else 'NORMAL'}<extra></extra>"
            ),
            row=1, col=idx
        )
        
        # Add reference lines for this metric's thresholds
        fig.add_hline(
            y=threshold_info['critical'],
            line=dict(dash="dash", color='rgba(239, 68, 68, 0.6)', width=2),
            row=1, col=idx,
            annotation=dict(
                text="Critical",
                font=dict(size=9, color='rgba(239, 68, 68, 0.8)'),
                showarrow=False,
                xanchor='left',
                x=0.02
            )
        )
        
        fig.add_hline(
            y=threshold_info['warning'],
            line=dict(dash="dot", color='rgba(245, 158, 11, 0.6)', width=1.5),
            row=1, col=idx,
            annotation=dict(
                text="Warning",
                font=dict(size=9, color='rgba(245, 158, 11, 0.8)'),
                showarrow=False,
                xanchor='left',
                x=0.02
            )
        )
        
        # Update y-axis for this subplot
        fig.update_yaxes(
            range=[0, max_y],
            title_text="Score" if idx == 1 else "",
            gridcolor='rgba(255,255,255,0.1)',
            row=1, col=idx
        )
        
        # Update x-axis
        fig.update_xaxes(
            showticklabels=False,
            row=1, col=idx
        )
    
    fig.update_layout(
        title=dict(
            text="Drift Detection Results",
            font=dict(size=18, color='white')
        ),
        height=450,
        plot_bgcolor='rgba(30, 30, 46, 0.4)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(t=80, b=40, l=60, r=40)
    )
    
    return fig


def main():
    """Main dashboard application."""
    init_session_state()
    
    # Hero section
    display_hero()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## Configuration")
        st.markdown("Configure drift simulation parameters")
        st.markdown("---")
        
        # Dataset selection
        dataset_name = st.selectbox(
            "Dataset",
            ['pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist', 'tissuemnist', 'pneumoniamnist'],
            help="Select medical imaging dataset"
        )
        
        # Architecture selection  
        st.markdown("### Model")
        model_manager = get_model_manager()
        available_archs = model_manager.get_available_architectures(dataset_name)
        
        if available_archs:
            architecture = st.selectbox("Architecture", available_archs)
        else:
            st.warning("⚠️ No pre-trained models found. Run training first.")
            architecture = 'simple_cnn'
        
        # Drift type selection
        st.markdown("### Drift Simulation")
        generator = DriftGenerator()
        drift_types = generator.get_available_drift_types()
        
        drift_type = st.selectbox(
            "Drift Type",
            list(drift_types.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()}"
        )
        
        st.info(f"ℹ️ {drift_types[drift_type]}")
        
        # Drift severity slider
        drift_severity = st.slider(
            "Drift Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Drag to adjust drift severity in real-time"
        )
        
        # Run button
        run_simulation = st.button("Run Simulation", use_container_width=True)
    
    # Main content area
    if run_simulation or st.session_state.data_loaded:
        # Store sidebar selections in session state
        st.session_state.drift_severity = drift_severity
        st.session_state.current_drift_type = drift_type
        
        # Load data
        if not st.session_state.data_loaded or st.session_state.dataset_name != dataset_name:
            with st.spinner(f"Loading {dataset_name}..."):
                loader = load_dataset(dataset_name)
                test_images, test_labels = loader.get_numpy_data('test')
                
                st.session_state.original_images = test_images
                st.session_state.original_labels = test_labels
                st.session_state.dataset_name = dataset_name
                st.session_state.data_loaded = True
        
        # Load and evaluate model first
        model_manager = get_model_manager()
        available_archs = model_manager.get_available_architectures(dataset_name)
        
        if architecture not in available_archs:
            st.error(f"❌ Model not found: {architecture} on {dataset_name}. Please train the model first.")
            st.stop()
        
        with st.spinner(f"Loading {architecture} model..."):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model_manager.load_pretrained(dataset_name, architecture, device=device)
            metadata_dict = model_manager.get_model_metadata(dataset_name, architecture)
        
        # Apply drift with multiple severity levels for progressive analysis
        with st.spinner("Simulating drift..."):
            original_images = st.session_state.original_images
            original_labels = st.session_state.original_labels
            
            # Generate drift at current severity
            drifted_images, _, metadata = st.session_state.drift_generator.simulate_gradual_drift(
                original_images,
                original_labels,
                drift_type=drift_type,
                n_steps=1,
                max_severity=drift_severity,
                return_steps=False
            )
            
            st.session_state.drifted_images = drifted_images
            
            # Evaluate model at each drift level (using full test set for accurate estimates)
            n_eval_samples = len(original_images)  # Use entire test set
            n_steps = 10
            severity_levels = np.linspace(0, 1.0, n_steps)
            accuracies = []
            
            # Also generate progressive images for animation (one sample at each severity)
            progressive_frames = []
            single_image = original_images[0:1]  # First image
            single_label = original_labels[0:1]
            
            for i, severity in enumerate(severity_levels):
                if i == 0:
                    # Original data for evaluation
                    acc, _, _, _ = evaluate_model_on_data(
                        model, original_images, original_labels, device
                    )
                    # Original image for animation
                    progressive_frames.append(single_image[0])
                else:
                    # Generate drifted data at this severity for evaluation
                    temp_drifted, _, _ = st.session_state.drift_generator.simulate_gradual_drift(
                        original_images,
                        original_labels,
                        drift_type=drift_type,
                        n_steps=1,
                        max_severity=severity,
                        return_steps=False
                    )
                    acc, _, _, _ = evaluate_model_on_data(
                        model, temp_drifted, original_labels, device
                    )
                    
                    # Generate single drifted image for animation
                    drifted_single, _, _ = st.session_state.drift_generator.simulate_gradual_drift(
                        single_image,
                        single_label,
                        drift_type=drift_type,
                        n_steps=1,
                        max_severity=severity,
                        return_steps=False
                    )
                    progressive_frames.append(drifted_single[0])
                    
                accuracies.append(acc)
        
        # Progressive Drift Analysis
        st.markdown("## 📉 Progressive Drift Impact")
        st.markdown("Watch model performance degrade as drift severity increases")
        st.markdown("Model performance degradation as drift severity increases")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Sample Image Drift Progression")
            st.markdown("*Single image transforming from original to maximum drift*")
            
            # Create animated GIF from progressive frames
            from PIL import Image as PILImage
            import io
            
            gif_frames = []
            
            for img in progressive_frames:
                # img is already a single image (H, W, C) or (H, W)
                img = np.squeeze(img)
                
                # Ensure uint8 format
                if img.max() <= 1.0 and img.min() >= 0.0:
                    img = (img * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                
                # Convert to PIL
                if img.ndim == 2:
                    pil_frame = PILImage.fromarray(img, mode='L')
                elif img.ndim == 3 and img.shape[2] == 1:
                    pil_frame = PILImage.fromarray(img[:, :, 0], mode='L')
                elif img.ndim == 3 and img.shape[2] == 3:
                    pil_frame = PILImage.fromarray(img, mode='RGB')
                else:
                    continue
                
                # Upscale for visibility
                new_size = (pil_frame.width * 6, pil_frame.height * 6)
                pil_frame = pil_frame.resize(new_size, PILImage.NEAREST)
                gif_frames.append(pil_frame)
            
            # Create and display GIF
            if len(gif_frames) > 0:
                gif_buffer = io.BytesIO()
                gif_frames[0].save(
                    gif_buffer,
                    format='GIF',
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=400,
                    loop=0,
                    optimize=False
                )
                gif_buffer.seek(0)
                st.image(gif_buffer, caption="Drift progression: 0% → 100% severity", use_container_width=True)
            else:
                st.error("No frames created")
        
        with col2:
            st.markdown("### Model Performance Degradation")
            st.markdown("*Accuracy on test set across drift severity levels*")
            
            # Create static performance degradation plot (simpler and more reliable)
            fig_perf = go.Figure()
            
            fig_perf.add_trace(go.Scatter(
                x=severity_levels * 100,  # Convert to percentage
                y=np.array(accuracies) * 100,
                mode='lines+markers',
                name='Test Accuracy',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#667eea'),
                hovertemplate='Severity: %{x:.0f}%<br>Accuracy: %{y:.2f}%<extra></extra>'
            ))
            
            # Add reference lines
            fig_perf.add_hline(
                y=accuracies[0] * 100,
                line_dash="dash",
                line_color="rgba(16, 185, 129, 0.5)",
                annotation_text=f"Baseline: {accuracies[0]*100:.1f}%",
                annotation_position="right"
            )
            
            fig_perf.add_hline(
                y=accuracies[0] * 100 * 0.9,  # 10% degradation
                line_dash="dot",
                line_color="rgba(245, 158, 11, 0.5)",
                annotation_text="90% of baseline",
                annotation_position="right"
            )
            
            fig_perf.update_layout(
                xaxis_title="Drift Severity (%)",
                yaxis_title="Test Accuracy (%)",
                xaxis_range=[0, 100],  # Fixed range from 0 to 100%
                height=400,
                plot_bgcolor='rgba(30, 30, 46, 0.4)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                hovermode='x unified',
                showlegend=False,
                margin=dict(l=60, r=40, t=40, b=60)
            )
            
            fig_perf.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig_perf.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Add degradation summary
            # Baseline is the accuracy at 0% drift (first evaluation point)
            baseline_acc = accuracies[0]
            
            # Interpolate accuracy at the exact drift_severity value
            # severity_levels goes from 0 to 1.0, find where drift_severity falls
            if drift_severity == 0:
                current_acc = baseline_acc
            elif drift_severity >= 1.0:
                current_acc = accuracies[-1]
            else:
                # Find the two surrounding points and interpolate
                idx = np.searchsorted(severity_levels, drift_severity)
                if idx >= len(accuracies):
                    current_acc = accuracies[-1]
                elif idx == 0:
                    current_acc = accuracies[0]
                else:
                    # Linear interpolation
                    lower_sev = severity_levels[idx - 1]
                    upper_sev = severity_levels[idx]
                    lower_acc = accuracies[idx - 1]
                    upper_acc = accuracies[idx]
                    
                    weight = (drift_severity - lower_sev) / (upper_sev - lower_sev)
                    current_acc = lower_acc + weight * (upper_acc - lower_acc)
            
            degradation = (baseline_acc - current_acc) / baseline_acc * 100
            
            # Store in session state for downstream use
            st.session_state.baseline_acc = baseline_acc
            st.session_state.current_acc = current_acc
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Baseline", f"{baseline_acc:.1%}", help="Original test accuracy")
            with col_b:
                st.metric(
                    f"At {drift_severity:.0%} Drift",
                    f"{current_acc:.1%}",
                    delta=f"{current_acc - baseline_acc:+.1%}",
                    delta_color="inverse"
                )
            with col_c:
                st.metric("Performance Drop", f"{degradation:.1f}%", help="Relative degradation from baseline")
        
        # Store current level performance for use in main metrics display
        st.session_state.baseline_acc = baseline_acc
        st.session_state.current_acc = current_acc
        
        # Display comparison at current severity
        st.markdown("## 🔍 Current Drift Level Examples")
        st.markdown(f"Visual comparison at {drift_severity:.0%} drift severity")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown(
                "<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                "padding: 10px; border-radius: 10px; margin-bottom: 15px;'>"
                "<h3 style='color: white; text-align: center; margin: 0;'>Original Data</h3>"
                "</div>",
                unsafe_allow_html=True
            )
            display_image_grid(
                original_images[:4],
                original_labels[:4],
                title=""
            )
        
        with col2:
            st.markdown(
                f"<div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); "
                "padding: 10px; border-radius: 10px; margin-bottom: 15px;'>"
                f"<h3 style='color: white; text-align: center; margin: 0;'>⚠️ Drifted Data (Severity: {drift_severity:.0%})</h3>"
                "</div>",
                unsafe_allow_html=True
            )
            display_image_grid(
                drifted_images[:4],
                original_labels[:4],
                title=""
            )
        
        # Detect drift
        st.markdown("## 📊 Drift Detection Analysis")
        st.markdown("Statistical tests quantify distribution shift severity")
        st.markdown("Statistical tests for distribution shift detection")
        st.markdown("---")
        
        with st.spinner("Detecting drift..."):
            # Flatten images for detection and normalize to 0-1
            original_flat = original_images.reshape(len(original_images), -1).astype(np.float32) / 255.0
            drifted_flat = drifted_images.reshape(len(drifted_images), -1).astype(np.float32) / 255.0
            
            # For high-dimensional data (images), sample features to reduce noise
            # Use every 4th pixel to reduce from ~784 to ~196 features
            n_features = original_flat.shape[1]
            if n_features > 200:
                # Sample features uniformly
                sample_indices = np.arange(0, n_features, max(1, n_features // 200))
                original_flat = original_flat[:, sample_indices]
                drifted_flat = drifted_flat[:, sample_indices]
            
            detector = DriftDetector()
            drift_results = detector.detect_drift(
                original_flat[:500],  # Sample for speed
                drifted_flat[:500],
                method='all'
            )
        
        # Evaluate model on current drift level (already loaded model above)
        with st.spinner(f"Evaluating {architecture} at current drift level..."):
            # Evaluate on original and drifted data (sample for speed)
            sample_size = min(500, len(original_images))
            original_acc, original_ece, _, _ = evaluate_model_on_data(
                model, original_images[:sample_size], original_labels[:sample_size], device
            )
            drifted_acc, drifted_ece, _, _ = evaluate_model_on_data(
                model, drifted_images[:sample_size], original_labels[:sample_size], device
            )
            
            # Get drift score from MMD
            mmd_result = drift_results.get('methods', {}).get('mmd', {})
            if 'distance' in mmd_result:
                drift_score = mmd_result['distance']
            elif 'distances' in mmd_result:
                drift_score = np.mean(mmd_result['distances'])
            else:
                drift_score = drift_severity * 0.4  # Fallback
            
            ece = drifted_ece
        
        # Display metrics
        display_metrics(original_acc, drifted_acc, drift_score, ece)
        
        # Drift detection results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_heatmap = plot_drift_heatmap(drift_results)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.markdown("### Detection Summary")
            
            # Handle nested structure
            methods_data = drift_results.get('methods', drift_results)
            
            for method, result in methods_data.items():
                if isinstance(result, dict) and result.get('drift_detected'):
                    # Get score for display - use 90th percentile for per-feature metrics
                    if 'score' in result:
                        score = result['score']
                    elif 'scores' in result:
                        score = np.percentile(result['scores'], 90)
                    elif 'statistics' in result:
                        score = np.percentile(result['statistics'], 90)
                    elif 'distance' in result:
                        score = result['distance']
                    elif 'distances' in result:
                        score = np.mean(result['distances'])
                    else:
                        score = 0
                    
                    st.markdown(
                        f"<div class='alert-critical'>"
                        f"<strong>DRIFT DETECTED - {method.upper()}</strong><br>"
                        f"Score: {score:.4f}<br>"
                        f"Status: <strong>DRIFT DETECTED</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                elif isinstance(result, dict):
                    # Get score for display - use 90th percentile for per-feature metrics
                    if 'score' in result:
                        score = result['score']
                    elif 'scores' in result:
                        score = np.percentile(result['scores'], 90)
                    elif 'statistics' in result:
                        score = np.percentile(result['statistics'], 90)
                    elif 'distance' in result:
                        score = result['distance']
                    elif 'distances' in result:
                        score = np.mean(result['distances'])
                    else:
                        score = 0
                    
                    st.markdown(
                        f"<div class='alert-success'>"
                        f"<strong>{method.upper()}</strong><br>"
                        f"Score: {score:.4f}<br>"
                        f"Status: <strong>NORMAL</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        
        # Alerts and recommendations
        st.markdown("## ⚠️ Alerts & Recommendations")
        st.markdown("---")
        
        alert_system = AlertSystem()
        
        # Check drift alerts using the actual drift detection results
        alerts = alert_system.check_drift_alert(drift_results)
        
        # Add custom alerts based on severity
        if drift_severity > 0.5 and not alerts:
            # Manual alert for high severity
            alerts.append({
                'type': 'drift',
                'severity': 'critical',
                'message': f"Severe drift detected (severity: {drift_severity:.0%})",
                'recommendations': [
                    "Consider model retraining immediately",
                    "Evaluate current predictions for reliability",
                    "Implement temporary fallback mechanisms",
                    "Investigate root cause of drift"
                ]
            })
        elif drift_severity > 0.3 and not alerts:
            alerts.append({
                'type': 'drift',
                'severity': 'warning',
                'message': f"Moderate drift detected (severity: {drift_severity:.0%})",
                'recommendations': [
                    "Monitor predictions closely",
                    "Collect new data for potential retraining",
                    "Analyze drift sources"
                ]
            })
        
        # Consolidate alerts by severity
        if alerts:
            # Determine highest severity
            has_critical = any(a['severity'] == 'critical' for a in alerts)
            has_warning = any(a['severity'] == 'warning' for a in alerts)
            
            if has_critical:
                severity_class = "alert-critical"
                severity_label = "CRITICAL"
            elif has_warning:
                severity_class = "alert-warning"
                severity_label = "WARNING"
            else:
                severity_class = "alert-success"
                severity_label = "INFO"
            
            # Collect all unique recommendations
            all_recommendations = []
            for alert in alerts:
                all_recommendations.extend(alert.get('recommendations', []))
            
            # Remove duplicates while preserving order
            unique_recommendations = []
            for rec in all_recommendations:
                if rec not in unique_recommendations:
                    unique_recommendations.append(rec)
            
            # Display consolidated alert
            st.markdown(
                f"<div class='{severity_class}'>"
                f"<strong>{severity_label}</strong>: Drift detected at {drift_severity:.0%} severity<br><br>"
                f"<strong>Recommended Actions:</strong><br>"
                + "<br>".join(f"• {r}" for r in unique_recommendations[:5])  # Limit to 5 recommendations
                + f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='alert-success'>"
                "<strong>✓ All Systems Normal</strong><br>"
                "No alerts detected. Model is performing within expected parameters."
                "</div>",
                unsafe_allow_html=True
            )
    
    else:
        # Show landing page before first simulation
        display_landing_page()


if __name__ == '__main__':
    main()
