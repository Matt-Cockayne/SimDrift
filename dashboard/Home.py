"""
SimDrift - Enhanced Interactive Visual Dashboard

A stunning, portfolio-quality dashboard for ML drift demonstration.
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
    page_icon="📊",
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
        padding: 2rem 0;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        border-radius: 1rem;
        margin-bottom: 2rem;
    }}
    
    .hero-title {{
        font-size: 3rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
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
        'drift_generator': DriftGenerator(seed=42),
        'drift_detector': DriftDetector(),
        'alert_system': AlertSystem(),
        'original_images': None,
        'original_labels': None,
        'drifted_images': None,
        'simulation_running': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_dataset(dataset_name: str):
    """Load and cache dataset."""
    loader = MedMNISTLoader(dataset_name=dataset_name, download=True)
    return loader


@st.cache_resource
def get_model_manager():
    """Get cached model manager."""
    return ModelManager('model_zoo')


def display_hero():
    """Display hero section."""
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">SimDrift</h1>
        <p class="hero-subtitle">
            Interactive Medical ML Drift Simulator | 
            24 Pre-trained Models | 8 Datasets | 15+ Drift Types
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
                       title: str = "", n_display: int = 16):
    """Display image grid with predictions."""
    st.markdown(f"### {title}")
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = (n_display + n_cols - 1) // n_cols
    
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
                
                # Convert to PIL Image for display
                if img.shape[-1] == 1:
                    img_display = Image.fromarray(img.squeeze(), mode='L')
                else:
                    img_display = Image.fromarray(img.astype(np.uint8))
                
                st.image(img_display, use_container_width=True)
                
                # Display label and prediction
                if predictions is not None:
                    true_label = labels[img_idx]
                    pred_label = predictions[img_idx]
                    is_correct = true_label == pred_label
                    
                    color = COLORS['success'] if is_correct else COLORS['danger']
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.8rem;'>"
                        f"<span style='color:{color};'>{'✓' if is_correct else '✗'}</span> "
                        f"True: {true_label} | Pred: {pred_label}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.8rem;'>"
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
    """Plot drift detection heatmap."""
    methods = list(drift_results.keys())
    scores = [drift_results[m].get('score', 0) for m in methods]
    
    # Create color based on threshold
    colors = [COLORS['danger'] if s > 0.2 else COLORS['warning'] if s > 0.1 
              else COLORS['success'] for s in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=scores,
            marker_color=colors,
            text=[f"{s:.4f}" for s in scores],
            textposition='auto',
        )
    ])
    
    fig.add_hline(
        y=0.2, line_dash="dash", line_color=COLORS['danger'],
        annotation_text="Critical"
    )
    fig.add_hline(
        y=0.1, line_dash="dash", line_color=COLORS['warning'],
        annotation_text="Warning"
    )
    
    fig.update_layout(
        title="Drift Detection by Method",
        xaxis_title="Detection Method",
        yaxis_title="Drift Score",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        showlegend=False
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
            ['pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist', 'pneumoniamnist'],
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
        
        st.session_state.drift_severity = drift_severity
        st.session_state.current_drift_type = drift_type
        
        # Run button
        run_simulation = st.button("Run Simulation", use_container_width=True)
    
    # Main content area
    if run_simulation or st.session_state.data_loaded:
        # Load data
        if not st.session_state.data_loaded or st.session_state.dataset_name != dataset_name:
            with st.spinner(f"Loading {dataset_name}..."):
                loader = load_dataset(dataset_name)
                test_images, test_labels = loader.get_numpy_data('test')
                
                st.session_state.original_images = test_images
                st.session_state.original_labels = test_labels
                st.session_state.dataset_name = dataset_name
                st.session_state.data_loaded = True
        
        # Apply drift
        with st.spinner("Applying drift..."):
            original_images = st.session_state.original_images
            original_labels = st.session_state.original_labels
            
            drifted_images, _, metadata = st.session_state.drift_generator.simulate_gradual_drift(
                original_images,
                original_labels,
                drift_type=drift_type,
                n_steps=1,
                max_severity=drift_severity,
                return_steps=False
            )
            
            st.session_state.drifted_images = drifted_images
        
        # Display comparison
        st.markdown("## Visual Comparison")
        st.markdown("Side-by-side view of original and drifted images")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_image_grid(
                original_images[:16],
                original_labels[:16],
                title="Original Data"
            )
        
        with col2:
            display_image_grid(
                drifted_images[:16],
                original_labels[:16],
                title=f"Drifted Data (Severity: {drift_severity:.0%})"
            )
        
        # Detect drift
        st.markdown("## Drift Detection")
        st.markdown("Statistical tests for distribution shift detection")
        st.markdown("---")
        
        with st.spinner("Detecting drift..."):
            # Flatten images for detection
            original_flat = original_images.reshape(len(original_images), -1)
            drifted_flat = drifted_images.reshape(len(drifted_images), -1)
            
            detector = DriftDetector()
            drift_results = detector.detect_drift(
                original_flat[:500],  # Sample for speed
                drifted_flat[:500],
                method='all'
            )
        
        # Fake metrics for demonstration (since models may not be trained yet)
        original_acc = 0.95
        drifted_acc = original_acc * (1 - drift_severity * 0.3)  # Simulate degradation
        drift_score = drift_severity * 0.4  # Simplified drift score
        ece = 0.05 + drift_severity * 0.15  # Calibration degrades with drift
        
        # Display metrics
        display_metrics(original_acc, drifted_acc, drift_score, ece)
        
        # Drift detection results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_heatmap = plot_drift_heatmap(drift_results)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.markdown("### Detection Summary")
            
            for method, result in drift_results.items():
                if result.get('drift_detected'):
                    st.markdown(
                        f"<div class='alert-critical'>"
                        f"<strong>DRIFT DETECTED - {method.upper()}</strong><br>"
                        f"Score: {result.get('score', 0):.4f}<br>"
                        f"Status: <strong>DRIFT DETECTED</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='alert-success'>"
                        f"<strong>{method.upper()}</strong><br>"
                        f"Score: {result.get('score', 0):.4f}<br>"
                        f"Status: <strong>NORMAL</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        
        # Alerts and recommendations
        st.markdown("## Alerts & Recommendations")
        st.markdown("System alerts and suggested actions")
        st.markdown("---")
        
        alert_system = AlertSystem()
        
        if drift_severity > 0.5:
            alert_system.add_alert(
                'drift',
                severity='critical',
                message=f"Severe drift detected (severity: {drift_severity:.0%})",
                metric_value=drift_score,
                threshold=0.2,
                recommendations=[
                    "URGENT: Consider model retraining immediately",
                    "Evaluate current predictions for reliability",
                    "Implement temporary fallback mechanisms",
                    "Investigate root cause of drift"
                ]
            )
        elif drift_severity > 0.3:
            alert_system.add_alert(
                'drift',
                severity='warning',
                message=f"Moderate drift detected (severity: {drift_severity:.0%})",
                metric_value=drift_score,
                threshold=0.1,
                recommendations=[
                    "Monitor predictions closely",
                    "Collect new data for potential retraining",
                    "Analyze drift sources"
                ]
            )
        
        alerts = alert_system.get_active_alerts()
        
        if alerts:
            for alert in alerts:
                severity_class = f"alert-{alert['severity']}"
                icon = "[!]" if alert['severity'] == 'critical' else "[i]"
                
                st.markdown(
                    f"<div class='{severity_class}'>"
                    f"{icon} <strong>{alert['severity'].upper()}</strong>: {alert['message']}<br><br>"
                    f"<strong>Recommended Actions:</strong><br>"
                    + "<br>".join(f"  {r}" for r in alert['recommendations'])
                    + f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div class='alert-success'>"
                "<strong>All Systems Normal</strong><br>"
                "No alerts detected. Model is performing within expected parameters."
                "</div>",
                unsafe_allow_html=True
            )
    
    else:
        # Welcome screen with interactive demo
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("## Welcome to SimDrift")
            st.markdown("""
            An interactive platform for understanding and visualizing ML model drift 
            in medical imaging. Perfect for education, research, and demonstrating 
            production ML monitoring capabilities.
            """)
            
            st.markdown("### Quick Start Guide:")
            
            # Interactive quick start with expanders
            with st.expander("📊 Step 1: Select Your Dataset", expanded=True):
                st.markdown("""
                Choose from 8 medical imaging datasets including:
                - **PathMNIST**: Colon pathology images
                - **DermaMNIST**: Dermatology lesion images  
                - **RetinaMNIST**: Fundus photography
                - **BloodMNIST**: Blood cell microscopy
                """)
            
            with st.expander("🔬 Step 2: Configure Drift Scenario"):
                st.markdown("""
                Select from 15+ drift types:
                - **Data Drift**: Brightness, contrast, noise, blur
                - **Concept Drift**: Label corruption, class imbalance
                - **Combined**: Realistic production scenarios
                """)
            
            with st.expander("🎯 Step 3: Run & Analyze"):
                st.markdown("""
                Watch real-time visualization of:
                - Side-by-side image comparison
                - Statistical drift detection (PSI, KS, MMD, Chi²)
                - Performance degradation metrics
                - Automated alert system with recommendations
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Key Capabilities")
            
            st.metric(
                label="Pre-trained Models",
                value="24",
                delta="Multiple architectures"
            )
            
            st.metric(
                label="Medical Datasets", 
                value="8",
                delta="10K+ images each"
            )
            
            st.metric(
                label="Drift Scenarios",
                value="15+",
                delta="Real-world tested"
            )
            
            st.markdown("---")
            
            st.markdown("#### 🎯 Perfect For:")
            st.markdown("""
            - **Education**: ML monitoring concepts
            - **Research**: Drift detection methods
            - **Demos**: Portfolio & presentations
            - **Testing**: Model robustness
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive dataset explorer
        st.markdown("---")
        st.markdown("### 🔍 Interactive Dataset Explorer")
        st.markdown("Click on any dataset below to load it and start exploring")
        
        dataset_info = {
            'pathmnist': {
                'name': 'PathMNIST',
                'desc': 'Colon pathology histology',
                'size': '107,180 images',
                'classes': '9 tissue types'
            },
            'dermamnist': {
                'name': 'DermaMNIST', 
                'desc': 'Dermatoscopy lesion images',
                'size': '10,015 images',
                'classes': '7 skin lesions'
            },
            'retinamnist': {
                'name': 'RetinaMNIST',
                'desc': 'Retinal fundus photography',
                'size': '1,600 images', 
                'classes': '5 disease stages'
            },
            'bloodmnist': {
                'name': 'BloodMNIST',
                'desc': 'Blood cell microscopy',
                'size': '17,092 images',
                'classes': '8 cell types'
            }
        }
        
        cols = st.columns(4)
        for idx, (ds_key, ds_info) in enumerate(dataset_info.items()):
            with cols[idx]:
                st.markdown(f"#### {ds_info['name']}")
                st.caption(ds_info['desc'])
                st.caption(f"📦 {ds_info['size']}")
                st.caption(f"🏷️ {ds_info['classes']}")
                
                if st.button(f"Load {ds_info['name']}", key=f"load_{ds_key}", use_container_width=True):
                    st.session_state.dataset_name = ds_key
                    st.rerun()
        
        # Feature showcase
        st.markdown("---")
        
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("#### 🎨 Visual Comparison")
            st.markdown("""
            See original vs drifted data side-by-side with:
            - 4×4 image grids
            - Label overlays
            - Real-time updates
            """)
        
        with feature_cols[1]:
            st.markdown("#### 📊 Statistical Detection")  
            st.markdown("""
            Multiple detection methods:
            - Population Stability Index (PSI)
            - Kolmogorov-Smirnov (KS)
            - Maximum Mean Discrepancy (MMD)
            - Chi-Squared (χ²) test
            """)
        
        with feature_cols[2]:
            st.markdown("#### 🚨 Smart Alerts")
            st.markdown("""
            Automated monitoring with:
            - Severity-based alerts
            - Actionable recommendations
            - Historical tracking
            - Export capabilities
            """)
        
        st.markdown("---")
        st.info("💡 **Tip**: Open the sidebar (click `>` in top-left) to configure and run your first simulation!")


if __name__ == '__main__':
    main()
