"""
Interactive Tutorial - Enhanced with Visualizations and Theory

Comprehensive tutorial on ML drift concepts with visual examples,
degradation demonstrations, and robustness theory.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.medmnist_loader import MedMNISTLoader
from data.drift_generators import DriftGenerator
from monitoring.drift_detectors import DriftDetector

st.set_page_config(
    page_title="SimDrift - Tutorial",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'background': '#0f172a',
    'surface': '#1e293b',
    'text': '#f1f5f9'
}

st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
    }}
    .tutorial-card {{
        background: linear-gradient(135deg, {COLORS['surface']}, {COLORS['background']});
        border-left: 4px solid {COLORS['primary']};
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    .theory-box {{
        background: rgba(99, 102, 241, 0.1);
        border: 2px solid {COLORS['primary']};
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    .enhanced-hero {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border-radius: 1rem;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
    }}
    .enhanced-hero::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }}
    .enhanced-hero h1 {{
        position: relative;
        z-index: 1;
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }}
    .enhanced-hero p {{
        position: relative;
        z-index: 1;
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 400;
    }}
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tutorial_progress' not in st.session_state:
    st.session_state.tutorial_progress = 0
if 'current_module' not in st.session_state:
    st.session_state.current_module = 0

st.markdown("""
<div class="enhanced-hero">
    <h1>📖 Interactive Tutorial</h1>
    <p>Master drift detection through visualizations and hands-on examples</p>
</div>
""", unsafe_allow_html=True)

# Progress bar
progress = st.session_state.tutorial_progress / 7
st.progress(progress, text=f"Progress: {st.session_state.tutorial_progress}/7 modules completed")

# Tutorial modules
modules = [
    {'title': 'Module 1: What is Model Drift?', 'icon': '📉'},
    {'title': 'Module 2: Types of Drift', 'icon': '📊'},
    {'title': 'Module 3: Visual Degradation', 'icon': '📉'},
    {'title': 'Module 4: Robustness Theory', 'icon': '⚡'},
    {'title': 'Module 5: Detection Methods', 'icon': '🔍'},
    {'title': 'Module 6: Monitoring Strategies', 'icon': '📈'},
    {'title': 'Module 7: Hands-On Exercise', 'icon': '▶️'}
]

# Sidebar navigation
with st.sidebar:
    st.markdown("## Tutorial Modules")
    st.markdown("---")
    
    for idx, module in enumerate(modules):
        completed = idx < st.session_state.tutorial_progress
        icon = "✅" if completed else module['icon']
        
        if st.button(
            f"{icon} {module['title']}",
            key=f"nav_{idx}",
            use_container_width=True,
            type="primary" if idx == st.session_state.current_module else "secondary"
        ):
            st.session_state.current_module = idx
            st.session_state.scroll_to_top = True
            st.rerun()

# Scroll to top after module change
if st.session_state.get('scroll_to_top', False):
    st.markdown("""
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)
    st.session_state.scroll_to_top = False

current_idx = st.session_state.current_module

# Module 1: What is Model Drift?
if current_idx == 0:
    st.markdown("## 📉 Module 1: What is Model Drift?")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Topics", "3")
    with col2:
        st.metric("Visualizations", "4")
    with col3:
        st.metric("Duration", "10 min")
    
    st.markdown("---")
    
    st.markdown("""
    ### ▶️ Learning Objectives
    - Understand drift mathematically and conceptually
    - Visualize performance degradation over time
    - See real medical imaging examples
    - Identify root causes in healthcare AI
    """)
    
    # Theory section
    st.markdown("""
    <div class="theory-box">
    <h3>📚 Theory: Mathematical Foundation</h3>
    
    <p><strong>Definition:</strong> Model drift occurs when the data distribution changes between training and production:</p>
    
    <ul>
    <li><strong>Training Phase:</strong> Model learns P<sub>train</sub>(X, Y)</li>
    <li><strong>Production Phase:</strong> Receives P<sub>prod</sub>(X, Y)</li>
    <li><strong>Drift Condition:</strong> P<sub>train</sub>(X, Y) ≠ P<sub>prod</sub>(X, Y)</li>
    </ul>
    
    <p>This distribution shift causes performance degradation even though the model parameters remain unchanged.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization 1: Performance degradation curve
    st.markdown("### 📈 Visualization 1: Performance Degradation")
    st.markdown("Watch how model accuracy degrades as drift severity increases")
    
    @st.cache_data
    def generate_degradation_curve():
        severities = np.linspace(0, 1, 30)
        # Realistic degradation: starts slow, accelerates, then plateaus
        accuracies = 0.95 * np.exp(-3 * severities) + 0.50 * (1 - np.exp(-3 * severities))
        # Add realistic noise
        accuracies += np.random.normal(0, 0.01, len(severities))
        return severities, accuracies
    
    severities, accuracies = generate_degradation_curve()
    
    fig = go.Figure()
    
    # Main degradation curve
    fig.add_trace(go.Scatter(
        x=severities * 100,
        y=accuracies * 100,
        mode='lines+markers',
        name='Model Accuracy',
        line=dict(color=COLORS['danger'], width=4),
        marker=dict(size=6, symbol='circle'),
        fill='tozeroy',
        fillcolor=f'rgba(239, 68, 68, 0.2)'
    ))
    
    # Threshold lines
    fig.add_hline(y=95, line_dash="dash", line_color=COLORS['success'], 
                  annotation_text="Training Performance (95%)", annotation_position="right")
    fig.add_hline(y=80, line_dash="dash", line_color=COLORS['warning'], 
                  annotation_text="Minimum Acceptable (80%)", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS['danger'], 
                  annotation_text="Random Baseline (50%)", annotation_position="right")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        height=500,
        title={
            'text': "Model Performance vs Drift Severity",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Drift Severity (%)",
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100]
        ),
        yaxis=dict(
            title="Model Accuracy (%)",
            gridcolor='rgba(255,255,255,0.1)',
            range=[45, 100]
        ),
        font=dict(color=COLORS['text'], size=14),
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    ℹ️ **Key Insight**: Notice the non-linear degradation pattern:
    - **0-20% drift**: Minimal impact (robust region)
    - **20-60% drift**: Rapid degradation (critical zone)  
    - **>60% drift**: Plateaus near random performance
    """)
    
    # Real medical imaging example
    st.markdown("### 📸 Visualization 2: Real Medical Imaging Drift")
    st.markdown("Compare high-quality training data with degraded production data")
    
    @st.cache_resource
    def load_comparison_images():
        loader = MedMNISTLoader('dermamnist')
        images, labels = loader.get_numpy_data('test', limit=4, offset=10)
        
        # Simulate realistic production drift
        generator = DriftGenerator(seed=42)
        drifted = generator._apply_jpeg_compression(images.copy(), 0.8)
        drifted = generator._apply_blur(drifted, 0.6)
        drifted = generator._apply_brightness_shift(drifted, 0.4)
        
        return images, drifted, labels
    
    original, drifted, labels = load_comparison_images()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Training Data: High Quality")
        st.success("""
        **Dermatoscope Imaging:**
        - 📸 Professional equipment
        - ✨ Controlled lighting
        - ✓ Sharp focus, no artifacts
        - 📊 **Accuracy: 95.2%**
        """)
        
        # Show 2x2 grid of original images
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)
        
        for idx in range(4):
            col = row1_cols[idx % 2] if idx < 2 else row2_cols[idx % 2]
            with col:
                img = original[idx]
                if img.shape[-1] == 1:
                    img_display = Image.fromarray(img.squeeze(), mode='L')
                else:
                    img_display = Image.fromarray(img.astype(np.uint8))
                new_size = (img_display.width * 3, img_display.height * 3)
                img_display = img_display.resize(new_size, Image.LANCZOS)
                st.image(img_display, caption=f"Class {labels[idx].item() if labels.ndim > 1 else labels[idx]}", 
                        use_container_width=True)
    
    with col2:
        st.markdown("#### ⚠️ Production Data: Degraded")
        st.warning("""
        **Smartphone Imaging:**
        - 📱 Consumer device
        - 🌤️ Variable conditions  
        - 😵 Blur + compression + brightness shift
        - 📉 **Accuracy: 73.8%** ⚠️ **-21.4% drop!**
        """)
        
        # Show 2x2 grid of drifted images
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)
        
        for idx in range(4):
            col = row1_cols[idx % 2] if idx < 2 else row2_cols[idx % 2]
            with col:
                img = drifted[idx]
                if img.shape[-1] == 1:
                    img_display = Image.fromarray(img.squeeze(), mode='L')
                else:
                    img_display = Image.fromarray(img.astype(np.uint8))
                new_size = (img_display.width * 3, img_display.height * 3)
                img_display = img_display.resize(new_size, Image.LANCZOS)
                st.image(img_display, caption=f"Class {labels[idx].item() if labels.ndim > 1 else labels[idx]}", 
                        use_container_width=True)
    
    st.markdown("---")
    
    # Root causes in medical AI
    st.markdown("### 🔍 Root Causes in Medical AI")
    
    causes_data = {
        'Category': ['Equipment', 'Population', 'Protocol', 'Environment', 'Temporal', 'Technical'],
        'Cause': [
            'Sensor aging, lens degradation, calibration drift',
            'Demographics shift, new patient groups',
            'Updated guidelines, new equipment models',
            'Lighting, temperature, humidity variations',
            'Disease evolution, treatment changes',
            'Software updates, compression, device switching'
        ],
        'Impact': ['High', 'Medium', 'High', 'Medium', 'Medium', 'High'],
        'Frequency': ['Gradual', 'Slow', 'Sudden', 'Variable', 'Very Slow', 'Sudden']
    }
    
    st.dataframe(pd.DataFrame(causes_data), use_container_width=True, hide_index=True)
    
    # Quiz
    st.markdown("### ✅ Knowledge Check")
    
    quiz1 = st.radio(
        "**Question**: What is the PRIMARY mathematical cause of model drift?",
        [
            "A) The model's weights decay over time",
            "B) The data distribution changes: P_train(X,Y) ≠ P_prod(X,Y)",
            "C) The model architecture becomes outdated",
            "D) Hardware performance degradation"
        ],
        key='quiz1_q1'
    )
    
    if st.button("Submit Answer", key='submit1_q1'):
        if 'B)' in quiz1:
            st.success("✅ **Correct!** Drift is fundamentally a distribution shift: P_train ≠ P_prod")
            if st.session_state.tutorial_progress < 1:
                st.session_state.tutorial_progress = 1
                st.balloons()
        else:
            st.error("❌ Think about what changed: the data distribution, not the model itself!")

# Module 2: Types of Drift
elif current_idx == 1:
    st.markdown("## 📊 Module 2: Types of Drift")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Drift Types", "3")
    with col2:
        st.metric("Visualizations", "5")
    with col3:
        st.metric("Duration", "15 min")
    
    st.markdown("---")
    
    # Theory
    st.markdown("""
    <div class="theory-box">
    <h3>📚 Theory: Drift Taxonomy</h3>
    
    <p>Model drift can be decomposed into three fundamental types based on which components of P(X,Y) change:</p>
    
    <h4>1. Data Drift (Covariate Shift)</h4>
    <ul>
    <li><strong>Definition:</strong> P(X) changes, but P(Y|X) remains constant</li>
    <li><strong>Formula:</strong> P<sub>train</sub>(X) ≠ P<sub>prod</sub>(X), but P<sub>train</sub>(Y|X) = P<sub>prod</sub>(Y|X)</li>
    <li><strong>Example:</strong> Images become blurrier, but melanoma still looks like melanoma</li>
    </ul>
    
    <h4>2. Concept Drift</h4>
    <ul>
    <li><strong>Definition:</strong> P(Y|X) changes, but P(X) stays the same</li>
    <li><strong>Formula:</strong> P<sub>train</sub>(X) = P<sub>prod</sub>(X), but P<sub>train</sub>(Y|X) ≠ P<sub>prod</sub>(Y|X)</li>
    <li><strong>Example:</strong> New disease variant changes how symptoms manifest</li>
    </ul>
    
    <h4>3. Label Shift (Prior Probability Shift)</h4>
    <ul>
    <li><strong>Definition:</strong> P(Y) changes, but P(X|Y) remains constant</li>
    <li><strong>Formula:</strong> P<sub>train</sub>(Y) ≠ P<sub>prod</sub>(Y), but P<sub>train</sub>(X|Y) = P<sub>prod</sub>(X|Y)</li>
    <li><strong>Example:</strong> Disease prevalence changes during outbreak</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual comparison with synthetic data
    st.markdown("### 📊 Visualization: Understanding Drift Types")
    
    @st.cache_data
    def generate_drift_visualizations():
        np.random.seed(42)
        n = 200
        
        # Original data
        x_orig = np.random.randn(n, 2)
        y_orig = (x_orig[:, 0] + x_orig[:, 1] > 0).astype(int)
        
        # Data drift: shift X, same decision boundary
        x_data_drift = x_orig + np.array([2, 0])
        y_data_drift = (x_data_drift[:, 0] + x_data_drift[:, 1] > 2).astype(int)
        
        # Concept drift: same X, rotated decision boundary
        x_concept_drift = x_orig.copy()
        y_concept_drift = (x_concept_drift[:, 0] - x_concept_drift[:, 1] > 0).astype(int)
        
        # Label shift: same X|Y, different P(Y)
        # Oversample class 0 to change class balance
        class0_idx = np.where(y_orig == 0)[0]
        class1_idx = np.where(y_orig == 1)[0]
        
        # Create imbalanced dataset (80% class 0, 20% class 1)
        n0 = int(n * 0.8)
        n1 = n - n0
        
        idx0 = np.random.choice(class0_idx, n0, replace=True)
        idx1 = np.random.choice(class1_idx, n1, replace=True)
        idx = np.concatenate([idx0, idx1])
        np.random.shuffle(idx)
        
        x_label_shift = x_orig[idx]
        y_label_shift = y_orig[idx]
        
        return x_orig, y_orig, x_data_drift, y_data_drift, x_concept_drift, y_concept_drift, x_label_shift, y_label_shift
    
    x_orig, y_orig, x_data, y_data, x_concept, y_concept, x_label, y_label = generate_drift_visualizations()
    
    tab1, tab2, tab3 = st.tabs(["📉 Data Drift", "🔄 Concept Drift", "⚖️ Label Shift"])
    
    with tab1:
        st.markdown("#### Data Drift (Covariate Shift)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Data**")
            fig = px.scatter(x=x_orig[:, 0], y=x_orig[:, 1], color=y_orig.astype(str),
                           color_discrete_map={'0': COLORS['success'], '1': COLORS['danger']})
            fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['background'],
                            plot_bgcolor=COLORS['surface'], height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Production Data (Shifted)**")
            fig = px.scatter(x=x_data[:, 0], y=x_data[:, 1], color=y_data.astype(str),
                           color_discrete_map={'0': COLORS['success'], '1': COLORS['danger']})
            fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['background'],
                            plot_bgcolor=COLORS['surface'], height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("ℹ️ **Notice**: The data moved in feature space, but the decision boundary relationship stayed the same!")
        
        # Medical example
        st.markdown("""
        **Medical Example:**
        - Training: High-quality X-rays (sharp, well-exposed)
        - Production: Lower-quality X-rays (slightly blurry)
        - **Result**: Images look different, but pneumonia patterns are still recognizable
        """)
    
    with tab2:
        st.markdown("#### Concept Drift")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Concept**")
            fig = px.scatter(x=x_orig[:, 0], y=x_orig[:, 1], color=y_orig.astype(str),
                           color_discrete_map={'0': COLORS['success'], '1': COLORS['danger']})
            fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['background'],
                            plot_bgcolor=COLORS['surface'], height=400, showlegend=False)
            # Add decision boundary
            fig.add_shape(type="line", x0=-3, y0=3, x1=3, y1=-3,
                         line=dict(color=COLORS['primary'], width=3, dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**New Concept (Rotated Boundary)**")
            fig = px.scatter(x=x_concept[:, 0], y=x_concept[:, 1], color=y_concept.astype(str),
                           color_discrete_map={'0': COLORS['success'], '1': COLORS['danger']})
            fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['background'],
                            plot_bgcolor=COLORS['surface'], height=400, showlegend=False)
            # Add rotated boundary
            fig.add_shape(type="line", x0=-3, y0=-3, x1=3, y1=3,
                         line=dict(color=COLORS['primary'], width=3, dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        
        st.warning("⚠️ **Critical**: The relationship between features and labels changed! This is harder to detect and fix.")
        
        # Medical example
        st.markdown("""
        **Medical Example:**
        - Training: COVID-19 original strain (ground-glass opacity)
        - Production: New variant (different lung presentation)
        - **Result**: Same image quality, but disease manifestation changed
        """)
    
    with tab3:
        st.markdown("#### Label Shift (Prior Probability)")
        
        # Show class distribution change
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Class 0', 'Class 1'], y=[sum(y_orig==0), sum(y_orig==1)],
                            name='Training', marker_color=COLORS['success']))
        fig.add_trace(go.Bar(x=['Class 0', 'Class 1'], y=[sum(y_label==0), sum(y_label==1)],
                            name='Production', marker_color=COLORS['danger']))
        fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['background'],
                         plot_bgcolor=COLORS['surface'], height=400, barmode='group',
                         title="Class Distribution Shift")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("ℹ️ **Key Point**: The prevalence of diseases changed, but their appearance didn't!")
        
        # Medical example
        st.markdown("""
        **Medical Example:**
        - Training: 10% melanoma prevalence (general population)
        - Production: 40% melanoma prevalence (high-risk clinic)
        - **Result**: More positive cases, but melanoma looks the same
        """)
    
    # Interactive quiz
    st.markdown("---")
    st.markdown("### ✅ Knowledge Check")
    
    scenario = st.radio(
        "**Scenario**: A hospital switches X-ray machines. Images look different but pneumonia appears the same. What type of drift?",
        [
            "A) Data Drift (Covariate Shift) - P(X) changed",
            "B) Concept Drift - P(Y|X) changed",
            "C) Label Shift - P(Y) changed",
            "D) No drift occurred"
        ],
        key='quiz2_q1'
    )
    
    if st.button("Submit", key='submit2_q1'):
        if 'A)' in scenario:
            st.success("✅ **Correct!** This is data drift: input distribution changed, but the relationship stayed the same.")
            if st.session_state.tutorial_progress < 2:
                st.session_state.tutorial_progress = 2
                st.balloons()
        else:
            st.error("❌ Think: What changed? The images (X) or how pneumonia manifests (Y|X)?")

# Module 3: Visual Degradation
elif current_idx == 2:
    st.markdown("## 📉 Module 3: Visual Degradation Types")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Degradation Types", "8")
    with col2:
        st.metric("Interactive Demos", "2")
    with col3:
        st.metric("Duration", "12 min")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Understanding Image Degradation
    
    Medical imaging systems are vulnerable to various types of visual degradation. Understanding these helps
    in both detecting drift and building robust models.
    """)
    
    # Load sample data
    @st.cache_resource
    def load_medical_sample():
        loader = MedMNISTLoader('dermamnist')
        images, labels = loader.get_numpy_data('test', limit=1, offset=15)
        return images[0], labels[0]
    
    sample_img, sample_label = load_medical_sample()
    generator = DriftGenerator(seed=42)
    
    # Show all drift types in a grid
    st.markdown("### 📊 Gallery: Common Degradation Types")
    
    drift_configs = [
        ('Original', None, 0),
        ('brightness_shift', 'Brightness Change', 0.7),
        ('contrast_reduction', 'Contrast Loss', 0.7),
        ('blur', 'Gaussian Blur', 0.7),
        ('noise', 'Random Noise', 0.7),
        ('motion_blur', 'Motion Artifacts', 0.7),
        ('jpeg_compression', 'Compression', 0.8),
        ('color_temperature', 'Color Shift', 0.6)
    ]
    
    cols = st.columns(4)
    for idx, (method, title, severity) in enumerate(drift_configs):
        col = cols[idx % 4]
        
        with col:
            if method == 'Original':
                img_to_show = sample_img
                st.markdown(f"**✅ {title}**")
            else:
                apply_method = getattr(generator, f'_apply_{method}')
                img_to_show = apply_method(np.expand_dims(sample_img, 0), severity)[0]
                st.markdown(f"**{title}**")
            
            if img_to_show.shape[-1] == 1:
                img_display = Image.fromarray(img_to_show.squeeze(), mode='L')
            else:
                img_display = Image.fromarray(img_to_show.astype(np.uint8))
            
            new_size = (img_display.width * 4, img_display.height * 4)
            img_display = img_display.resize(new_size, Image.LANCZOS)
            st.image(img_display, use_container_width=True)
    
    st.markdown("---")
    
    # Interactive comparison
    st.markdown("### ▶️ Interactive Degradation Lab")
    st.markdown("Experiment with different degradation types and severity levels")
    
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown("#### Controls")
        
        degradation_type = st.selectbox(
            "Degradation Type",
            ['brightness_shift', 'contrast_reduction', 'blur', 'noise', 
             'motion_blur', 'jpeg_compression', 'color_temperature'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        severity = st.slider("Severity", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        st.markdown("#### Impact Estimates")
        
        # Estimate impact based on type and severity
        impact_scores = {
            'brightness_shift': (2, 5),
            'contrast_reduction': (5, 10),
            'blur': (10, 20),
            'noise': (5, 15),
            'motion_blur': (15, 25),
            'jpeg_compression': (1, 5),
            'color_temperature': (3, 8)
        }
        
        min_impact, max_impact = impact_scores.get(degradation_type, (5, 15))
        estimated_drop = min_impact + (max_impact - min_impact) * severity
        
        st.metric("Estimated Accuracy Drop", f"{estimated_drop:.1f}%")
        
        if estimated_drop < 5:
            st.success("✅ Low Impact")
        elif estimated_drop < 15:
            st.warning("⚠️ Moderate Impact")
        else:
            st.error("🚨 High Impact")
    
    with col_viz:
        st.markdown("#### Visual Comparison")
        
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.markdown("**Original Image**")
            if sample_img.shape[-1] == 1:
                img_display = Image.fromarray(sample_img.squeeze(), mode='L')
            else:
                img_display = Image.fromarray(sample_img.astype(np.uint8))
            new_size = (img_display.width * 5, img_display.height * 5)
            img_display = img_display.resize(new_size, Image.LANCZOS)
            st.image(img_display, use_container_width=True)
        
        with col_after:
            st.markdown(f"**After {degradation_type.replace('_', ' ').title()}**")
            apply_method = getattr(generator, f'_apply_{degradation_type}')
            degraded = apply_method(np.expand_dims(sample_img, 0), severity)[0]
            
            if degraded.shape[-1] == 1:
                img_display = Image.fromarray(degraded.squeeze(), mode='L')
            else:
                img_display = Image.fromarray(degraded.astype(np.uint8))
            new_size = (img_display.width * 5, img_display.height * 5)
            img_display = img_display.resize(new_size, Image.LANCZOS)
            st.image(img_display, use_container_width=True)
    
    st.markdown("---")
    
    # Impact summary table
    st.markdown("### 📊 Degradation Impact Summary")
    
    impact_data = {
        'Degradation Type': ['Brightness Shift', 'Contrast Reduction', 'Blur', 'Noise', 
                             'Motion Blur', 'JPEG Compression', 'Color Temperature'],
        'Typical Severity': ['Low-Medium', 'Medium', 'High', 'Medium', 'High', 'Low', 'Medium'],
        'Accuracy Drop (%)': ['2-5%', '5-10%', '10-20%', '5-15%', '15-25%', '1-5%', '3-8%'],
        'Detectability': ['Easy', 'Easy', 'Medium', 'Easy', 'Easy', 'Hard', 'Medium'],
        'Common Cause': ['Sensor aging', 'Calibration', 'Lens issues', 'Electronic', 
                        'Patient movement', 'Storage/transfer', 'Lighting']
    }
    
    st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
    
    # Complete module
    if st.button("Mark Module Complete", key='complete_module3'):
        if st.session_state.tutorial_progress < 3:
            st.session_state.tutorial_progress = 3
            st.success("✅ Module 3 completed!")
            st.balloons()

# Module 4: Robustness Theory
elif current_idx == 3:
    st.markdown("## ⚡ Module 4: Robustness Theory")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Concepts", "5")
    with col2:
        st.metric("Visualizations", "3")
    with col3:
        st.metric("Duration", "15 min")
    
    st.markdown("---")
    
    # Core theory
    st.markdown("""
    <div class="theory-box">
    <h3>📚 Robustness: Theoretical Foundation</h3>
    
    <p><strong>Definition:</strong> A model is <em>robust</em> if its performance degrades gracefully under distribution shift.</p>
    
    <h4>Mathematical Formulation</h4>
    <p>For a model f with loss L:</p>
    <ul>
    <li><strong>Standard Risk:</strong> R<sub>train</sub>(f) = E<sub>(x,y)~P<sub>train</sub></sub>[L(f(x), y)]</li>
    <li><strong>Robust Risk:</strong> R<sub>worst</sub>(f) = max<sub>P∈U</sub> E<sub>(x,y)~P</sub>[L(f(x), y)]</li>
    </ul>
    <p>Where U is a set of allowed perturbations around P<sub>train</sub></p>
    
    <h4>Robustness vs Accuracy Tradeoff</h4>
    <p>Often, maximizing accuracy on clean data conflicts with robustness to perturbations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization 1: Robustness curves
    st.markdown("### 📊 Visualization: Model Robustness Comparison")
    st.markdown("Compare how different model architectures handle increasing drift")
    
    @st.cache_data
    def generate_robustness_curves():
        severities = np.linspace(0, 1, 30)
        
        # Different model robustness profiles
        # Standard model: high accuracy, low robustness
        acc_standard = 0.96 * np.exp(-4 * severities) + 0.52 * (1 - np.exp(-4 * severities))
        
        # Robust model: slightly lower accuracy, much better under drift
        acc_robust = 0.93 * np.exp(-1.5 * severities) + 0.65 * (1 - np.exp(-1.5 * severities))
        
        # Augmented model: middle ground
        acc_augmented = 0.94 * np.exp(-2.5 * severities) + 0.58 * (1 - np.exp(-2.5 * severities))
        
        return severities, acc_standard, acc_robust, acc_augmented
    
    severities, acc_std, acc_rob, acc_aug = generate_robustness_curves()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=severities * 100,
        y=acc_std * 100,
        mode='lines+markers',
        name='Standard Model',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=severities * 100,
        y=acc_rob * 100,
        mode='lines+markers',
        name='Robust Model',
        line=dict(color=COLORS['success'], width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=severities * 100,
        y=acc_aug * 100,
        mode='lines+markers',
        name='Augmented Model',
        line=dict(color=COLORS['secondary'], width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        height=500,
        title="Robustness Curves: Model Comparison",
        xaxis=dict(title="Drift Severity (%)"),
        yaxis=dict(title="Accuracy (%)"),
        font=dict(color=COLORS['text'], size=14),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    ℹ️ **Key Insight**: The robust model starts slightly lower but maintains performance much better under drift.
    This is the **accuracy-robustness tradeoff**.
    """)
    
    # Robustness strategies
    st.markdown("### 🛠️ Building Robust Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 1. Data Augmentation
        
        Train on artificially perturbed data:
        
        ```python
        augmentations = [
            RandomBrightness(0.2),
            RandomContrast(0.3),
            GaussianBlur(sigma=0.5),
            RandomNoise(std=0.1)
        ]
        ```
        
        **Benefits:**
        - ✅ Exposes model to variations
        - ✅ Improves generalization
        - ✅ Low implementation cost
        
        **Limitations:**
        - ❌ Can't cover all drifts
        - ❌ May hurt clean accuracy
        """)
        
        st.markdown("""
        #### 2. Robust Training
        
        Optimize for worst-case performance:
        
        ```python
        # Adversarial training
        for x, y in data:
            x_adv = generate_adversarial(x)
            loss = max(L(f(x), y), L(f(x_adv), y))
            optimize(loss)
        ```
        
        **Benefits:**
        - ✅ Theoretically grounded
        - ✅ Strong robustness guarantees
        
        **Limitations:**
        - ❌ Computationally expensive
        - ❌ Accuracy-robustness tradeoff
        """)
    
    with col2:
        st.markdown("""
        #### 3. Ensemble Methods
        
        Combine multiple models:
        
        ```python
        models = [
            ModelA(),  # High accuracy
            ModelB(),  # Robust to blur
            ModelC()   # Robust to noise
        ]
        prediction = ensemble(models, x)
        ```
        
        **Benefits:**
        - ✅ Leverages diverse strengths
        - ✅ Often best performance
        
        **Limitations:**
        - ❌ Higher inference cost
        - ❌ More complex deployment
        """)
        
        st.markdown("""
        #### 4. Architecture Design
        
        Build robustness into model structure:
        
        - **Normalization layers**: BatchNorm, LayerNorm
        - **Residual connections**: Skip connections
        - **Attention mechanisms**: Focus on robust features
        - **Multi-scale features**: Different receptive fields
        
        **Benefits:**
        - ✅ No training overhead
        - ✅ Architectural prior
        
        **Limitations:**
        - ❌ Design complexity
        - ❌ Domain-specific
        """)
    
    st.markdown("---")
    
    # Robustness metrics
    st.markdown("### 📏 Measuring Robustness")
    
    st.markdown("""
    #### Key Metrics:
    
    1. **Effective Robustness**: Accuracy under perturbations - baseline accuracy
    2. **Robustness Curve**: Plot accuracy vs perturbation strength
    3. **Area Under Curve (AUC)**: Integral of robustness curve
    4. **Worst-Case Accuracy**: Minimum accuracy across perturbation set
    5. **Relative Performance**: (Accuracy_drift / Accuracy_clean) × 100%
    """)
    
    # Example calculation
    st.markdown("#### Example Calculation")
    
    metrics_data = {
        'Model': ['Standard CNN', 'Augmented CNN', 'Robust Transformer'],
        'Clean Accuracy': [95.2, 94.1, 93.5],
        'Drift Accuracy (50%)': [73.8, 82.3, 87.1],
        'Effective Robustness': [-21.4, -11.8, -6.4],
        'Relative Performance': [77.5, 87.5, 93.2]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    st.success("""
    ✅ **Winner**: Robust Transformer
    - Only 6.4% degradation vs 21.4% for standard model
    - 93.2% relative performance maintained
    - Small sacrifice in clean accuracy (1.7%) for major robustness gain
    """)
    
    # Quiz
    st.markdown("### ✅ Knowledge Check")
    
    quiz_robust = st.radio(
        "**Question**: What is the primary tradeoff in robust model design?",
        [
            "A) Training time vs inference speed",
            "B) Clean accuracy vs robustness to drift",
            "C) Model size vs performance",
            "D) Interpretability vs accuracy"
        ],
        key='quiz_robust'
    )
    
    if st.button("Submit", key='submit_robust'):
        if 'B)' in quiz_robust:
            st.success("✅ **Correct!** Robust models often sacrifice clean accuracy for better performance under drift.")
            if st.session_state.tutorial_progress < 4:
                st.session_state.tutorial_progress = 4
                st.balloons()
        else:
            st.error("❌ Think about what we optimize: clean performance vs worst-case performance.")

# Module 5: Detection Methods (Enhanced)
elif current_idx == 4:
    st.markdown("## 🔍 Module 5: Detection Methods")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Methods", "5")
    with col2:
        st.metric("Visualizations", "4")
    with col3:
        st.metric("Duration", "20 min")
    
    st.markdown("---")
    
    # Theory section
    st.markdown("""
    <div class="theory-box">
    <h3>📚 Theory: Statistical Drift Detection</h3>
    
    <p><strong>Core Principle:</strong> Compare probability distributions P<sub>train</sub> and P<sub>prod</sub> using statistical tests.</p>
    
    <h4>Detection Framework</h4>
    <ol>
    <li><strong>Hypothesis Testing:</strong> H<sub>0</sub>: P<sub>train</sub> = P<sub>prod</sub> vs H<sub>1</sub>: P<sub>train</sub> ≠ P<sub>prod</sub></li>
    <li><strong>Test Statistic:</strong> Compute distance/divergence between distributions</li>
    <li><strong>Decision Rule:</strong> Reject H<sub>0</sub> if statistic exceeds threshold</li>
    </ol>
    
    <h4>Method Categories</h4>
    <ul>
    <li><strong>Parametric:</strong> Assume distribution form (e.g., chi-squared test)</li>
    <li><strong>Non-parametric:</strong> Distribution-free (e.g., KS test, MMD)</li>
    <li><strong>Distance-based:</strong> Measure distribution divergence (e.g., Wasserstein, PSI)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Method comparison with visualizations
    st.markdown("### 📊 Detection Methods: Deep Dive")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "PSI", "KS Test", "Chi-Squared", "MMD", "Wasserstein"
    ])
    
    with tab1:
        st.markdown("#### Population Stability Index (PSI)")
        
        col_theory, col_viz = st.columns([1, 1])
        
        with col_theory:
            st.markdown("""
            **Mathematical Formula:**
            ```
            PSI = Σ (P_prod,i - P_train,i) × ln(P_prod,i / P_train,i)
            ```
            
            **How it works:**
            1. Bin continuous features into intervals
            2. Calculate proportion of samples in each bin
            3. Compute divergence between proportions
            
            **Thresholds:**
            - PSI < 0.1: No significant drift ✅
            - 0.1 ≤ PSI < 0.2: Moderate drift ⚠️
            - PSI ≥ 0.2: Significant drift 🚨
            
            **Best for:**
            - Categorical/binned features
            - Quick monitoring dashboards
            - Industry standard metric
            """)
        
        with col_viz:
            # Generate PSI visualization
            @st.cache_data
            def generate_psi_viz():
                bins = np.linspace(-3, 3, 10)
                train_dist = np.random.normal(0, 1, 1000)
                prod_dist = np.random.normal(0.5, 1.2, 1000)
                
                train_hist, _ = np.histogram(train_dist, bins=bins)
                prod_hist, _ = np.histogram(prod_dist, bins=bins)
                
                train_prop = train_hist / train_hist.sum()
                prod_prop = prod_hist / prod_hist.sum()
                
                return bins, train_prop, prod_prop
            
            bins, train_prop, prod_prop = generate_psi_viz()
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=bin_centers, y=train_prop, name='Training',
                marker_color=COLORS['success'], opacity=0.7
            ))
            fig.add_trace(go.Bar(
                x=bin_centers, y=prod_prop, name='Production',
                marker_color=COLORS['danger'], opacity=0.7
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['surface'],
                height=400,
                title="PSI: Distribution Comparison",
                xaxis_title="Feature Value (Binned)",
                yaxis_title="Proportion",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate PSI
            psi = np.sum((prod_prop - train_prop) * np.log((prod_prop + 1e-10) / (train_prop + 1e-10)))
            st.metric("PSI Score", f"{psi:.3f}", delta="Moderate Drift" if psi > 0.1 else "No Drift")
    
    with tab2:
        st.markdown("#### Kolmogorov-Smirnov (KS) Test")
        
        col_theory, col_viz = st.columns([1, 1])
        
        with col_theory:
            st.markdown("""
            **Mathematical Formula:**
            ```
            D = max|F_train(x) - F_prod(x)|
            ```
            Where F is the cumulative distribution function (CDF)
            
            **How it works:**
            1. Compute empirical CDFs for both distributions
            2. Find maximum vertical distance between CDFs
            3. Test if distance is statistically significant
            
            **Interpretation:**
            - p-value < 0.05: Reject null hypothesis (drift detected)
            - D statistic: Magnitude of distribution shift
            
            **Best for:**
            - Continuous univariate features
            - Non-parametric testing
            - No distribution assumptions needed
            
            **Limitations:**
            - Univariate only
            - Sensitive to sample size
            - Less powerful for tail differences
            """)
        
        with col_viz:
            # Generate KS visualization
            @st.cache_data
            def generate_ks_viz():
                x = np.linspace(-4, 6, 200)
                train_cdf = np.array([np.mean(np.random.normal(0, 1, 1000) <= xi) for xi in x])
                prod_cdf = np.array([np.mean(np.random.normal(1, 1.3, 1000) <= xi) for xi in x])
                return x, train_cdf, prod_cdf
            
            x, train_cdf, prod_cdf = generate_ks_viz()
            ks_stat = np.max(np.abs(train_cdf - prod_cdf))
            ks_idx = np.argmax(np.abs(train_cdf - prod_cdf))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=train_cdf, mode='lines',
                name='Training CDF', line=dict(color=COLORS['success'], width=3)
            ))
            fig.add_trace(go.Scatter(
                x=x, y=prod_cdf, mode='lines',
                name='Production CDF', line=dict(color=COLORS['danger'], width=3)
            ))
            # Add KS statistic line
            fig.add_shape(
                type="line",
                x0=x[ks_idx], y0=train_cdf[ks_idx],
                x1=x[ks_idx], y1=prod_cdf[ks_idx],
                line=dict(color=COLORS['warning'], width=3, dash="dash")
            )
            fig.add_annotation(
                x=x[ks_idx], y=(train_cdf[ks_idx] + prod_cdf[ks_idx])/2,
                text=f"KS = {ks_stat:.3f}",
                showarrow=True, arrowhead=2
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['surface'],
                height=400,
                title="KS Test: CDF Comparison",
                xaxis_title="Feature Value",
                yaxis_title="Cumulative Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Chi-Squared (χ²) Test")
        
        st.markdown("""
        **Mathematical Formula:**
        ```
        χ² = Σ (O_i - E_i)² / E_i
        ```
        Where O = observed, E = expected frequencies
        
        **How it works:**
        1. Create contingency table of observed vs expected frequencies
        2. Calculate chi-squared statistic
        3. Compare to chi-squared distribution with df = (bins - 1)
        
        **Best for:**
        - Categorical features
        - Count data
        - Goodness-of-fit testing
        
        **Example: Medical Imaging Quality**
        """)
        
        # Chi-squared example
        quality_data = {
            'Quality Level': ['Excellent', 'Good', 'Fair', 'Poor'],
            'Training (%)': [40, 35, 20, 5],
            'Production (%)': [25, 30, 30, 15],
            'χ² Contribution': [5.63, 0.71, 5.00, 20.00]
        }
        
        st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)
        
        st.info("""
        ℹ️ **Interpretation**: χ² = 31.34, p < 0.001
        - Strong evidence of drift in quality distribution
        - Largest contributions from "Excellent" and "Poor" categories
        - Production data shifted toward lower quality
        """)
    
    with tab4:
        st.markdown("#### Maximum Mean Discrepancy (MMD)")
        
        col_theory, col_viz = st.columns([1, 1])
        
        with col_theory:
            st.markdown("""
            **Mathematical Formula:**
            ```
            MMD²(P, Q) = ||μ_P - μ_Q||²_H
            ```
            Where μ is mean embedding in reproducing kernel Hilbert space
            
            **Kernel Formulation:**
            ```
            MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
            ```
            
            **How it works:**
            1. Map distributions to high-dimensional space using kernel
            2. Compute distance between mean embeddings
            3. Test if distance exceeds threshold
            
            **Advantages:**
            - ✅ Multivariate (handles high dimensions)
            - ✅ Powerful for complex shifts
            - ✅ Kernel trick enables non-linear detection
            
            **Kernel Choices:**
            - RBF (Gaussian): k(x,y) = exp(-γ||x-y||²)
            - Linear: k(x,y) = x·y
            - Polynomial: k(x,y) = (x·y + c)^d
            """)
        
        with col_viz:
            # MMD visualization with embeddings
            st.markdown("**Kernel Space Visualization**")
            
            @st.cache_data
            def generate_mmd_viz():
                np.random.seed(42)
                # 2D for visualization
                train_2d = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 100)
                prod_2d = np.random.multivariate_normal([1.5, 1], [[1.2, -0.2], [-0.2, 1.2]], 100)
                return train_2d, prod_2d
            
            train_2d, prod_2d = generate_mmd_viz()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_2d[:, 0], y=train_2d[:, 1],
                mode='markers', name='Training',
                marker=dict(color=COLORS['success'], size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=prod_2d[:, 0], y=prod_2d[:, 1],
                mode='markers', name='Production',
                marker=dict(color=COLORS['danger'], size=6, opacity=0.6)
            ))
            # Add centroids
            fig.add_trace(go.Scatter(
                x=[train_2d[:, 0].mean()], y=[train_2d[:, 1].mean()],
                mode='markers', name='Train μ',
                marker=dict(color=COLORS['success'], size=20, symbol='x', line_width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[prod_2d[:, 0].mean()], y=[prod_2d[:, 1].mean()],
                mode='markers', name='Prod μ',
                marker=dict(color=COLORS['danger'], size=20, symbol='x', line_width=3)
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['surface'],
                height=400,
                title="MMD: Mean Embedding Distance",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate simple MMD
            mmd_est = np.linalg.norm(train_2d.mean(axis=0) - prod_2d.mean(axis=0))
            st.metric("MMD Estimate", f"{mmd_est:.3f}", delta="Drift Detected")
    
    with tab5:
        st.markdown("#### Wasserstein Distance (Earth Mover's)")
        
        st.markdown("""
        **Mathematical Formula:**
        ```
        W(P, Q) = inf_{γ} E[(X,Y)~γ][||X - Y||]
        ```
        Minimum "cost" to transform one distribution into another
        
        **Intuition:**
        - Think of distributions as piles of dirt
        - Compute minimum work to reshape one pile into another
        - "Earth Mover's Distance"
        
        **Properties:**
        - ✅ True metric (satisfies triangle inequality)
        - ✅ Robust to outliers
        - ✅ Interpretable units (same as data)
        - ❌ Computationally expensive for high dimensions
        
        **Applications in Medical AI:**
        - Comparing patient demographics
        - Image quality distribution shifts
        - Feature distribution monitoring
        """)
        
        # Wasserstein comparison
        comparison_data = {
            'Method': ['PSI', 'KS Test', 'Chi-Squared', 'MMD', 'Wasserstein'],
            'Univariate': ['✅', '✅', '✅', '❌', '✅'],
            'Multivariate': ['❌', '❌', '❌', '✅', '✅'],
            'Speed': ['Fast', 'Fast', 'Fast', 'Medium', 'Slow'],
            'Interpretability': ['High', 'High', 'High', 'Low', 'Medium'],
            'Best Use Case': ['Monitoring', 'Continuous', 'Categorical', 'Complex', 'Robust']
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Interactive detection demo with multiple methods
    st.markdown("### ▶️ Comprehensive Detection Demo")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        demo_dataset = st.selectbox("Dataset", ['pathmnist', 'dermamnist', 'retinamnist', 'pneumoniamnist'], index=0)
        demo_drift = st.selectbox("Drift Type", ['blur', 'brightness_shift', 'contrast_reduction', 'noise'], index=0)
    with col_config2:
        demo_severity = st.slider("Drift Severity", 0.0, 1.0, 0.6, 0.1)
        demo_samples = st.slider("Sample Size", 100, 500, 200, 50)
    
    if st.button("🔍 Run Detection Analysis", type="primary", use_container_width=True):
        with st.spinner("Running statistical tests..."):
            # Load data
            loader = MedMNISTLoader(demo_dataset)
            reference, _ = loader.get_numpy_data('test', limit=demo_samples, offset=0)
            
            # Apply drift
            generator = DriftGenerator(seed=42)
            apply_method = getattr(generator, f'_apply_{demo_drift}')
            drifted = apply_method(reference.copy(), demo_severity)
            
            # Flatten images for detection (N, H, W, C) -> (N, H*W*C)
            ref_shape = reference.shape
            reference_flat = reference.reshape(ref_shape[0], -1)
            drifted_flat = drifted.reshape(ref_shape[0], -1)
            
            # Run all detection methods
            detector = DriftDetector()
            
            st.markdown("#### Detection Results")
            
            results_list = []
            for method in ['psi', 'ks', 'chi2', 'mmd', 'wasserstein']:
                try:
                    result = detector.detect_drift(reference_flat, drifted_flat, method=method)
                    detected = result.get('drift_detected', False)
                    method_result = result['methods'].get(method, {})
                    
                    # Extract appropriate statistic based on actual return structure
                    if method == 'psi':
                        # PSI returns 'scores' as array
                        scores = method_result.get('scores', [])
                        score = np.mean(scores) if len(scores) > 0 else 0.0
                    elif method == 'ks':
                        # KS returns 'statistics' as array
                        statistics = method_result.get('statistics', [])
                        score = np.mean(statistics) if len(statistics) > 0 else 0.0
                    elif method == 'chi2':
                        # Chi2 returns 'statistics' as array
                        statistics = method_result.get('statistics', [])
                        score = np.mean(statistics) if len(statistics) > 0 else 0.0
                    elif method == 'mmd':
                        # MMD returns 'score' as single value
                        score = method_result.get('score', 0.0)
                    elif method == 'wasserstein':
                        # Wasserstein returns 'scores' as array
                        scores = method_result.get('scores', [])
                        score = np.mean(scores) if len(scores) > 0 else 0.0
                    else:
                        score = 0.0
                    
                    # Determine confidence based on method-specific thresholds
                    if method == 'psi':
                        confidence = 'High' if score > 0.2 else 'Medium' if score > 0.1 else 'Low'
                    elif method in ['ks', 'chi2']:
                        confidence = 'High' if score > 0.3 else 'Medium' if score > 0.15 else 'Low'
                    elif method == 'mmd':
                        confidence = 'High' if score > 0.1 else 'Medium' if score > 0.05 else 'Low'
                    else:  # wasserstein
                        confidence = 'High' if score > 10 else 'Medium' if score > 5 else 'Low'
                    
                    results_list.append({
                        'Method': method.upper(),
                        'Statistic': f"{score:.4f}",
                        'Detected': '🔴 Yes' if detected else '🟢 No',
                        'Confidence': confidence
                    })
                except Exception as e:
                    # Some methods may fail on high-dimensional data
                    results_list.append({
                        'Method': method.upper(),
                        'Statistic': 'N/A',
                        'Detected': '⚠️ Skip',
                        'Confidence': 'N/A'
                    })
            
            st.dataframe(pd.DataFrame(results_list), use_container_width=True, hide_index=True)
            
            st.info("""
            ℹ️ **Note on Methods**:
            - **PSI**: Fast, works on flattened features
            - **KS/Wasserstein**: May skip on very high-dimensional data
            - **Chi-Squared**: Best for categorical/binned data
            - **MMD**: Robust for multivariate detection
            """)
            
            # Consensus (only count methods that ran successfully)
            valid_results = [r for r in results_list if r['Detected'] not in ['⚠️ Skip', 'N/A']]
            detected_count = sum([1 for r in valid_results if '🔴' in r['Detected']])
            total_methods = len(valid_results)
            
            if total_methods > 0:
                st.markdown(f"**Consensus**: {detected_count}/{total_methods} methods detected drift")
                
                if detected_count >= (total_methods * 0.6):  # 60% or more
                    st.error("🚨 **Strong evidence of drift** - Majority of methods agree")
                elif detected_count >= 1:
                    st.warning("⚠️ **Possible drift** - Some methods detected changes")
                else:
                    st.success("✅ **No drift detected** - Distributions appear similar")
            else:
                st.warning("⚠️ Unable to run detection methods on this configuration")
            
            if st.session_state.tutorial_progress < 5:
                st.session_state.tutorial_progress = 5
                st.balloons()
    
    # Quiz
    st.markdown("---")
    st.markdown("### ✅ Knowledge Check")
    
    quiz5 = st.radio(
        "**Question**: Which detection method is best for multivariate drift in high-dimensional medical images?",
        [
            "A) PSI - Fast and interpretable",
            "B) KS Test - Non-parametric",
            "C) Chi-Squared - Good for counts",
            "D) MMD - Handles high dimensions with kernel trick"
        ],
        key='quiz5'
    )
    
    if st.button("Submit Answer", key='submit5'):
        if 'D)' in quiz5:
            st.success("✅ **Correct!** MMD with kernels can detect complex multivariate drift in high-dimensional spaces like medical images.")
            if st.session_state.tutorial_progress < 5:
                st.session_state.tutorial_progress = 5
        else:
            st.error("❌ Consider: Medical images are high-dimensional. Which method handles multiple dimensions?")

# Module 6: Monitoring Strategies
elif current_idx == 5:
    st.markdown("## 📈 Module 6: Monitoring Strategies")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Strategies", "4")
    with col2:
        st.metric("Visualizations", "3")
    with col3:
        st.metric("Duration", "18 min")
    
    st.markdown("---")
    
    # Theory
    st.markdown("""
    <div class="theory-box">
    <h3>📚 Theory: Production Monitoring Framework</h3>
    
    <p><strong>Monitoring Lifecycle:</strong> Continuous observation → Detection → Analysis → Mitigation</p>
    
    <h4>Key Principles</h4>
    <ol>
    <li><strong>Continuous Monitoring:</strong> Track metrics in real-time or batch</li>
    <li><strong>Multi-level Detection:</strong> Data, model, performance drift</li>
    <li><strong>Alerting System:</strong> Trigger actions when thresholds exceeded</li>
    <li><strong>Root Cause Analysis:</strong> Identify source of drift</li>
    <li><strong>Mitigation Strategy:</strong> Retrain, recalibrate, or rollback</li>
    </ol>
    
    <h4>Monitoring Levels</h4>
    <ul>
    <li><strong>Input Monitoring:</strong> Track P(X) changes</li>
    <li><strong>Output Monitoring:</strong> Track P(Ŷ) changes</li>
    <li><strong>Performance Monitoring:</strong> Track accuracy, precision, recall</li>
    <li><strong>Business Monitoring:</strong> Track clinical outcomes, costs</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy 1: Sliding Window
    st.markdown("### 📊 Strategy 1: Sliding Window Monitoring")
    
    col_desc, col_viz = st.columns([1, 1])
    
    with col_desc:
        st.markdown("""
        **Concept:**
        - Maintain reference window (training data)
        - Compare against sliding production window
        - Update window as new data arrives
        
        **Parameters:**
        - Window size (e.g., 1000 samples)
        - Slide interval (e.g., 100 samples)
        - Detection frequency (e.g., daily)
        
        **Advantages:**
        - ✅ Adapts to gradual drift
        - ✅ Memory efficient
        - ✅ Low latency
        
        **Use Cases:**
        - Online learning systems
        - Real-time monitoring
        - Gradual distribution shifts
        """)
    
    with col_viz:
        # Sliding window visualization
        @st.cache_data
        def generate_sliding_window_viz():
            time_steps = np.arange(0, 100)
            # Simulate drift over time
            drift_signal = 0.85 - 0.003 * time_steps + 0.05 * np.sin(time_steps / 10)
            drift_signal += np.random.normal(0, 0.02, len(time_steps))
            return time_steps, drift_signal
        
        time_steps, signal = generate_sliding_window_viz()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_steps, y=signal,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=4)
        ))
        
        # Add window regions
        window_start = 40
        window_end = 60
        fig.add_vrect(
            x0=window_start, x1=window_end,
            fillcolor=COLORS['success'], opacity=0.2,
            annotation_text="Current Window", annotation_position="top left"
        )
        
        # Threshold line
        fig.add_hline(
            y=0.80, line_dash="dash", line_color=COLORS['warning'],
            annotation_text="Alert Threshold (80%)"
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=400,
            title="Sliding Window: Performance Over Time",
            xaxis_title="Time (days)",
            yaxis_title="Model Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Strategy 2: Statistical Process Control
    st.markdown("### 📊 Strategy 2: Statistical Process Control (SPC)")
    
    col_desc, col_viz = st.columns([1, 1])
    
    with col_desc:
        st.markdown("""
        **Concept:**
        - Borrow from manufacturing quality control
        - Control charts with control limits
        - Detect out-of-control conditions
        
        **Control Chart Types:**
        - **X-bar Chart**: Monitor mean
        - **R Chart**: Monitor range/variance
        - **CUSUM**: Cumulative sum of deviations
        - **EWMA**: Exponentially weighted moving average
        
        **Control Limits:**
        ```
        UCL = μ + 3σ (Upper Control Limit)
        LCL = μ - 3σ (Lower Control Limit)
        ```
        
        **Detection Rules:**
        1. Point beyond control limits
        2. 8 consecutive points on same side
        3. 2 of 3 points beyond 2σ
        4. 4 of 5 points beyond 1σ
        """)
    
    with col_viz:
        # SPC chart
        @st.cache_data
        def generate_spc_chart():
            np.random.seed(42)
            n = 50
            # Stable period then drift
            data = np.concatenate([
                np.random.normal(0.90, 0.02, 30),
                np.random.normal(0.85, 0.03, 20)
            ])
            return np.arange(n), data
        
        time, data = generate_spc_chart()
        mean = 0.90
        std = 0.02
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=data,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6)
        ))
        
        # Control limits
        fig.add_hline(y=mean, line_dash="solid", line_color=COLORS['text'], 
                     annotation_text="Center Line (μ)")
        fig.add_hline(y=mean + 3*std, line_dash="dash", line_color=COLORS['danger'],
                     annotation_text="UCL (μ+3σ)")
        fig.add_hline(y=mean - 3*std, line_dash="dash", line_color=COLORS['danger'],
                     annotation_text="LCL (μ-3σ)")
        fig.add_hline(y=mean + 2*std, line_dash="dot", line_color=COLORS['warning'])
        fig.add_hline(y=mean - 2*std, line_dash="dot", line_color=COLORS['warning'])
        
        # Highlight out-of-control points
        out_of_control = data < (mean - 3*std)
        fig.add_trace(go.Scatter(
            x=time[out_of_control], y=data[out_of_control],
            mode='markers',
            name='Out of Control',
            marker=dict(color=COLORS['danger'], size=12, symbol='x', line_width=2)
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=400,
            title="SPC Chart: Control Limits",
            xaxis_title="Sample Number",
            yaxis_title="Model Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Strategy 3: Multi-metric Dashboard
    st.markdown("### 📊 Strategy 3: Multi-Metric Dashboard")
    
    st.markdown("""
    **Comprehensive Monitoring:**
    Track multiple dimensions simultaneously for complete picture
    """)
    
    # Simulated dashboard metrics
    @st.cache_data
    def generate_dashboard_metrics():
        return {
            'Model Performance': {
                'Accuracy': (0.87, 0.95, -0.08),
                'Precision': (0.84, 0.92, -0.08),
                'Recall': (0.89, 0.94, -0.05),
                'F1-Score': (0.86, 0.93, -0.07)
            },
            'Data Quality': {
                'Missing %': (2.3, 0.5, +1.8),
                'Outliers %': (5.1, 2.0, +3.1),
                'Mean Pixel Intensity': (127, 145, -18),
                'Image Blur Score': (0.72, 0.35, +0.37)
            },
            'Drift Scores': {
                'PSI': (0.24, 0.05, +0.19),
                'KS Statistic': (0.18, 0.02, +0.16),
                'MMD': (0.31, 0.08, +0.23),
                'Wasserstein': (12.5, 3.2, +9.3)
            },
            'Business Metrics': {
                'Throughput (imgs/hr)': (450, 500, -50),
                'Avg Confidence': (0.82, 0.91, -0.09),
                'Alert Rate %': (15.2, 3.5, +11.7),
                'Manual Review %': (22.0, 8.0, +14.0)
            }
        }
    
    metrics = generate_dashboard_metrics()
    
    tab_perf, tab_quality, tab_drift, tab_business = st.tabs([
        "📈 Performance", "📋 Data Quality", "🔍 Drift Detection", "💼 Business"
    ])
    
    for tab, (category, values) in zip(
        [tab_perf, tab_quality, tab_drift, tab_business],
        metrics.items()
    ):
        with tab:
            cols = st.columns(4)
            for idx, (metric_name, (current, baseline, delta)) in enumerate(values.items()):
                with cols[idx % 4]:
                    delta_str = f"{delta:+.2f}" if abs(delta) < 100 else f"{delta:+.0f}"
                    # Determine if delta is bad (depends on metric)
                    is_bad = (delta < 0 and 'Accuracy' in metric_name or 'Precision' in metric_name or 
                             'Recall' in metric_name or 'F1' in metric_name or 'Throughput' in metric_name or
                             'Confidence' in metric_name) or \
                            (delta > 0 and ('Missing' in metric_name or 'Outliers' in metric_name or 
                             'Blur' in metric_name or 'Alert' in metric_name or 'Review' in metric_name or
                             'PSI' in metric_name or 'KS' in metric_name or 'MMD' in metric_name or
                             'Wasserstein' in metric_name))
                    
                    st.metric(
                        metric_name,
                        f"{current:.2f}" if abs(current) < 100 else f"{current:.0f}",
                        delta=delta_str,
                        delta_color="inverse" if is_bad else "normal"
                    )
    
    st.info("""
    ℹ️ **Dashboard Interpretation**:
    - Performance metrics declining 5-8%
    - Data quality degraded (more outliers, increased blur)
    - All drift detectors flagging significant changes
    - Business impact: increased manual review, lower confidence
    
    **Recommended Action**: Investigate image acquisition pipeline and consider model retraining
    """)
    
    st.markdown("---")
    
    # Strategy 4: Alerting & Response
    st.markdown("### 📊 Strategy 4: Alerting & Response Workflow")
    
    st.markdown("""
    **Alert Severity Levels:**
    """)
    
    alert_levels = {
        'Level': ['🟢 Info', '🟡 Warning', '🟠 Critical', '🔴 Emergency'],
        'Condition': [
            'Minor deviation (1-2σ)',
            'Moderate drift (2-3σ)',
            'Severe drift (>3σ)',
            'Model failure (>50% degradation)'
        ],
        'Response': [
            'Log and monitor',
            'Notify team, increase monitoring',
            'Trigger investigation, prepare mitigation',
            'Immediate action: rollback or disable'
        ],
        'Timeline': ['24-48 hours', '4-12 hours', '1-4 hours', 'Immediate']
    }
    
    st.dataframe(pd.DataFrame(alert_levels), use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Response Playbook:**
    
    1. **Detection Phase**
       - Automated monitoring detects anomaly
       - Alert triggered based on severity
       - Capture diagnostic data snapshot
    
    2. **Analysis Phase**
       - Investigate root cause (data, model, infrastructure)
       - Compare against historical patterns
       - Assess scope and impact
    
    3. **Decision Phase**
       - Evaluate mitigation options:
         * Continue monitoring (false alarm)
         * Retrain model with new data
         * Adjust preprocessing/normalization
         * Rollback to previous version
         * Disable problematic features
    
    4. **Action Phase**
       - Implement chosen mitigation
       - Monitor effectiveness
       - Update runbooks and thresholds
    
    5. **Post-Mortem**
       - Document incident and resolution
       - Update monitoring strategy
       - Prevent future occurrences
    """)
    
    # Quiz
    st.markdown("---")
    st.markdown("### ✅ Knowledge Check")
    
    quiz6 = st.radio(
        "**Question**: Your medical AI shows 15% accuracy drop and PSI > 0.3. What's the appropriate response?",
        [
            "A) 🟢 Log and continue monitoring",
            "B) 🟡 Notify team and increase monitoring frequency",
            "C) 🟠 Trigger investigation and prepare mitigation plan",
            "D) 🔴 Immediately disable model and rollback"
        ],
        key='quiz6'
    )
    
    if st.button("Submit Answer", key='submit6'):
        if 'C)' in quiz6:
            st.success("✅ **Correct!** 15% drop with high PSI is Critical level - requires investigation and mitigation planning.")
            if st.session_state.tutorial_progress < 6:
                st.session_state.tutorial_progress = 6
                st.balloons()
        else:
            st.error("❌ Consider the severity: 15% accuracy drop with significant drift (PSI > 0.3) is serious but not catastrophic.")

# Module 7: Hands-On Exercise
elif current_idx == 6:
    st.markdown("## ▶️ Module 7: Hands-On Drift Detection Challenge")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tasks", "5")
    with col2:
        st.metric("Difficulty", "⭐⭐⭐")
    with col3:
        st.metric("Duration", "25 min")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Challenge: Medical Imaging Drift Investigation
    
    **Scenario:** You're monitoring a dermatology AI model deployed in multiple clinics. 
    Performance has degraded and you need to diagnose and address the drift.
    
    **Your Mission:**
    1. Analyze drift patterns across different degradation types
    2. Identify which detection method is most sensitive
    3. Quantify the impact on model performance
    4. Recommend mitigation strategies
    """)
    
    # Initialize session state for exercise
    if 'exercise_completed' not in st.session_state:
        st.session_state.exercise_completed = []
    
    # Task 1: Data Analysis
    st.markdown("---")
    st.markdown("### 📋 Task 1: Compare Multiple Drift Types")
    
    st.markdown("""
    Run detection on 3 different degradation types and compare results.
    
    **Instructions:**
    1. Select a degradation type
    2. Choose severity level
    3. Run detection
    4. Record results
    5. Repeat for 2 more degradation types
    """)
    
    col_task1_config, col_task1_results = st.columns([1, 2])
    
    with col_task1_config:
        task1_degradation = st.selectbox(
            "Degradation Type",
            ['blur', 'brightness_shift', 'contrast_reduction', 'noise', 'jpeg_compression'],
            key='task1_deg'
        )
        task1_severity = st.slider("Severity", 0.0, 1.0, 0.5, key='task1_sev')
        
        if st.button("Run Detection", key='task1_run'):
            st.session_state.task1_running = True
    
    with col_task1_results:
        if st.session_state.get('task1_running', False):
            with st.spinner("Analyzing..."):
                loader = MedMNISTLoader('dermamnist')
                reference, _ = loader.get_numpy_data('test', limit=200, offset=0)
                
                generator = DriftGenerator(seed=42)
                apply_method = getattr(generator, f'_apply_{task1_degradation}')
                drifted = apply_method(reference.copy(), task1_severity)
                
                # Flatten images for detection
                ref_shape = reference.shape
                reference_flat = reference.reshape(ref_shape[0], -1)
                drifted_flat = drifted.reshape(ref_shape[0], -1)
                
                detector = DriftDetector()
                
                # Run multiple methods
                results = []
                for method in ['psi', 'ks', 'mmd']:
                    try:
                        result = detector.detect_drift(reference_flat, drifted_flat, method=method)
                        detected = result.get('drift_detected', False)
                        method_result = result['methods'].get(method, {})
                        
                        if method == 'psi':
                            scores = method_result.get('scores', [])
                            score = np.mean(scores) if len(scores) > 0 else 0.0
                        elif method == 'ks':
                            statistics = method_result.get('statistics', [])
                            score = np.mean(statistics) if len(statistics) > 0 else 0.0
                        else:
                            score = method_result.get('score', 0.0)
                        
                        results.append({
                            'Method': method.upper(),
                            'Score': f"{score:.4f}",
                            'Detected': '✅' if detected else '❌'
                        })
                    except:
                        pass
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
                # Add to history
                if 'task1_history' not in st.session_state:
                    st.session_state.task1_history = []
                
                st.session_state.task1_history.append({
                    'Type': task1_degradation,
                    'Severity': task1_severity,
                    'Detections': sum([1 for r in results if r['Detected'] == '✅'])
                })
                
                if len(st.session_state.task1_history) >= 3 and 1 not in st.session_state.exercise_completed:
                    st.session_state.exercise_completed.append(1)
                    st.success("✅ Task 1 Complete! You've analyzed 3 degradation types.")
    
    # Show history
    if st.session_state.get('task1_history', []):
        st.markdown("**Your Analysis History:**")
        st.dataframe(pd.DataFrame(st.session_state.task1_history), use_container_width=True, hide_index=True)
    
    # Task 2: Method Sensitivity
    st.markdown("---")
    st.markdown("### 📋 Task 2: Identify Most Sensitive Detection Method")
    
    st.markdown("""
    Test detection sensitivity at different severity levels.
    
    **Question**: At what severity level does each method first detect drift?
    """)
    
    if st.button("Run Sensitivity Analysis", key='task2_run'):
        with st.spinner("Running analysis across severity levels..."):
            loader = MedMNISTLoader('dermamnist')
            reference, _ = loader.get_numpy_data('test', limit=150, offset=0)
            generator = DriftGenerator(seed=42)
            detector = DriftDetector()
            
            # Flatten reference once
            ref_shape = reference.shape
            reference_flat = reference.reshape(ref_shape[0], -1)
            
            severities = [0.2, 0.4, 0.6, 0.8]
            sensitivity_results = {'Severity': severities}
            
            for method in ['psi', 'ks', 'mmd']:
                detections = []
                for sev in severities:
                    drifted = generator._apply_blur(reference.copy(), sev)
                    drifted_flat = drifted.reshape(ref_shape[0], -1)
                    result = detector.detect_drift(reference_flat, drifted_flat, method=method)
                    detections.append('✅' if result.get('drift_detected', False) else '❌')
                sensitivity_results[method.upper()] = detections
            
            st.dataframe(pd.DataFrame(sensitivity_results), use_container_width=True, hide_index=True)
            
            st.info("""
            ℹ️ **Analysis**:
            - Methods have different sensitivity thresholds
            - Some detect earlier (more sensitive to subtle changes)
            - Trade-off between sensitivity and false positive rate
            """)
            
            if 2 not in st.session_state.exercise_completed:
                st.session_state.exercise_completed.append(2)
                st.success("✅ Task 2 Complete! You've analyzed detection sensitivity.")
    
    # Task 3: Performance Impact
    st.markdown("---")
    st.markdown("### 📋 Task 3: Quantify Performance Impact")
    
    st.markdown("""
    **Challenge**: Estimate accuracy degradation for different drift types.
    
    Use the Impact Estimator to predict performance drops.
    """)
    
    col_impact1, col_impact2 = st.columns(2)
    
    with col_impact1:
        impact_type = st.selectbox(
            "Degradation",
            ['blur', 'brightness_shift', 'contrast_reduction', 'noise'],
            key='impact_type'
        )
        impact_sev = st.slider("Severity", 0.0, 1.0, 0.5, key='impact_sev')
    
    with col_impact2:
        # Estimate impact
        impact_estimates = {
            'blur': (10, 20),
            'brightness_shift': (2, 5),
            'contrast_reduction': (5, 10),
            'noise': (5, 15)
        }
        
        min_imp, max_imp = impact_estimates[impact_type]
        estimated_drop = min_imp + (max_imp - min_imp) * impact_sev
        
        st.metric("Estimated Accuracy Drop", f"{estimated_drop:.1f}%", delta=f"-{estimated_drop:.1f}%", delta_color="inverse")
        st.metric("Projected Accuracy", f"{95.0 - estimated_drop:.1f}%", delta=f"from 95.0%")
        
        if estimated_drop > 10:
            st.error("🚨 Severe impact - Immediate action required")
        elif estimated_drop > 5:
            st.warning("⚠️ Moderate impact - Monitor closely")
        else:
            st.info("ℹ️ Minor impact - Continue monitoring")
    
    if st.button("Record Impact Assessment", key='task3_submit'):
        if 3 not in st.session_state.exercise_completed:
            st.session_state.exercise_completed.append(3)
            st.success("✅ Task 3 Complete! You've quantified performance impact.")
    
    # Task 4: Mitigation Strategy
    st.markdown("---")
    st.markdown("### 📋 Task 4: Develop Mitigation Strategy")
    
    st.markdown("""
    **Scenario**: You've detected blur drift with 18% accuracy drop.
    
    **Select the best mitigation approach:**
    """)
    
    task4_answer = st.radio(
        "What's your recommendation?",
        [
            "A) Retrain model with augmented data (blur transformations)",
            "B) Apply preprocessing sharpening filter to production data",
            "C) Adjust confidence threshold to reduce false positives",
            "D) Rollback to previous model version"
        ],
        key='task4_radio'
    )
    
    if st.button("Submit Strategy", key='task4_submit'):
        if 'A)' in task4_answer or 'B)' in task4_answer:
            st.success("""
            ✅ **Good choice!**
            - Option A: Long-term solution, makes model robust to blur
            - Option B: Short-term fix, addresses immediate issue
            
            Both are valid depending on timeline and resources.
            """)
            if 4 not in st.session_state.exercise_completed:
                st.session_state.exercise_completed.append(4)
        else:
            st.warning("""
            ⚠️ **Consider alternatives**:
            - Option C: Doesn't address root cause (blur)
            - Option D: Doesn't solve the drift problem
            
            Try options that address the blur directly.
            """)
    
    # Task 5: Final Challenge
    st.markdown("---")
    st.markdown("### 📋 Task 5: Comprehensive Analysis")
    
    st.markdown("""
    **Final Challenge**: Run a complete drift analysis pipeline.
    
    1. Select parameters
    2. Run detection
    3. Analyze results
    4. Make recommendation
    """)
    
    col_final1, col_final2 = st.columns(2)
    
    with col_final1:
        final_dataset = st.selectbox("Dataset", ['dermamnist', 'pathmnist', 'retinamnist', 'pneumoniamnist'], key='final_ds')
        final_drift = st.selectbox("Drift Type", ['blur', 'brightness_shift', 'noise'], key='final_drift')
        final_severity = st.slider("Severity", 0.0, 1.0, 0.7, key='final_sev')
    
    with col_final2:
        if st.button("🚀 Run Complete Analysis", type="primary", key='final_run'):
            with st.spinner("Running comprehensive analysis..."):
                loader = MedMNISTLoader(final_dataset)
                reference, _ = loader.get_numpy_data('test', limit=200, offset=0)
                
                generator = DriftGenerator(seed=42)
                apply_method = getattr(generator, f'_apply_{final_drift}')
                drifted = apply_method(reference.copy(), final_severity)
                
                # Flatten images for detection
                ref_shape = reference.shape
                reference_flat = reference.reshape(ref_shape[0], -1)
                drifted_flat = drifted.reshape(ref_shape[0], -1)
                
                detector = DriftDetector()
                
                # Run all methods
                st.markdown("**Detection Results:**")
                all_results = []
                for method in ['psi', 'ks', 'mmd']:
                    result = detector.detect_drift(reference_flat, drifted_flat, method=method)
                    detected = result.get('drift_detected', False)
                    all_results.append({'Method': method.upper(), 'Detected': '✅' if detected else '❌'})
                
                st.dataframe(pd.DataFrame(all_results), use_container_width=True, hide_index=True)
                
                detected_count = sum([1 for r in all_results if r['Detected'] == '✅'])
                
                st.markdown("**Impact Assessment:**")
                st.metric("Drift Severity", f"{final_severity*100:.0f}%")
                st.metric("Detection Consensus", f"{detected_count}/3 methods")
                
                st.markdown("**Recommendation:**")
                if detected_count >= 2 and final_severity > 0.5:
                    st.error("""
                    🚨 **Critical Drift Detected**
                    
                    **Immediate Actions:**
                    1. Increase monitoring frequency to hourly
                    2. Investigate data acquisition pipeline
                    3. Prepare model retraining with robust augmentation
                    4. Consider temporary confidence threshold adjustment
                    5. Alert clinical team of potential accuracy impact
                    """)
                elif detected_count >= 1:
                    st.warning("""
                    ⚠️ **Moderate Drift Detected**
                    
                    **Recommended Actions:**
                    1. Continue monitoring daily
                    2. Collect more production samples for analysis
                    3. Evaluate preprocessing adjustments
                    4. Schedule model performance review
                    """)
                else:
                    st.success("""
                    ✅ **No Significant Drift**
                    
                    **Maintain Current Strategy:**
                    1. Continue routine monitoring
                    2. Log metrics for trend analysis
                    3. No immediate action required
                    """)
                
                if 5 not in st.session_state.exercise_completed:
                    st.session_state.exercise_completed.append(5)
                    st.balloons()
                    st.success("✅ Task 5 Complete! You've finished the comprehensive analysis!")
    
    # Progress summary
    st.markdown("---")
    st.markdown("### 📊 Your Progress")
    
    progress_pct = len(st.session_state.exercise_completed) / 5 * 100
    st.progress(progress_pct / 100)
    st.markdown(f"**Completed: {len(st.session_state.exercise_completed)}/5 tasks** ({progress_pct:.0f}%)")
    
    if len(st.session_state.exercise_completed) >= 5:
        st.success("""
        🎉 **Congratulations!** You've completed all tutorial modules!
        
        **What you've learned:**
        - ✅ Mathematical foundations of drift
        - ✅ Different types of drift (data, concept, label)
        - ✅ Visual degradation patterns in medical imaging
        - ✅ Robustness theory and accuracy tradeoffs
        - ✅ Statistical detection methods (PSI, KS, MMD, etc.)
        - ✅ Production monitoring strategies
        - ✅ Hands-on drift detection and mitigation
        
        **Next Steps:**
        - Explore the Drift Lab for interactive experiments
        - Test models in the Model Zoo
        - Review Analytics for detailed performance analysis
        """)
        
        if st.session_state.tutorial_progress < 7:
            st.session_state.tutorial_progress = 7
    
    else:
        st.info(f"""
        📝 **Keep going!** Complete all {5 - len(st.session_state.exercise_completed)} remaining tasks to finish the tutorial.
        """)

else:
    st.markdown("## ⚠️ Module Not Found")
    st.error("This module is not available. Please select a valid module from the sidebar.")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if current_idx > 0:
        if st.button("⬅️ Previous Module", use_container_width=True):
            st.session_state.current_module -= 1
            st.rerun()

with col2:
    st.markdown(f"<div style='text-align: center; padding: 1rem;'><strong>Module {current_idx + 1} of {len(modules)}</strong></div>", 
                unsafe_allow_html=True)

with col3:
    if current_idx < len(modules) - 1:
        if st.button("Next Module ➡️", use_container_width=True):
            st.session_state.current_module += 1
            st.rerun()

# Footer
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
