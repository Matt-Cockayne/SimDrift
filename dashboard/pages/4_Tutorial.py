"""
Interactive Tutorial - Learn About Drift Detection

Step-by-step guided tutorial on ML drift concepts,
detection methods, and best practices.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.medmnist_loader import MedMNISTLoader
from data.drift_generators import DriftGenerator
from monitoring.drift_detectors import DriftDetector

st.set_page_config(
    page_title="SimDrift - Tutorial",
    page_icon="📊",
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
    .quiz-option {{
        background: {COLORS['surface']};
        border: 2px solid {COLORS['primary']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }}
    .quiz-option:hover {{
        background: {COLORS['primary']};
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tutorial_progress' not in st.session_state:
    st.session_state.tutorial_progress = 0
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}

st.title("Interactive Tutorial")
st.markdown("### Learn about ML drift detection through hands-on examples")
st.markdown("Step-by-step guide covering concepts, detection methods, and practical exercises")
st.markdown("---")

# Progress bar
progress = st.session_state.tutorial_progress / 6
st.progress(progress, text=f"Progress: {st.session_state.tutorial_progress}/6 modules completed")

# Tutorial modules
modules = [
    {
        'title': 'Module 1: What is Model Drift?',
        'icon': '[1]',
        'content': 'Introduction to drift concepts'
    },
    {
        'title': 'Module 2: Types of Drift',
        'icon': '[2]',
        'content': 'Data drift vs Concept drift'
    },
    {
        'title': 'Module 3: Visual Degradation',
        'icon': '[3]',
        'content': 'Real-world visual drift examples'
    },
    {
        'title': 'Module 4: Detection Methods',
        'icon': '[4]',
        'content': 'Statistical tests for drift'
    },
    {
        'title': 'Module 5: Monitoring Strategies',
        'icon': '[5]',
        'content': 'Best practices for production'
    },
    {
        'title': 'Module 6: Hands-On Exercise',
        'icon': '[6]',
        'content': 'Practice drift detection'
    }
]

# Sidebar - Module navigation
with st.sidebar:
    st.markdown("## Tutorial Modules")
    st.markdown("---")
    
    for idx, module in enumerate(modules):
        completed = idx < st.session_state.tutorial_progress
        icon = "[✓]" if completed else module['icon']
        
        if st.button(
            f"{icon} {module['title']}",
            key=f"nav_{idx}",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.current_module = idx

if 'current_module' not in st.session_state:
    st.session_state.current_module = 0

current_idx = st.session_state.current_module

# Module content
if current_idx == 0:
    st.markdown("## Module 1: What is Model Drift?")
    st.markdown("---")
    
    st.markdown("""
    <div class="tutorial-card">
    
    ### Learning Objectives
    - Understand what model drift means
    - Learn why drift happens in production
    - Recognize the impact of drift on ML systems
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### What is Model Drift?
    
    **Model drift** occurs when the statistical properties of the data your model receives in 
    production differ from the data it was trained on. This causes model performance to degrade over time.
    
    #### Why Does Drift Happen?
    
    1. **Sensor Degradation**: Medical imaging devices wear out or get miscalibrated
    2. **Population Changes**: Patient demographics shift over time
    3. **Protocol Updates**: New imaging protocols or equipment
    4. **Environmental Factors**: Lighting, temperature, humidity changes
    5. **Adversarial Changes**: Data patterns evolve naturally
    
    #### Real-World Example
    
    A skin lesion classifier trained on high-quality dermatoscope images starts receiving 
    smartphone photos from a telemedicine app. The model's accuracy drops from 95% to 75% 
    because:
    - Different image quality (blur, compression)
    - Different lighting conditions
    - Different angles and distances
    """)
    
    # Interactive example
    st.markdown("### Interactive Example")
    st.markdown("Compare training data quality with production data quality")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Data (High Quality)**")
        st.info("📸 Professional dermatoscope images\n\n✨ Controlled lighting\n\n🎯 95% accuracy")
    
    with col2:
        st.markdown("**Production Data (Smartphone)**")
        st.warning("📱 Smartphone camera photos\n\n🌤️ Variable lighting\n\n📉 75% accuracy (drift!)")
    
    # Quiz
    st.markdown("### ✅ Knowledge Check")
    
    answer = st.radio(
        "What is the main cause of model drift?",
        [
            "A) The model forgets its training",
            "B) The production data differs from training data",
            "C) The model gets corrupted",
            "D) Hardware failures"
        ],
        key='quiz_1'
    )
    
    if st.button("Submit Answer", key='submit_1'):
        if 'B)' in answer:
            st.success("✅ Correct! Drift happens when production data differs from training data.")
            if st.session_state.tutorial_progress < 1:
                st.session_state.tutorial_progress = 1
        else:
            st.error("❌ Not quite. Drift is caused by changes in the data distribution, not the model itself.")

elif current_idx == 1:
    st.markdown("## Module 2: Types of Drift")
    st.markdown("---")
    
    st.markdown("""
    <div class="tutorial-card">
    
    ### Learning Objectives
    - Distinguish between data drift and concept drift
    - Understand covariate shift
    - Recognize label shift
    
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Data Drift (Covariate Shift)
        
        The **input distribution** changes, but the relationship between inputs and outputs stays the same.
        
        **Example**: Medical images become blurrier due to equipment aging
        - X distribution changes: P(X) → P'(X)
        - Relationship unchanged: P(Y|X) = P'(Y|X)
        
        **Symptoms**:
        - Features drift but labels don't
        - Model confidence drops
        - Input statistics change
        """)
    
    with col2:
        st.markdown("""
        ### Concept Drift
        
        The **relationship** between inputs and outputs changes.
        
        **Example**: Disease presentation changes (new variant)
        - X distribution same: P(X) = P'(X)
        - Relationship changes: P(Y|X) ≠ P'(Y|X)
        
        **Symptoms**:
        - Model predictions wrong
        - Accuracy drops significantly
        - Ground truth feedback shows errors
        """)
    
    # Visual comparison
    st.markdown("### Visual Comparison")
    st.markdown("Examine the difference between data drift and concept drift")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Data Drift Example", "Concept Drift Example"])
    
    with tab1:
        st.markdown("""
        **Scenario**: X-ray machine's sensor degrades over time
        
        - Week 1: Sharp, clear images → Model accuracy 95%
        - Week 10: Progressively blurrier → Model accuracy 85%
        - Week 20: Very blurry → Model accuracy 70%
        
        The **features changed** (image quality), but pneumonia still looks the same when visible.
        """)
        
        # Simulated drift visualization
        x_vals = np.linspace(0, 10, 100)
        y_original = np.sin(x_vals) + np.random.normal(0, 0.1, 100)
        y_drifted = np.sin(x_vals) + np.random.normal(0, 0.3, 100)  # More noise
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_original, mode='markers', name='Original', 
                                marker=dict(color=COLORS['success'])))
        fig.add_trace(go.Scatter(x=x_vals, y=y_drifted, mode='markers', name='Drifted (noisy)', 
                                marker=dict(color=COLORS['danger'])))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=300,
            title="Data Drift: Same pattern, different noise"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("""
        **Scenario**: New COVID variant changes lung presentation
        
        - Original: Ground-glass opacity = COVID
        - New variant: Consolidation = COVID
        
        The **relationship changed** between image features and diagnosis.
        """)
        
        # Concept drift visualization
        x_vals = np.linspace(0, 10, 100)
        y_original = np.sin(x_vals)
        y_concept_drift = np.sin(x_vals + 2)  # Phase shift
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_original, mode='lines', name='Original Concept', 
                                line=dict(color=COLORS['success'], width=3)))
        fig.add_trace(go.Scatter(x=x_vals, y=y_concept_drift, mode='lines', name='New Concept', 
                                line=dict(color=COLORS['danger'], width=3)))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=300,
            title="Concept Drift: Different relationship"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Quiz
    st.markdown("### ✅ Knowledge Check")
    
    scenario = st.radio(
        "Which type of drift is this? 'A hospital switches from one X-ray machine to another brand, images look different but diseases look the same'",
        [
            "A) Data Drift (Covariate Shift)",
            "B) Concept Drift",
            "C) Label Shift",
            "D) No drift occurred"
        ],
        key='quiz_2'
    )
    
    if st.button("Submit Answer", key='submit_2'):
        if 'A)' in scenario:
            st.success("✅ Correct! This is data drift - the input distribution changed but the relationship stayed the same.")
            if st.session_state.tutorial_progress < 2:
                st.session_state.tutorial_progress = 2
        else:
            st.error("❌ Think about what changed: the images (X) or the disease patterns (Y|X)?")

elif current_idx == 2:
    st.markdown("## Module 3: Visual Degradation")
    st.markdown("---")
    
    st.markdown("""
    <div class="tutorial-card">
    
    ### Learning Objectives
    - Identify common visual drift types
    - See real examples of degradation
    - Understand impact on model performance
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Common Visual Drift Types")
    
    # Load sample data
    @st.cache_resource
    def load_sample():
        loader = MedMNISTLoader('pathmnist')
        images, _ = loader.get_numpy_data('test')
        return images[:1]
    
    original = load_sample()
    generator = DriftGenerator(seed=42)
    
    # Show different drift types
    drift_types = ['brightness_shift', 'contrast_reduction', 'blur', 'noise', 'motion_blur', 'jpeg_compression']
    
    cols = st.columns(3)
    
    for idx, drift_type in enumerate(drift_types):
        col = cols[idx % 3]
        
        with col:
            st.markdown(f"**{drift_type.replace('_', ' ').title()}**")
            
            method = getattr(generator, f'_apply_{drift_type}')
            drifted = method(original.copy(), 0.7)
            
            if drifted.shape[-1] == 1:
                img_display = Image.fromarray(drifted[0].squeeze(), mode='L')
            else:
                img_display = Image.fromarray(drifted[0].astype(np.uint8))
            
            st.image(img_display, use_container_width=True)
    
    # Impact table
    st.markdown("### Impact on Model Performance")
    st.markdown("Understand how different drift types affect model accuracy")
    st.markdown("---")
    
    impact_data = {
        'Drift Type': ['Brightness', 'Contrast', 'Blur', 'Noise', 'Motion Blur', 'Compression'],
        'Severity': ['Low', 'Medium', 'High', 'Medium', 'High', 'Low'],
        'Accuracy Drop': ['2-5%', '5-10%', '10-20%', '5-15%', '15-25%', '1-5%'],
        'Detectability': ['Easy', 'Easy', 'Medium', 'Easy', 'Easy', 'Hard']
    }
    
    st.table(impact_data)
    
    # Interactive demo
    st.markdown("### Try It Yourself")
    st.markdown("Experiment with different drift types and severity levels")
    st.markdown("---")
    
    selected_drift = st.selectbox("Choose a drift type", drift_types)
    severity = st.slider("Severity", 0.0, 1.0, 0.5, 0.1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original**")
        if original.shape[-1] == 1:
            img_display = Image.fromarray(original[0].squeeze(), mode='L')
        else:
            img_display = Image.fromarray(original[0].astype(np.uint8))
        st.image(img_display, use_container_width=True)
    
    with col2:
        st.markdown("**Drifted**")
        method = getattr(generator, f'_apply_{selected_drift}')
        drifted = method(original.copy(), severity)
        
        if drifted.shape[-1] == 1:
            img_display = Image.fromarray(drifted[0].squeeze(), mode='L')
        else:
            img_display = Image.fromarray(drifted[0].astype(np.uint8))
        st.image(img_display, use_container_width=True)
    
    if st.button("Mark Complete", key='complete_3'):
        if st.session_state.tutorial_progress < 3:
            st.session_state.tutorial_progress = 3
            st.success("✅ Module completed!")

elif current_idx == 3:
    st.markdown("## Module 4: Detection Methods")
    st.markdown("---")
    
    st.markdown("""
    <div class="tutorial-card">
    
    ### Learning Objectives
    - Learn statistical tests for drift detection
    - Understand when to use each method
    - Interpret test results
    
    </div>
    """, unsafe_allow_html=True)
    
    methods = {
        'PSI (Population Stability Index)': {
            'description': 'Measures distribution shift by comparing binned distributions',
            'pros': '✅ Easy to interpret, fast computation',
            'cons': '❌ Requires binning, loses information',
            'use_when': 'Quick monitoring, categorical features',
            'threshold': 0.2
        },
        'KS Test (Kolmogorov-Smirnov)': {
            'description': 'Tests if two samples come from same distribution',
            'pros': '✅ No assumptions, distribution-free',
            'cons': '❌ Only univariate, sensitive to sample size',
            'use_when': 'Continuous features, small datasets',
            'threshold': 0.05
        },
        'Chi-Square Test': {
            'description': 'Tests independence between categorical variables',
            'pros': '✅ Well-established, interpretable',
            'cons': '❌ Requires sufficient sample size',
            'use_when': 'Categorical data, count-based features',
            'threshold': 0.05
        },
        'MMD (Maximum Mean Discrepancy)': {
            'description': 'Kernel-based distance between distributions',
            'pros': '✅ Multivariate, powerful',
            'cons': '❌ Computational cost, kernel choice',
            'use_when': 'High-dimensional data, complex patterns',
            'threshold': 0.1
        },
        'Wasserstein Distance': {
            'description': 'Optimal transport distance between distributions',
            'pros': '✅ Geometric interpretation, robust',
            'cons': '❌ Computational complexity',
            'use_when': 'Image data, continuous distributions',
            'threshold': 0.1
        }
    }
    
    for method_name, info in methods.items():
        with st.expander(f"**{method_name}**"):
            st.markdown(f"**Description**: {info['description']}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(info['pros'])
            with col2:
                st.markdown(info['cons'])
            st.markdown(f"**Use When**: {info['use_when']}")
            st.markdown(f"**Typical Threshold**: {info['threshold']}")
    
    # Interactive comparison
    st.markdown("### Compare Detection Methods")
    st.markdown("Run multiple statistical tests and compare their results")
    st.markdown("---")
    
    @st.cache_resource
    def load_and_drift():
        loader = MedMNISTLoader('pathmnist')
        images, _ = loader.get_numpy_data('test')
        reference = images[:200]
        
        generator = DriftGenerator(seed=42)
        drifted = generator._apply_blur(reference.copy(), 0.6)
        
        return reference, drifted
    
    if st.button("Run All Tests", type="primary"):
        with st.spinner("Running drift detection tests..."):
            reference, drifted = load_and_drift()
            
            detector = DriftDetector()
            
            results = {}
            for method in ['psi', 'ks_test', 'chi_square', 'mmd', 'wasserstein']:
                score, detected = detector.detect_drift(reference, drifted, method=method)
                results[method] = {'score': score, 'detected': detected}
            
            st.session_state.detection_results = results
    
    if 'detection_results' in st.session_state:
        st.markdown("#### Test Results")
        st.markdown("Compare drift detection scores across different methods")
        st.markdown("---")
        
        for method, result in st.session_state.detection_results.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{method.upper()}**")
            with col2:
                st.metric("Score", f"{result['score']:.4f}")
            with col3:
                if result['detected']:
                    st.error("🔴 Drift Detected")
                else:
                    st.success("🟢 No Drift")
    
    if st.button("Mark Complete", key='complete_4'):
        if st.session_state.tutorial_progress < 4:
            st.session_state.tutorial_progress = 4
            st.success("✅ Module completed!")

elif current_idx == 4:
    st.markdown("## Module 5: Monitoring Strategies")
    st.markdown("---")
    
    st.markdown("""
    <div class="tutorial-card">
    
    ### Learning Objectives
    - Design effective monitoring pipelines
    - Set appropriate alert thresholds
    - Handle drift in production
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Best Practices
    
    #### 1. Continuous Monitoring
    - Monitor **every prediction** if possible
    - Batch monitoring for high-volume systems
    - Real-time alerting for critical applications
    
    #### 2. Multi-Level Alerts
    
    - **🟢 Green (0-0.2)**: Normal operation, log for analysis
    - **🟡 Yellow (0.2-0.4)**: Minor drift, investigate trends
    - **🟠 Orange (0.4-0.6)**: Moderate drift, prepare mitigation
    - **🔴 Red (>0.6)**: Severe drift, immediate action required
    
    #### 3. Window Strategies
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Fixed Window**
        - Compare to original training data
        - Simple, stable baseline
        - May miss gradual shifts
        
        ```python
        reference = train_data
        current = production_batch
        drift_score = detect(reference, current)
        ```
        """)
    
    with col2:
        st.markdown("""
        **Sliding Window**
        - Compare recent batches
        - Adapts to gradual drift
        - Detects sudden changes
        
        ```python
        reference = last_n_batches
        current = latest_batch
        drift_score = detect(reference, current)
        ```
        """)
    
    st.markdown("""
    #### 4. Mitigation Strategies
    
    When drift is detected:
    
    1. **Investigate Root Cause**
       - Check data sources
       - Review recent changes
       - Analyze drift patterns
    
    2. **Short-term Fixes**
       - Route to human reviewers
       - Use confidence thresholds
       - Ensemble with robust models
    
    3. **Long-term Solutions**
       - Retrain with new data
       - Update feature engineering
       - Adapt model architecture
    
    4. **Continuous Improvement**
       - A/B test new models
       - Monitor retraining impact
       - Update monitoring thresholds
    """)
    
    # Monitoring checklist
    st.markdown("### Production Checklist")
    st.markdown("Essential tasks for production ML monitoring")
    st.markdown("---")
    
    checklist = [
        "Set up automated drift monitoring",
        "Define alert thresholds for each metric",
        "Create escalation procedures",
        "Log all predictions and features",
        "Store ground truth labels when available",
        "Schedule regular model retraining",
        "Document drift incidents",
        "Review monitoring dashboard weekly"
    ]
    
    for item in checklist:
        st.checkbox(item, key=f"check_{item}")
    
    if st.button("Mark Complete", key='complete_5'):
        if st.session_state.tutorial_progress < 5:
            st.session_state.tutorial_progress = 5
            st.success("✅ Module completed!")

elif current_idx == 5:
    st.markdown("## Module 6: Hands-On Exercise")
    st.markdown("---")
    
    st.markdown("""
    <div class="tutorial-card">
    
    ### Your Mission
    Detect and diagnose drift in a medical imaging dataset!
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Scenario
    
    You're monitoring a skin lesion classifier in production. The model was trained on 
    high-quality dermatoscope images, but recently accuracy has dropped. Your job is to:
    
    1. Load the reference and production datasets
    2. Run drift detection tests
    3. Diagnose the type of drift
    4. Recommend mitigation strategies
    """)
    
    if st.button("Start Exercise", type="primary"):
        with st.spinner("Loading datasets..."):
            loader = MedMNISTLoader('dermamnist')
            images, _ = loader.get_numpy_data('test')
            reference = images[:300]
            
            # Simulate production drift (compression + blur)
            generator = DriftGenerator(seed=42)
            production = generator._apply_jpeg_compression(reference.copy(), 0.7)
            production = generator._apply_blur(production, 0.5)
            
            st.session_state.exercise_ref = reference
            st.session_state.exercise_prod = production
            st.success("✅ Data loaded!")
    
    if 'exercise_ref' in st.session_state:
        st.markdown("### Step 1: Visual Inspection")
        st.markdown("Compare reference and production data side by side")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Reference (Training)**")
            idx = np.random.randint(0, len(st.session_state.exercise_ref))
            img = st.session_state.exercise_ref[idx]
            if img.shape[-1] == 1:
                img_display = Image.fromarray(img.squeeze(), mode='L')
            else:
                img_display = Image.fromarray(img.astype(np.uint8))
            st.image(img_display, use_container_width=True)
        
        with col2:
            st.markdown("**Production**")
            img = st.session_state.exercise_prod[idx]
            if img.shape[-1] == 1:
                img_display = Image.fromarray(img.squeeze(), mode='L')
            else:
                img_display = Image.fromarray(img.astype(np.uint8))
            st.image(img_display, use_container_width=True)
        
        st.markdown("### Step 2: Run Detection Tests")
        st.markdown("Execute statistical tests to quantify drift")
        st.markdown("---")
        
        if st.button("Run Detection", type="primary"):
            detector = DriftDetector()
            
            # Flatten images for drift detection
            ref_flat = st.session_state.exercise_ref.reshape(len(st.session_state.exercise_ref), -1)
            prod_flat = st.session_state.exercise_prod.reshape(len(st.session_state.exercise_prod), -1)
            
            results = {}
            for method in ['psi', 'ks_test', 'mmd', 'wasserstein']:
                result = detector.detect_drift(
                    ref_flat,
                    prod_flat,
                    method=method
                )
                # Extract score and detected from the nested structure
                method_result = result['methods'].get(method, {})
                if 'score' in method_result:
                    score = method_result['score']
                elif 'scores' in method_result:
                    score = np.mean(method_result['scores'])
                elif 'distance' in method_result:
                    score = method_result['distance']
                elif 'distances' in method_result:
                    score = np.mean(method_result['distances'])
                else:
                    score = 0.0
                detected = method_result.get('drift_detected', False)
                results[method] = {'score': score, 'detected': detected}
            
            st.session_state.exercise_results = results
        
        if 'exercise_results' in st.session_state:
            st.markdown("#### 📊 Your Results")
            
            for method, result in st.session_state.exercise_results.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{method.upper()}**")
                with col2:
                    st.metric("Score", f"{result['score']:.4f}")
                with col3:
                    if result['detected']:
                        st.error("🔴 Drift")
                    else:
                        st.success("🟢 No Drift")
            
            st.markdown("### Step 3: Diagnosis")
            st.markdown("Interpret the results and identify the drift type")
            st.markdown("---")
            
            diagnosis = st.radio(
                "Based on the visual inspection and test results, what type of drift occurred?",
                [
                    "A) Data Drift - Image quality degradation",
                    "B) Concept Drift - Disease patterns changed",
                    "C) No significant drift detected",
                    "D) Label drift - Class distribution changed"
                ]
            )
            
            if st.button("Submit Diagnosis"):
                if 'A)' in diagnosis:
                    st.success("""
                    ✅ Correct! This is **data drift** caused by image quality degradation.
                    
                    Evidence:
                    - Visual inspection shows compression artifacts and blur
                    - Statistical tests detected distribution shift
                    - The disease patterns themselves haven't changed
                    
                    **Recommended Actions**:
                    1. Investigate image acquisition pipeline
                    2. Apply preprocessing to normalize image quality
                    3. Retrain model with augmentation (blur, compression)
                    4. Set up quality monitoring alerts
                    """)
                    
                    if st.session_state.tutorial_progress < 6:
                        st.session_state.tutorial_progress = 6
                        st.balloons()
                        st.success("🎉 Congratulations! You've completed all tutorial modules!")
                else:
                    st.error("Not quite. Look at the images - what changed? The features (image quality) or the relationship (disease patterns)?")

# Navigation buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if current_idx > 0:
        if st.button("⬅️ Previous Module"):
            st.session_state.current_module -= 1
            st.rerun()

with col3:
    if current_idx < len(modules) - 1:
        if st.button("Next Module ➡️"):
            st.session_state.current_module += 1
            st.rerun()
