"""
Drift Lab - Custom Drift Creation and Experimentation

Create custom drift scenarios by combining multiple drift types
with individual severity controls.
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

st.set_page_config(
    page_title="SimDrift - Drift Lab",
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
        font-size: 3rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }}
    
    .hero-subtitle {{
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95);
        position: relative;
        z-index: 1;
        margin-top: 0.5rem;
    }}
    .drift-card {{
        background: linear-gradient(135deg, {COLORS['surface']}, {COLORS['background']});
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'custom_drift_config' not in st.session_state:
    st.session_state.custom_drift_config = {}
if 'lab_images' not in st.session_state:
    st.session_state.lab_images = None

@st.cache_resource
def load_dataset(dataset_name: str):
    return MedMNISTLoader(dataset_name=dataset_name)

st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">⚡ Drift Lab</h1>
    <p class="hero-subtitle">
        Experiment with Custom Drift Scenarios
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Combine multiple drift types to create realistic production scenarios or explore edge cases.
Design, preview, and save custom drift configurations.
""")
st.markdown("---")

# Sidebar - Dataset selection
with st.sidebar:
    st.markdown("## Dataset Selection")
    st.markdown("---")
    dataset_name = st.selectbox(
        "Select Dataset",
        ['pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist', 'pneumoniamnist']
    )
    
    if st.button("Load Dataset", type="primary"):
        with st.spinner("Loading dataset..."):
            loader = load_dataset(dataset_name)
            images, labels = loader.get_numpy_data('test')
            st.session_state.lab_images = images[:100]  # Sample for speed
            st.session_state.lab_labels = labels[:100]
            st.success(f"Loaded {len(images)} images!")

# Main content
if st.session_state.lab_images is not None:
    tab1, tab2, tab3 = st.tabs(["Design Drift", "Preview", "Save Scenario"])
    
    with tab1:
        st.markdown("### Configure Drift Components")
        st.info("Select and configure multiple drift types to create a custom scenario")
        st.markdown("---")
        
        # Available drift types
        generator = DriftGenerator()
        all_drift_types = generator.get_available_drift_types()
        
        # Categorize drift types
        visual_drifts = [
            'brightness', 'contrast', 'blur', 'noise', 'motion_blur',
            'jpeg_compression', 'occlusion', 'zoom', 'vignette',
            'color_temperature', 'saturation'
        ]
        
        concept_drifts = ['demographic']
        
        # Visual Drifts
        st.markdown("#### Visual Degradation")
        st.markdown("Apply image quality degradations commonly seen in production")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        drift_config = {}
        
        for idx, drift_type in enumerate(visual_drifts):
            col = col1 if idx % 2 == 0 else col2
            
            with col:
                with st.expander(f"**{drift_type.replace('_', ' ').title()}**"):
                    st.caption(all_drift_types.get(drift_type, "No description"))
                    
                    enabled = st.checkbox(
                        "Enable",
                        key=f"enable_{drift_type}",
                        value=drift_type in ['brightness', 'blur']  # Default selection
                    )
                    
                    if enabled:
                        severity = st.slider(
                            "Severity",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.05,
                            key=f"severity_{drift_type}"
                        )
                        drift_config[drift_type] = severity
        
        # Concept Drift
        st.markdown("#### Concept Drift")
        st.markdown("Simulate changes in data distribution or population")
        st.markdown("---")
        
        for drift_type in concept_drifts:
            with st.expander(f"**{drift_type.replace('_', ' ').title()}**"):
                st.caption(all_drift_types.get(drift_type, "No description"))
                
                enabled = st.checkbox(
                    "Enable",
                    key=f"enable_{drift_type}"
                )
                
                if enabled:
                    severity = st.slider(
                        "Severity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        key=f"severity_{drift_type}"
                    )
                    drift_config[drift_type] = severity
        
        st.session_state.custom_drift_config = drift_config
        
        # Summary
        st.markdown("### Configuration Summary")
        if drift_config:
            summary_data = {
                "Drift Type": [dt.replace('_', ' ').title() for dt in drift_config.keys()],
                "Severity": [f"{s:.0%}" for s in drift_config.values()]
            }
            st.table(summary_data)
        else:
            st.warning("No drift types selected. Enable at least one drift type above.")
    
    with tab2:
        st.markdown("### Preview Drift Effect")
        st.markdown("Generate a preview to see the combined effect of all selected drift types")
        st.markdown("---")
        
        if st.session_state.custom_drift_config:
            if st.button("Generate Preview", type="primary"):
                with st.spinner("Applying drift transformations..."):
                    # Apply each drift sequentially
                    drifted = st.session_state.lab_images.copy()
                    generator = DriftGenerator(seed=42)
                    
                    for drift_type, severity in st.session_state.custom_drift_config.items():
                        if drift_type == 'demographic':
                            drifted, labels = generator._apply_demographic_shift(
                                drifted, st.session_state.lab_labels, severity
                            )
                        else:
                            # Apply visual drift
                            method = getattr(generator, f'_apply_{drift_type}', None)
                            if method:
                                drifted = method(drifted, severity)
                    
                    st.session_state.preview_drifted = drifted
            
            # Display comparison
            if 'preview_drifted' in st.session_state:
                st.markdown("#### Before vs After")
                
                n_display = st.slider("Number of samples", 4, 16, 8, 4)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original**")
                    cols = st.columns(4)
                    for i in range(min(n_display, len(st.session_state.lab_images))):
                        with cols[i % 4]:
                            img = st.session_state.lab_images[i]
                            if img.shape[-1] == 1:
                                img_display = Image.fromarray(img.squeeze(), mode='L')
                            else:
                                img_display = Image.fromarray(img.astype(np.uint8))
                            st.image(img_display, use_container_width=True)
                
                with col2:
                    st.markdown("**Drifted**")
                    cols = st.columns(4)
                    for i in range(min(n_display, len(st.session_state.preview_drifted))):
                        with cols[i % 4]:
                            img = st.session_state.preview_drifted[i]
                            if img.shape[-1] == 1:
                                img_display = Image.fromarray(img.squeeze(), mode='L')
                            else:
                                img_display = Image.fromarray(img.astype(np.uint8))
                            st.image(img_display, use_container_width=True)
                
                # Statistics
                st.markdown("#### Drift Statistics")
                st.markdown("Quantitative measures of drift impact")
                st.markdown("---")
                
                original_flat = st.session_state.lab_images.reshape(len(st.session_state.lab_images), -1)
                drifted_flat = st.session_state.preview_drifted.reshape(len(st.session_state.preview_drifted), -1)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    orig_mean = original_flat.mean()
                    drift_mean = drifted_flat.mean()
                    st.metric("Mean Pixel Value", f"{drift_mean:.2f}", f"{drift_mean - orig_mean:+.2f}")
                
                with col2:
                    orig_std = original_flat.std()
                    drift_std = drifted_flat.std()
                    st.metric("Std Deviation", f"{drift_std:.2f}", f"{drift_std - orig_std:+.2f}")
                
                with col3:
                    mse = np.mean((original_flat - drifted_flat) ** 2)
                    st.metric("MSE", f"{mse:.2f}")
                
                with col4:
                    max_diff = np.abs(original_flat - drifted_flat).max()
                    st.metric("Max Difference", f"{max_diff:.2f}")
        
        else:
            st.info("👈 Configure drift types in the 'Design Drift' tab first")
    
    with tab3:
        st.markdown("### 💾 Save Custom Scenario")
        
        if st.session_state.custom_drift_config:
            scenario_name = st.text_input(
                "Scenario Name",
                value="custom_drift_scenario",
                help="Give your scenario a memorable name"
            )
            
            scenario_desc = st.text_area(
                "Description",
                value="",
                help="Describe what this scenario simulates",
                height=100
            )
            
            severity_rating = st.select_slider(
                "Severity Rating",
                options=['low', 'moderate', 'high', 'severe'],
                value='moderate'
            )
            
            # Show configuration
            st.markdown("#### Configuration Preview")
            st.json({
                "name": scenario_name,
                "description": scenario_desc,
                "severity": severity_rating,
                "drift_config": st.session_state.custom_drift_config
            })
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("💾 Save Scenario", type="primary", use_container_width=True):
                    # Save to session state (could be extended to save to file)
                    if 'saved_scenarios' not in st.session_state:
                        st.session_state.saved_scenarios = []
                    
                    scenario = {
                        'name': scenario_name,
                        'description': scenario_desc,
                        'severity': severity_rating,
                        'config': st.session_state.custom_drift_config.copy()
                    }
                    
                    st.session_state.saved_scenarios.append(scenario)
                    st.success(f"Scenario '{scenario_name}' saved successfully!")
            
            with col2:
                if st.button("📋 Copy Configuration", use_container_width=True):
                    import json
                    config_str = json.dumps(st.session_state.custom_drift_config, indent=2)
                    st.code(config_str, language='json')
            
            # Show saved scenarios
            if 'saved_scenarios' in st.session_state and st.session_state.saved_scenarios:
                st.markdown("#### 📚 Saved Scenarios")
                
                for idx, scenario in enumerate(st.session_state.saved_scenarios):
                    with st.expander(f"{scenario['name']} ({scenario['severity']})"):
                        st.markdown(f"**Description:** {scenario['description']}")
                        st.markdown(f"**Drift Types:** {len(scenario['config'])}")
                        st.json(scenario['config'])
                        
                        if st.button(f"Load Scenario", key=f"load_{idx}"):
                            st.session_state.custom_drift_config = scenario['config'].copy()
                            st.success(f"Loaded '{scenario['name']}'!")
                            st.rerun()
        
        else:
            st.info("👈 Configure drift types in the 'Design Drift' tab to save a scenario")

else:
    st.info("👈 Load a dataset from the sidebar to begin experimenting")
    
    st.markdown("""
    ### About Drift Lab
    
    The Drift Lab allows you to:
    
    - **Combine Multiple Drift Types**: Stack visual and concept drifts
    - **Fine-tune Severity**: Independent control for each drift type
    - **Preview Effects**: See drift impact before full analysis
    - **Save Scenarios**: Create reusable drift configurations
    - **Export Configurations**: Share scenarios with colleagues
    
    ### Use Cases
    
    1. **Testing Robustness**: See how models handle combined drifts
    2. **Scenario Planning**: Model realistic production degradation
    3. **Education**: Demonstrate drift concepts interactively
    4. **Research**: Explore drift interactions and dependencies
    """)

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
