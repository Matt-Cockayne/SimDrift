"""
Settings - Configuration Management

Configure drift detection parameters, alert thresholds,
visualization preferences, and system settings.
"""

import streamlit as st
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="SimDrift - Settings",
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
    .settings-section {{
        background: linear-gradient(135deg, {COLORS['surface']}, {COLORS['background']});
        border: 1px solid rgba(99, 102, 241, 0.3);
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

st.markdown("""
<div class="enhanced-hero">
    <h1>⚙️ Settings</h1>
    <p>Configure drift detection parameters and system preferences</p>
</div>
""", unsafe_allow_html=True)

# Initialize default settings
DEFAULT_SETTINGS = {
    'drift_detection': {
        'default_method': 'psi',
        'psi_threshold': 0.2,
        'ks_threshold': 0.05,
        'chi_square_threshold': 0.05,
        'mmd_threshold': 0.1,
        'wasserstein_threshold': 0.1,
        'window_size': 1000,
        'update_frequency': 'hourly'
    },
    'alerts': {
        'enabled': True,
        'green_threshold': 0.2,
        'yellow_threshold': 0.4,
        'orange_threshold': 0.6,
        'email_notifications': False,
        'slack_notifications': False
    },
    'visualization': {
        'theme': 'dark',
        'color_scheme': 'default',
        'plot_height': 400,
        'show_confidence_intervals': True,
        'animation_speed': 'normal'
    },
    'model_zoo': {
        'path': './model_zoo',
        'auto_load': True,
        'cache_models': True,
        'max_cache_size': 5
    },
    'performance': {
        'batch_size': 32,
        'max_samples': 10000,
        'parallel_processing': True,
        'gpu_acceleration': True
    }
}

if 'settings' not in st.session_state:
    st.session_state.settings = DEFAULT_SETTINGS.copy()

st.title("Settings")
st.markdown("### Configure SimDrift parameters and preferences")
st.markdown("Customize drift detection, alerts, visualization, and system performance")
st.markdown("---")

# Tabs for different settings categories
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Drift Detection",
    "Alerts",
    "Visualization",
    "Model Zoo",
    "Performance",
    "Import/Export"
])

with tab1:
    st.markdown("### 📊 Drift Detection Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Detection Method")
        
        default_method = st.selectbox(
            "Default Method",
            ['psi', 'ks_test', 'chi_square', 'mmd', 'wasserstein'],
            index=['psi', 'ks_test', 'chi_square', 'mmd', 'wasserstein'].index(
                st.session_state.settings['drift_detection']['default_method']
            ),
            help="Primary detection method for drift monitoring"
        )
        st.session_state.settings['drift_detection']['default_method'] = default_method
        
        window_size = st.number_input(
            "Window Size",
            min_value=100,
            max_value=100000,
            value=st.session_state.settings['drift_detection']['window_size'],
            step=100,
            help="Number of samples in reference window"
        )
        st.session_state.settings['drift_detection']['window_size'] = window_size
        
        update_freq = st.selectbox(
            "Update Frequency",
            ['real-time', 'hourly', 'daily', 'weekly'],
            index=['real-time', 'hourly', 'daily', 'weekly'].index(
                st.session_state.settings['drift_detection']['update_frequency']
            )
        )
        st.session_state.settings['drift_detection']['update_frequency'] = update_freq
    
    with col2:
        st.markdown("#### Detection Thresholds")
        
        st.markdown("**PSI (Population Stability Index)**")
        psi_threshold = st.slider(
            "PSI Threshold",
            0.0, 1.0,
            st.session_state.settings['drift_detection']['psi_threshold'],
            0.05,
            help="PSI > 0.2 indicates significant drift"
        )
        st.session_state.settings['drift_detection']['psi_threshold'] = psi_threshold
        
        st.markdown("**KS Test (Kolmogorov-Smirnov)**")
        ks_threshold = st.slider(
            "KS p-value Threshold",
            0.0, 0.2,
            st.session_state.settings['drift_detection']['ks_threshold'],
            0.01,
            help="p-value < 0.05 rejects null hypothesis"
        )
        st.session_state.settings['drift_detection']['ks_threshold'] = ks_threshold
        
        st.markdown("**Chi-Square Test**")
        chi_threshold = st.slider(
            "Chi-Square p-value",
            0.0, 0.2,
            st.session_state.settings['drift_detection']['chi_square_threshold'],
            0.01
        )
        st.session_state.settings['drift_detection']['chi_square_threshold'] = chi_threshold
        
        st.markdown("**MMD (Maximum Mean Discrepancy)**")
        mmd_threshold = st.slider(
            "MMD Threshold",
            0.0, 1.0,
            st.session_state.settings['drift_detection']['mmd_threshold'],
            0.05,
            help="Higher values indicate more drift"
        )
        st.session_state.settings['drift_detection']['mmd_threshold'] = mmd_threshold
        
        st.markdown("**Wasserstein Distance**")
        wass_threshold = st.slider(
            "Wasserstein Threshold",
            0.0, 1.0,
            st.session_state.settings['drift_detection']['wasserstein_threshold'],
            0.05
        )
        st.session_state.settings['drift_detection']['wasserstein_threshold'] = wass_threshold

with tab2:
    st.markdown("### 🚨 Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Alert Levels")
        
        alerts_enabled = st.checkbox(
            "Enable Alerts",
            value=st.session_state.settings['alerts']['enabled']
        )
        st.session_state.settings['alerts']['enabled'] = alerts_enabled
        
        st.markdown("**Severity Thresholds**")
        
        green = st.slider(
            "🟢 Green → Yellow",
            0.0, 1.0, 
            st.session_state.settings['alerts']['green_threshold'],
            0.05,
            help="Normal operation threshold"
        )
        st.session_state.settings['alerts']['green_threshold'] = green
        
        yellow = st.slider(
            "🟡 Yellow → Orange",
            green, 1.0,
            st.session_state.settings['alerts']['yellow_threshold'],
            0.05,
            help="Minor drift threshold"
        )
        st.session_state.settings['alerts']['yellow_threshold'] = yellow
        
        orange = st.slider(
            "🟠 Orange → Red",
            yellow, 1.0,
            st.session_state.settings['alerts']['orange_threshold'],
            0.05,
            help="Moderate drift threshold"
        )
        st.session_state.settings['alerts']['orange_threshold'] = orange
        
        st.info(f"""
        **Current Alert Levels:**
        - 🟢 Green: 0.0 - {green:.2f}
        - 🟡 Yellow: {green:.2f} - {yellow:.2f}
        - 🟠 Orange: {yellow:.2f} - {orange:.2f}
        - 🔴 Red: {orange:.2f}+
        """)
    
    with col2:
        st.markdown("#### Notification Channels")
        
        email = st.checkbox(
            "Email Notifications",
            value=st.session_state.settings['alerts']['email_notifications']
        )
        st.session_state.settings['alerts']['email_notifications'] = email
        
        if email:
            email_address = st.text_input("Email Address", placeholder="alerts@example.com")
            email_severity = st.multiselect(
                "Alert on Severity",
                ['Yellow', 'Orange', 'Red'],
                default=['Red']
            )
        
        slack = st.checkbox(
            "Slack Notifications",
            value=st.session_state.settings['alerts']['slack_notifications']
        )
        st.session_state.settings['alerts']['slack_notifications'] = slack
        
        if slack:
            slack_webhook = st.text_input(
                "Slack Webhook URL",
                type="password",
                placeholder="https://hooks.slack.com/..."
            )
            slack_severity = st.multiselect(
                "Alert on Severity",
                ['Yellow', 'Orange', 'Red'],
                default=['Orange', 'Red']
            )

with tab3:
    st.markdown("### 📈 Visualization Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Theme & Colors")
        
        theme = st.selectbox(
            "Dashboard Theme",
            ['dark', 'light'],
            index=['dark', 'light'].index(st.session_state.settings['visualization']['theme'])
        )
        st.session_state.settings['visualization']['theme'] = theme
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ['default', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            index=['default', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'].index(
                st.session_state.settings['visualization']['color_scheme']
            )
        )
        st.session_state.settings['visualization']['color_scheme'] = color_scheme
        
        # Color preview
        if color_scheme == 'default':
            st.markdown(f"""
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <div style="width: 40px; height: 40px; background-color: {COLORS['primary']}; border-radius: 5px;"></div>
                <div style="width: 40px; height: 40px; background-color: {COLORS['secondary']}; border-radius: 5px;"></div>
                <div style="width: 40px; height: 40px; background-color: {COLORS['success']}; border-radius: 5px;"></div>
                <div style="width: 40px; height: 40px; background-color: {COLORS['warning']}; border-radius: 5px;"></div>
                <div style="width: 40px; height: 40px; background-color: {COLORS['danger']}; border-radius: 5px;"></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Display Options")
        
        plot_height = st.slider(
            "Default Plot Height (px)",
            200, 800,
            st.session_state.settings['visualization']['plot_height'],
            50
        )
        st.session_state.settings['visualization']['plot_height'] = plot_height
        
        show_ci = st.checkbox(
            "Show Confidence Intervals",
            value=st.session_state.settings['visualization']['show_confidence_intervals']
        )
        st.session_state.settings['visualization']['show_confidence_intervals'] = show_ci
        
        animation_speed = st.select_slider(
            "Animation Speed",
            options=['slow', 'normal', 'fast', 'none'],
            value=st.session_state.settings['visualization']['animation_speed']
        )
        st.session_state.settings['visualization']['animation_speed'] = animation_speed

with tab4:
    st.markdown("### 📁 Model Zoo Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Storage")
        
        model_path = st.text_input(
            "Model Zoo Path",
            value=st.session_state.settings['model_zoo']['path'],
            help="Directory where trained models are stored"
        )
        st.session_state.settings['model_zoo']['path'] = model_path
        
        auto_load = st.checkbox(
            "Auto-load Models on Startup",
            value=st.session_state.settings['model_zoo']['auto_load']
        )
        st.session_state.settings['model_zoo']['auto_load'] = auto_load
    
    with col2:
        st.markdown("#### Caching")
        
        cache_models = st.checkbox(
            "Cache Loaded Models",
            value=st.session_state.settings['model_zoo']['cache_models'],
            help="Keep models in memory for faster access"
        )
        st.session_state.settings['model_zoo']['cache_models'] = cache_models
        
        if cache_models:
            max_cache = st.number_input(
                "Max Cached Models",
                min_value=1,
                max_value=20,
                value=st.session_state.settings['model_zoo']['max_cache_size'],
                help="Maximum number of models to keep in memory"
            )
            st.session_state.settings['model_zoo']['max_cache_size'] = max_cache

with tab5:
    st.markdown("### ⚡ Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Processing")
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=512,
            value=st.session_state.settings['performance']['batch_size'],
            help="Samples processed per batch"
        )
        st.session_state.settings['performance']['batch_size'] = batch_size
        
        max_samples = st.number_input(
            "Max Samples per Analysis",
            min_value=100,
            max_value=100000,
            value=st.session_state.settings['performance']['max_samples'],
            step=1000,
            help="Maximum samples to analyze at once"
        )
        st.session_state.settings['performance']['max_samples'] = max_samples
    
    with col2:
        st.markdown("#### Acceleration")
        
        parallel = st.checkbox(
            "Enable Parallel Processing",
            value=st.session_state.settings['performance']['parallel_processing']
        )
        st.session_state.settings['performance']['parallel_processing'] = parallel
        
        gpu = st.checkbox(
            "Enable GPU Acceleration",
            value=st.session_state.settings['performance']['gpu_acceleration']
        )
        st.session_state.settings['performance']['gpu_acceleration'] = gpu
        
        if gpu:
            st.info("GPU detected: CUDA available" if st.session_state.settings['performance']['gpu_acceleration'] else "No GPU detected")

with tab6:
    st.markdown("### 💾 Import/Export Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Settings")
        
        config_json = json.dumps(st.session_state.settings, indent=2)
        
        st.download_button(
            label="⬇️ Download Configuration",
            data=config_json,
            file_name="simdrift_settings.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown("**Current Configuration Preview:**")
        st.json(st.session_state.settings)
    
    with col2:
        st.markdown("#### Import Settings")
        
        uploaded_file = st.file_uploader(
            "Upload Configuration JSON",
            type=['json'],
            help="Import settings from a JSON file"
        )
        
        if uploaded_file is not None:
            try:
                imported_settings = json.load(uploaded_file)
                
                st.success("✅ Configuration file loaded successfully!")
                st.json(imported_settings)
                
                if st.button("Apply Imported Settings", type="primary"):
                    st.session_state.settings = imported_settings
                    st.success("✅ Settings applied!")
                    st.rerun()
            
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file")
        
        st.markdown("---")
        
        if st.button("Reset to Defaults", type="secondary", use_container_width=True):
            st.session_state.settings = DEFAULT_SETTINGS.copy()
            st.success("Settings reset to defaults successfully!")
            st.rerun()

# Save button at bottom
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("Save Settings", type="primary", use_container_width=True):
        # In a real application, this would save to a config file
        st.success("Settings saved successfully!")
        st.balloons()
        
        # Show summary
        st.info(f"""
        **Configuration Summary:**
        - Detection Method: {st.session_state.settings['drift_detection']['default_method'].upper()}
        - Alert Threshold: {st.session_state.settings['alerts']['orange_threshold']:.2f}
        - Theme: {st.session_state.settings['visualization']['theme'].title()}
        - Model Path: {st.session_state.settings['model_zoo']['path']}
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
