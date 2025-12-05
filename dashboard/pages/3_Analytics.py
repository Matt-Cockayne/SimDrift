"""
Analytics Dashboard - Deep Metrics Analysis

Comprehensive analytics for drift detection, model performance,
and temporal analysis with statistical tests and data export.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.drift_detectors import DriftDetector
from data.medmnist_loader import MedMNISTLoader
from data.drift_generators import DriftGenerator

st.set_page_config(
    page_title="SimDrift - Analytics",
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
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['surface']}, {COLORS['background']});
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_dataset(dataset_name: str):
    return MedMNISTLoader(dataset_name=dataset_name, download=True)

def generate_time_series_data(n_points=100):
    """Generate simulated time-series drift metrics"""
    dates = [datetime.now() - timedelta(hours=i) for i in range(n_points)][::-1]
    
    # Simulate gradual drift over time
    drift_scores = []
    accuracy = []
    base_drift = 0.1
    base_acc = 0.95
    
    for i in range(n_points):
        # Gradual drift increase with noise
        drift = base_drift + (i / n_points) * 0.4 + np.random.normal(0, 0.05)
        drift_scores.append(max(0, min(1, drift)))
        
        # Accuracy decreases as drift increases
        acc = base_acc - (i / n_points) * 0.2 + np.random.normal(0, 0.02)
        accuracy.append(max(0.5, min(1.0, acc)))
    
    return pd.DataFrame({
        'timestamp': dates,
        'drift_score': drift_scores,
        'accuracy': accuracy,
        'samples_processed': np.random.randint(50, 200, n_points)
    })

st.title("📊 Analytics Dashboard")
st.markdown("### Deep dive into drift metrics and model performance")
st.markdown("Comprehensive analysis of drift detection, temporal trends, and statistical tests")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## Analysis Settings")
    st.markdown("---")
    
    time_window = st.selectbox(
        "Time Window",
        ['Last Hour', 'Last 6 Hours', 'Last 24 Hours', 'Last 7 Days'],
        index=2
    )
    
    window_map = {
        'Last Hour': 12,
        'Last 6 Hours': 72,
        'Last 24 Hours': 100,
        'Last 7 Days': 168
    }
    n_points = window_map[time_window]
    
    detection_method = st.selectbox(
        "Detection Method",
        ['psi', 'ks_test', 'chi_square', 'mmd', 'wasserstein']
    )
    
    threshold = st.slider(
        "Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Drift score threshold for alerts"
    )
    
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_resource.clear()
        st.rerun()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistical Tests", "Trends", "Export"])

with tab1:
    st.markdown("### Performance Overview")
    st.markdown("Monitor drift scores and model performance over time")
    st.markdown("---")
    
    # Generate time series data
    df = generate_time_series_data(n_points)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_drift = df['drift_score'].iloc[-1]
        prev_drift = df['drift_score'].iloc[-10]
        st.metric(
            "Current Drift Score",
            f"{current_drift:.3f}",
            f"{current_drift - prev_drift:+.3f}",
            delta_color="inverse"
        )
    
    with col2:
        current_acc = df['accuracy'].iloc[-1]
        prev_acc = df['accuracy'].iloc[-10]
        st.metric(
            "Current Accuracy",
            f"{current_acc:.1%}",
            f"{current_acc - prev_acc:+.2%}"
        )
    
    with col3:
        total_samples = df['samples_processed'].sum()
        st.metric(
            "Total Samples",
            f"{total_samples:,}"
        )
    
    with col4:
        alerts = (df['drift_score'] > threshold).sum()
        st.metric(
            "Drift Alerts",
            alerts,
            delta_color="off"
        )
    
    # Time series plot
    st.markdown("#### Drift Score Over Time")
    st.markdown("Track drift detection scores with alert thresholds")
    st.markdown("---")
    
    fig = go.Figure()
    
    # Drift score line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['drift_score'],
        mode='lines+markers',
        name='Drift Score',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4)
    ))
    
    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"Alert Threshold ({threshold})"
    )
    
    # Alert regions
    alert_mask = df['drift_score'] > threshold
    if alert_mask.any():
        fig.add_trace(go.Scatter(
            x=df[alert_mask]['timestamp'],
            y=df[alert_mask]['drift_score'],
            mode='markers',
            name='Alerts',
            marker=dict(
                color=COLORS['danger'],
                size=10,
                symbol='x'
            )
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        height=400,
        xaxis_title="Time",
        yaxis_title="Drift Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy vs Drift correlation
    st.markdown("#### Accuracy vs Drift Correlation")
    st.markdown("Analyze the relationship between drift and model performance")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['drift_score'],
            y=df['accuracy'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['drift_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Drift<br>Score")
            ),
            text=[f"Time: {t.strftime('%H:%M')}" for t in df['timestamp']],
            hovertemplate='Drift: %{x:.3f}<br>Accuracy: %{y:.1%}<br>%{text}<extra></extra>'
        ))
        
        # Trend line
        z = np.polyfit(df['drift_score'], df['accuracy'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['drift_score'].min(), df['drift_score'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            line=dict(color=COLORS['danger'], dash='dash', width=2),
            name='Trend'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=400,
            xaxis_title="Drift Score",
            yaxis_title="Accuracy"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation statistics
        correlation = np.corrcoef(df['drift_score'], df['accuracy'])[0, 1]
        
        st.markdown("**Correlation Analysis**")
        st.metric("Pearson Correlation", f"{correlation:.3f}")
        
        if abs(correlation) > 0.7:
            st.error("Strong negative correlation detected")
        elif abs(correlation) > 0.4:
            st.warning("Moderate negative correlation")
        else:
            st.success("Weak correlation")
        
        st.markdown("---")
        
        st.markdown("**Summary Statistics**")
        st.metric("Mean Drift", f"{df['drift_score'].mean():.3f}")
        st.metric("Std Dev", f"{df['drift_score'].std():.3f}")
        st.metric("Max Drift", f"{df['drift_score'].max():.3f}")

with tab2:
    st.markdown("### Statistical Drift Tests")
    st.markdown("Run hypothesis tests to detect distribution shifts")
    st.markdown("---")
    
    st.info("Run statistical hypothesis tests to detect drift between reference and current distributions")
    
    # Dataset selection
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            ['pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist'],
            key='stats_dataset'
        )
    
    with col2:
        drift_type = st.selectbox(
            "Drift Type",
            ['brightness', 'contrast', 'blur', 'noise'],
            key='stats_drift'
        )
    
    drift_severity = st.slider(
        "Drift Severity",
        0.0, 1.0, 0.5, 0.05,
        key='stats_severity'
    )
    
    if st.button("Run Statistical Tests", type="primary"):
        with st.spinner("Running statistical tests..."):
            # Load data
            loader = load_dataset(dataset_name)
            images, _ = loader.get_numpy_data('test')
            reference = images[:500]
            
            # Apply drift
            generator = DriftGenerator(seed=42)
            method = getattr(generator, f'_apply_{drift_type}')
            current = method(reference.copy(), drift_severity)
            
            # Run tests
            detector = DriftDetector()
            
            # Flatten for statistical tests
            ref_flat = reference.reshape(len(reference), -1).mean(axis=1)
            cur_flat = current.reshape(len(current), -1).mean(axis=1)
            
            # Run multiple tests
            results = {}
            
            # PSI
            psi_score, _ = detector.detect_drift(reference, current, method='psi')
            results['PSI'] = {'score': psi_score, 'threshold': 0.2}
            
            # KS Test
            ks_score, _ = detector.detect_drift(reference, current, method='ks_test')
            results['KS Test'] = {'score': ks_score, 'threshold': 0.05}
            
            # Chi-Square
            chi_score, _ = detector.detect_drift(reference, current, method='chi_square')
            results['Chi-Square'] = {'score': chi_score, 'threshold': 0.05}
            
            # MMD
            mmd_score, _ = detector.detect_drift(reference, current, method='mmd')
            results['MMD'] = {'score': mmd_score, 'threshold': 0.1}
            
            # Wasserstein
            wass_score, _ = detector.detect_drift(reference, current, method='wasserstein')
            results['Wasserstein'] = {'score': wass_score, 'threshold': 0.1}
            
            st.session_state.test_results = results
    
    if 'test_results' in st.session_state:
        st.markdown("#### Test Results")
        
        results_df = pd.DataFrame([
            {
                'Test': name,
                'Score': data['score'],
                'Threshold': data['threshold'],
                'Drift Detected': '🔴 Yes' if data['score'] > data['threshold'] else '🟢 No'
            }
            for name, data in st.session_state.test_results.items()
        ])
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = go.Figure()
        
        test_names = list(st.session_state.test_results.keys())
        scores = [st.session_state.test_results[t]['score'] for t in test_names]
        thresholds = [st.session_state.test_results[t]['threshold'] for t in test_names]
        
        fig.add_trace(go.Bar(
            x=test_names,
            y=scores,
            name='Test Score',
            marker_color=COLORS['primary']
        ))
        
        fig.add_trace(go.Scatter(
            x=test_names,
            y=thresholds,
            mode='markers+lines',
            name='Threshold',
            marker=dict(size=12, symbol='line-ew-open', color=COLORS['danger']),
            line=dict(color=COLORS['danger'], dash='dash')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=400,
            xaxis_title="Test Method",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Temporal Trends")
    st.markdown("Analyze how metrics evolve over time")
    st.markdown("---")
    
    # Multi-metric comparison
    df = generate_time_series_data(n_points)
    
    st.markdown("#### Multi-Metric Timeline")
    
    metrics_to_plot = st.multiselect(
        "Select Metrics",
        ['drift_score', 'accuracy', 'samples_processed'],
        default=['drift_score', 'accuracy']
    )
    
    if metrics_to_plot:
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            # Normalize for comparison
            if metric == 'samples_processed':
                normalized = df[metric] / df[metric].max()
            else:
                normalized = df[metric]
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=normalized,
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            height=400,
            xaxis_title="Time",
            yaxis_title="Normalized Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution over time
    st.markdown("#### Distribution Evolution")
    
    # Create hourly bins
    df['hour'] = df['timestamp'].dt.hour
    
    fig = go.Figure()
    
    for hour in sorted(df['hour'].unique())[:10]:  # Show first 10 hours
        hour_data = df[df['hour'] == hour]['drift_score']
        
        fig.add_trace(go.Box(
            y=hour_data,
            name=f"Hour {hour}",
            boxmean='sd'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        height=400,
        xaxis_title="Time Period",
        yaxis_title="Drift Score Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Export Data")
    st.markdown("Download analytics data in various formats")
    st.markdown("---")
    
    df = generate_time_series_data(n_points)
    
    st.markdown("#### 📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Export as CSV")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"drift_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Export as JSON")
        json_str = df.to_json(orient='records', date_format='iso', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"drift_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

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
