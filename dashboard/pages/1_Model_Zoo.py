"""
Model Zoo Page - Compare and explore all trained models
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.model_manager import ModelManager

st.set_page_config(
    page_title="Model Zoo - SimDrift",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply same styling as main app
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">📊 Model Zoo</h1>
    <p class="hero-subtitle">
        Explore and Compare Pre-trained Models
    </p>
</div>
""", unsafe_allow_html=True)

# Load model manager
@st.cache_resource
def get_model_manager():
    return ModelManager('model_zoo')

manager = get_model_manager()

# Check if models exist
datasets = manager.get_available_datasets()

if not datasets:
    st.warning("No models found in model zoo. Run `python train_all_models.py` to train models.")
    st.markdown("""
    ### Training Instructions
    
    To populate the model zoo, run:
    ```bash
    # Quick test (1 model)
    python train_all_models.py --quick-test
    
    # Train all models (takes longer)
    python train_all_models.py
    
    # Train specific dataset
    python train_all_models.py --datasets pathmnist
    ```
    """)
    st.stop()

# Display summary
manager.print_summary()

# Get comparison data
df = manager.compare_models()

if df.empty:
    st.warning("No model data available")
    st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Leaderboard", "Performance Analysis", "Model Details", "Compare Models"])

with tab1:
    st.markdown("### Model Leaderboard")
    st.markdown("Compare all trained models ranked by performance metrics")
    st.markdown("---")
    
    # Sort options
    col1, col2 = st.columns([1, 3])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ['test_accuracy', 'test_f1', 'ece', 'parameters', 'training_time_minutes']
        )
    
    # Display leaderboard
    df_sorted = df.sort_values(sort_by, ascending=(sort_by in ['ece', 'parameters', 'training_time_minutes']))
    df_sorted.insert(0, 'rank', range(1, len(df_sorted) + 1))
    
    # Format dataframe for display
    df_display = df_sorted.copy()
    df_display['test_accuracy'] = df_display['test_accuracy'].apply(lambda x: f"{x:.2%}")
    df_display['test_f1'] = df_display['test_f1'].apply(lambda x: f"{x:.4f}")
    df_display['ece'] = df_display['ece'].apply(lambda x: f"{x:.4f}")
    df_display['parameters'] = df_display['parameters'].apply(lambda x: f"{x:,}")
    df_display['training_time_minutes'] = df_display['training_time_minutes'].apply(lambda x: f"{x:.1f}m")
    
    st.dataframe(
        df_display[['rank', 'dataset', 'architecture', 'test_accuracy', 'test_f1', 'ece', 'parameters']],
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.markdown("## 📈 Performance Analysis")
    
    # Accuracy by dataset
    fig1 = px.bar(
        df,
        x='dataset',
        y='test_accuracy',
        color='architecture',
        barmode='group',
        title='Test Accuracy by Dataset and Architecture',
        labels={'test_accuracy': 'Test Accuracy', 'dataset': 'Dataset'}
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Calibration vs Accuracy
    fig2 = px.scatter(
        df,
        x='test_accuracy',
        y='ece',
        size='parameters',
        color='architecture',
        hover_data=['dataset'],
        title='Calibration vs Accuracy (bubble size = parameters)',
        labels={'test_accuracy': 'Test Accuracy', 'ece': 'Expected Calibration Error'}
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Performance vs Efficiency
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.scatter(
            df,
            x='parameters',
            y='test_accuracy',
            color='dataset',
            size='test_f1',
            hover_data=['architecture'],
            title='Accuracy vs Model Size',
            labels={'parameters': 'Parameters', 'test_accuracy': 'Test Accuracy'}
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'])
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.scatter(
            df,
            x='training_time_minutes',
            y='test_accuracy',
            color='architecture',
            size='parameters',
            hover_data=['dataset'],
            title='Accuracy vs Training Time',
            labels={'training_time_minutes': 'Training Time (min)', 'test_accuracy': 'Test Accuracy'}
        )
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'])
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.markdown("## 📋 Model Details")
    
    # Select model
    col1, col2 = st.columns(2)
    with col1:
        selected_dataset = st.selectbox("Dataset", datasets)
    with col2:
        architectures = manager.get_available_architectures(selected_dataset)
        selected_arch = st.selectbox("Architecture", architectures)
    
    # Get model info
    try:
        model_info = manager.get_model_info(selected_dataset, selected_arch)
        metadata = manager.get_model_metadata(selected_dataset, selected_arch)
        
        # Display info
        st.markdown("### Model Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{model_info.test_accuracy:.2%}")
            st.metric("Parameters", f"{model_info.parameters:,}")
        
        with col2:
            st.metric("F1 Score", f"{model_info.test_f1:.4f}")
            st.metric("ECE", f"{model_info.ece:.4f}")
        
        with col3:
            st.metric("Training Time", f"{model_info.training_time_minutes:.1f} min")
            st.metric("Dataset", selected_dataset)
        
        # Training history
        if 'training' in metadata and 'history' in metadata:
            st.markdown("### Training History")
            
            history = metadata.get('history', {})
            if history:
                epochs = list(range(1, len(history.get('train_loss', [])) + 1))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=history.get('train_acc', []),
                    mode='lines+markers',
                    name='Train Accuracy',
                    line=dict(color=COLORS['primary'])
                ))
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=history.get('val_acc', []),
                    mode='lines+markers',
                    name='Val Accuracy',
                    line=dict(color=COLORS['secondary'])
                ))
                
                fig.update_layout(
                    title='Training Progress',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text']),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Download model
        st.markdown("### Download")
        st.info(f"📦 Model checkpoint: `{model_info.checkpoint_path}`")
        st.code(f"""
# Load model in Python
from models.model_manager import ModelManager

manager = ModelManager('model_zoo')
model = manager.load_pretrained('{selected_dataset}', '{selected_arch}')
        """, language='python')
        
    except Exception as e:
        st.error(f"Error loading model info: {e}")

with tab4:
    st.markdown("### Compare Models")
    st.markdown("Side-by-side comparison of model performance")
    st.markdown("---")
    
    # Select models to compare
    st.markdown("### Select Models to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset1 = st.selectbox("Dataset 1", datasets, key='ds1')
        archs1 = manager.get_available_architectures(dataset1)
        arch1 = st.selectbox("Architecture 1", archs1, key='arch1')
    
    with col2:
        dataset2 = st.selectbox("Dataset 2", datasets, key='ds2')
        archs2 = manager.get_available_architectures(dataset2)
        arch2 = st.selectbox("Architecture 2", archs2, key='arch2')
    
    if st.button("Compare"):
        try:
            info1 = manager.get_model_info(dataset1, arch1)
            info2 = manager.get_model_info(dataset2, arch2)
            
            # Comparison table
            comparison_data = {
                'Metric': ['Test Accuracy', 'F1 Score', 'ECE', 'Parameters', 'Training Time (min)'],
                f'{dataset1}/{arch1}': [
                    f"{info1.test_accuracy:.2%}",
                    f"{info1.test_f1:.4f}",
                    f"{info1.ece:.4f}",
                    f"{info1.parameters:,}",
                    f"{info1.training_time_minutes:.1f}"
                ],
                f'{dataset2}/{arch2}': [
                    f"{info2.test_accuracy:.2%}",
                    f"{info2.test_f1:.4f}",
                    f"{info2.ece:.4f}",
                    f"{info2.parameters:,}",
                    f"{info2.training_time_minutes:.1f}"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Visual comparison
            metrics = ['test_accuracy', 'test_f1', 'ece']
            values1 = [info1.test_accuracy, info1.test_f1, info1.ece]
            values2 = [info2.test_accuracy, info2.test_f1, info2.ece]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=metrics,
                y=values1,
                name=f'{dataset1}/{arch1}',
                marker_color=COLORS['primary']
            ))
            fig.add_trace(go.Bar(
                x=metrics,
                y=values2,
                name=f'{dataset2}/{arch2}',
                marker_color=COLORS['secondary']
            ))
            
            fig.update_layout(
                title='Performance Comparison',
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text']),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error comparing models: {e}")

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
