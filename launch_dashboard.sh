#!/bin/bash
# Launch script for SimDrift enhanced dashboard

echo "🔬 Launching SimDrift Enhanced Dashboard..."
echo ""

# Check if in correct directory
if [ ! -f "dashboard/enhanced_app.py" ]; then
    echo "❌ Error: Please run this script from the SimDrift root directory"
    exit 1
fi

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "   Consider activating SimDrift environment: conda activate SimDrift"
    echo ""
fi

# Launch dashboard
echo "✅ Starting dashboard on http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run dashboard/enhanced_app.py
