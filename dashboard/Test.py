import streamlit as st

st.title("SimDrift - Health Check")
st.write("✅ Streamlit is working!")

try:
    import torch
    st.write(f"✅ PyTorch version: {torch.__version__}")
except Exception as e:
    st.error(f"❌ PyTorch error: {e}")

try:
    import medmnist
    st.write(f"✅ MedMNIST is working!")
except Exception as e:
    st.error(f"❌ MedMNIST error: {e}")

try:
    from pathlib import Path
    model_zoo = Path("model_zoo")
    if model_zoo.exists():
        models = list(model_zoo.glob("**/*.pth"))
        st.write(f"✅ Found {len(models)} model files")
    else:
        st.warning("⚠️ model_zoo directory not found")
except Exception as e:
    st.error(f"❌ File system error: {e}")
