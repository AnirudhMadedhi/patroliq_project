import streamlit as st

st.set_page_config(page_title="PatrolIQ", layout="wide")

st.title("🚔 PatrolIQ – Crime Analytics Dashboard")

st.markdown("""
This application visualizes **trained ML model outputs** retrieved from MLflow:
- Geographic crime hotspot clustering
- Temporal crime pattern analysis
- PCA & UMAP dimensionality reduction
- Model performance monitoring
""")

st.success("Models are trained offline and consumed from MLflow artifacts.")
