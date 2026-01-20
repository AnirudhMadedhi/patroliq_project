import streamlit as st
import plotly.express as px
from mlflow_utils import load_latest_artifact

st.title("📉 Dimensionality Reduction")

pca_df = load_latest_artifact(
    "PatrolIQ_PCA",
    "PCA",
    "pca_transformed_data.csv"
)

umap_df = load_latest_artifact(
    "PatrolIQ_UMAP",
    "UMAP",
    "umap_coordinates.csv"
)

st.subheader("PCA Projection (2D)")
fig_pca = px.scatter(pca_df, x="PC1", y="PC2")
st.plotly_chart(fig_pca, use_container_width=True)

st.subheader("UMAP Projection (2D)")
fig_umap = px.scatter(umap_df, x="UMAP_1", y="UMAP_2")
st.plotly_chart(fig_umap, use_container_width=True)
