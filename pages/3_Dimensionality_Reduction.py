import streamlit as st
import pandas as pd
import plotly.express as px
from mlflow_utils import load_latest_artifact

st.title("📉 Dimensionality Reduction & Crime Pattern Visualization")

# =================================================
# PCA VARIANCE RETENTION
# =================================================
st.markdown("### 📊 PCA Variance Retention")

try:
    pca_variance = load_latest_artifact(
        "PatrolIQ_PCA",
        "PCA",
        "pca_variance.csv"
    )

    st.dataframe(pca_variance)

    cumulative_variance = pca_variance["cumulative_variance"].iloc[-1]
    st.success(f"Cumulative variance retained: {cumulative_variance:.2%}")

except Exception:
    st.warning("PCA variance file not found.")

# =================================================
# LOAD DATA
# =================================================
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

crime_df = pd.read_csv("data/sample_data.csv")

# =================================================
# LOAD CLUSTER LABELS
# =================================================
try:
    geo_labels = load_latest_artifact(
        "PatrolIQ_Clustering_Models",
        "KMeans",
        "geo_kmeans_labels.npy"
    )

    umap_df["Geo_Cluster"] = geo_labels

except Exception:
    st.warning("Geographic cluster labels not found.")
    umap_df["Geo_Cluster"] = "Unknown"

# Attach crime type for interpretation
umap_df["Crime_Type"] = crime_df["Primary Type"].values

# =================================================
# PCA VISUALIZATION
# =================================================
st.subheader("PCA Projection (2D)")
fig_pca = px.scatter(pca_df, x="PC1", y="PC2", opacity=0.6)
st.plotly_chart(fig_pca, use_container_width=True)

# =================================================
# UMAP — CLUSTER SEPARATION
# =================================================
st.markdown("## 🔍 UMAP Crime Pattern Visualization")

st.subheader("UMAP – Geographic Crime Clusters")
fig_umap_cluster = px.scatter(
    umap_df,
    x="UMAP_1",
    y="UMAP_2",
    color="Geo_Cluster",
    title="UMAP Projection Colored by Geographic Crime Clusters",
    opacity=0.7
)
st.plotly_chart(fig_umap_cluster, use_container_width=True)

# =================================================
# UMAP — CRIME TYPE SEPARATION
# =================================================
st.subheader("UMAP – Crime Type Separation")
fig_umap_type = px.scatter(
    umap_df,
    x="UMAP_1",
    y="UMAP_2",
    color="Crime_Type",
    title="UMAP Projection Colored by Crime Type",
    opacity=0.6
)
st.plotly_chart(fig_umap_type, use_container_width=True)

# =================================================
# PCA INTERPRETATION
# =================================================
st.markdown("### 🧠 Interpretation of Principal Components")

try:
    pca_top_features = load_latest_artifact(
        "PatrolIQ_PCA",
        "PCA",
        "pca_top_features.csv"
    )

    st.dataframe(pca_top_features)

except Exception:
    st.warning("PCA top-features file not found.")
