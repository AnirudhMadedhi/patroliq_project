import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

st.title("📉 Dimensionality Reduction & Crime Pattern Visualization")

# =================================================
# FILE PATHS (PROJECT ROOT)
# =================================================
PCA_VARIANCE_PATH = "pca_variance.csv"
PCA_DATA_PATH = "pca_transformed_data.csv"
UMAP_DATA_PATH = "umap_coordinates.csv"
GEO_LABELS_PATH = "geo_kmeans_labels.npy"
CRIME_DATA_PATH = "sample_data.csv"
PCA_FEATURES_PATH = "pca_top_features.csv"

# =================================================
# PCA VARIANCE RETENTION
# =================================================
st.markdown("### 📊 PCA Variance Retention")

if os.path.exists(PCA_VARIANCE_PATH):
    pca_variance = pd.read_csv(PCA_VARIANCE_PATH)
    st.dataframe(pca_variance)

    cumulative_variance = pca_variance["cumulative_variance"].iloc[-1]
    st.success(f"Cumulative variance retained: {cumulative_variance:.2%}")
else:
    st.warning("PCA variance file not found.")

# =================================================
# LOAD PCA & UMAP DATA
# =================================================
if not os.path.exists(PCA_DATA_PATH):
    st.error("PCA transformed data not found.")
    st.stop()

if not os.path.exists(UMAP_DATA_PATH):
    st.error("UMAP coordinates not found.")
    st.stop()

pca_df = pd.read_csv(PCA_DATA_PATH)
umap_df = pd.read_csv(UMAP_DATA_PATH)

# =================================================
# LOAD CRIME DATA
# =================================================
if not os.path.exists(CRIME_DATA_PATH):
    st.error("Crime dataset not found.")
    st.stop()

crime_df = pd.read_csv(CRIME_DATA_PATH)

# =================================================
# LOAD CLUSTER LABELS (OPTIONAL)
# =================================================
if os.path.exists(GEO_LABELS_PATH):
    geo_labels = np.load(GEO_LABELS_PATH)
    umap_df["Geo_Cluster"] = geo_labels
else:
    st.warning("Geographic cluster labels not found.")
    umap_df["Geo_Cluster"] = "Unknown"

# Attach crime type
umap_df["Crime_Type"] = crime_df["Primary Type"].values

# =================================================
# PCA VISUALIZATION
# =================================================
st.subheader("PCA Projection (2D)")

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    opacity=0.6,
    title="2D PCA Projection of Crime Data"
)

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

if os.path.exists(PCA_FEATURES_PATH):
    pca_top_features = pd.read_csv(PCA_FEATURES_PATH)
    st.dataframe(pca_top_features)
else:
    st.warning("PCA top-features file not found.")
