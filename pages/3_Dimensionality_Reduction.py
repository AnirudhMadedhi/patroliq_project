import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📉 Dimensionality Reduction")

# =========================
# PCA VARIANCE
# =========================
st.markdown("### 📊 PCA Variance Retention")

try:
    pca_variance = pd.read_csv("pca_variance.csv")

    st.write("Explained and Cumulative Variance:")
    st.dataframe(pca_variance)

    cumulative_variance = pca_variance["cumulative_variance"].iloc[-1]

    st.success(
        f"Cumulative variance retained by PCA: "
        f"{cumulative_variance:.2%}"
    )

except Exception:
    st.warning(
        "PCA variance file not found. "
        "Ensure pca_variance.csv exists in the project root."
    )

# =========================
# LOAD REDUCED DATA
# =========================
try:
    pca_df = pd.read_csv("pca_transformed_data.csv")
except Exception:
    st.error("pca_transformed_data.csv not found.")
    st.stop()

try:
    umap_df = pd.read_csv("umap_coordinates.csv")
except Exception:
    st.error("umap_coordinates.csv not found.")
    st.stop()

# =========================
# PCA 2D PROJECTION
# =========================
st.subheader("PCA Projection (2D)")
fig_pca = px.scatter(pca_df, x="PC1", y="PC2")
st.plotly_chart(fig_pca, use_container_width=True)

# =========================
# PCA 3D PROJECTION
# =========================
if {"PC1", "PC2", "PC3"}.issubset(pca_df.columns):

    st.subheader("PCA Projection (3D)")

    fig_pca_3d = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        opacity=0.7
    )

    st.plotly_chart(fig_pca_3d, use_container_width=True)

else:
    st.info("PC3 not available. PCA was generated with fewer components.")

# =========================
# UMAP 2D PROJECTION
# =========================
st.subheader("UMAP Projection (2D)")
fig_umap = px.scatter(umap_df, x="UMAP_1", y="UMAP_2")
st.plotly_chart(fig_umap, use_container_width=True)

# =========================
# PCA INTERPRETATION
# =========================
st.markdown("### 🧠 Interpretation of Principal Components")

try:
    pca_top_features = pd.read_csv("pca_top_features.csv")

    st.write(
        "Top original features contributing to each principal component:"
    )
    st.dataframe(pca_top_features)

except Exception:
    st.warning(
        "pca_top_features.csv not found. "
        "Ensure it exists in the project root."
    )
