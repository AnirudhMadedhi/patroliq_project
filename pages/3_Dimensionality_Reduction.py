import streamlit as st
import plotly.express as px
from mlflow_utils import load_latest_artifact

st.title("📉 Dimensionality Reduction")

# =========================
# ADDITION 1: PCA VARIANCE
# =========================
st.markdown("### 📊 PCA Variance Retention")

try:
    pca_variance = load_latest_artifact(
        "PatrolIQ_PCA",
        "PCA",
        "pca_variance.csv"
    )

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
        "Ensure variance is logged during PCA training."
    )

# =========================
# EXISTING CODE (UNCHANGED)
# =========================
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

# =========================
# ADDITION 2: PCA 3D PLOT
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
    st.info("PC3 not available. PCA was generated in 2D.")

# =========================
# EXISTING CODE (UNCHANGED)
# =========================
st.subheader("UMAP Projection (2D)")
fig_umap = px.scatter(umap_df, x="UMAP_1", y="UMAP_2")
st.plotly_chart(fig_umap, use_container_width=True)

# =========================
# ADDITION 3: FEATURE IMPORTANCE (GLOBAL)
# =========================
# st.markdown("### 🔍 Features Driving Crime Patterns")

# try:
#     feature_importance = load_latest_artifact(
#         "PatrolIQ_PCA",
#         "PCA",
#         "pca_feature_importance.csv"
#     )

#     top_features = (
#         feature_importance
#         .sort_values(by="importance", ascending=False)
#         .head(10)
#     )

#     st.dataframe(top_features)

#     fig_feat = px.bar(
#         top_features,
#         x="importance",
#         y="feature",
#         orientation="h",
#         title="Top Features Influencing Crime Patterns"
#     )

#     st.plotly_chart(fig_feat, use_container_width=True)

# except Exception:
#     st.warning(
#         "Feature importance file not found. "
#         "Ensure PCA loadings are logged during training."
#     )

# =========================
# ADDITION 4: WHAT EACH PC REPRESENTS
# =========================
st.markdown("### 🧠 Interpretation of Principal Components")

try:
    pca_top_features = load_latest_artifact(
        "PatrolIQ_PCA",
        "PCA",
        "pca_top_features.csv"
    )

    st.write(
        "Top original features contributing to each principal component:"
    )
    st.dataframe(pca_top_features)

except Exception:
    st.warning(
        "PCA top-features file not found. "
        "Ensure pca_top_features.csv is logged."
    )
