import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap


# -------------------------------------------------
# Common utility: Feature scaling
# -------------------------------------------------
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# -------------------------------------------------
# 1️⃣ K-MEANS CLUSTERING
# -------------------------------------------------
def run_kmeans(X, n_clusters=5):
    X_scaled = scale_features(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return model, labels, score


# -------------------------------------------------
# 2️⃣ DBSCAN CLUSTERING
# -------------------------------------------------
def run_dbscan(X, eps=0.3, min_samples=10):
    X_scaled = scale_features(X)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    unique_labels = set(labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        score = silhouette_score(X_scaled, labels)
    else:
        score = -1

    return model, labels, score


# -------------------------------------------------
# 3️⃣ HIERARCHICAL CLUSTERING
# -------------------------------------------------
def run_hierarchical(X, n_clusters=5):
    X_scaled = scale_features(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return model, labels, score

# -------------------------------------------------
# 4️⃣ PCA (Dimensionality Reduction + INTERPRETATION)
# -------------------------------------------------
def run_pca(X, variance_threshold=0.75):
    """
    Performs PCA retaining a fixed proportion of variance (70–80%),
    and logs:
    - PCA transformed data
    - Explained & cumulative variance
    - Feature importance ranking
    - Top original features per principal component
    """

    X_scaled = scale_features(X)

    # ---- PCA FIT (VARIANCE-BASED) ----
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ===============================
    # PCA VARIANCE
    # ===============================
    explained_variance = pca.explained_variance_ratio_

    pca_variance = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "explained_variance": explained_variance,
        "cumulative_variance": np.cumsum(explained_variance)
    })

    pca_variance.to_csv("pca_variance.csv", index=False)

    # ===============================
    # PCA FEATURE IMPORTANCE (GLOBAL)
    # ===============================
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    feature_importance = (
        loadings.abs()
        .mul(explained_variance, axis=1)
        .sum(axis=1)
        .reset_index()
    )

    feature_importance.columns = ["feature", "importance"]
    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False
    )

    feature_importance.to_csv(
        "pca_feature_importance.csv",
        index=False
    )

    # =================================================
    # TOP ORIGINAL FEATURES DRIVING EACH PC
    # =================================================
    top_features_per_pc = []

    for pc in loadings.columns:
        top_feats = (
            loadings[pc]
            .abs()
            .sort_values(ascending=False)
            .head(3)
            .index
            .tolist()
        )

        top_features_per_pc.append({
            "component": pc,
            "top_features": ", ".join(top_feats)
        })

    top_features_df = pd.DataFrame(top_features_per_pc)
    top_features_df.to_csv(
        "pca_top_features.csv",
        index=False
    )

    return pca, X_pca



# # -------------------------------------------------
# # 4️⃣ PCA (Dimensionality Reduction + INTERPRETATION)
# # -------------------------------------------------
# def run_pca(X, n_components=3):
#     """
#     Performs PCA with 2–3 components, logs:
#     - PCA transformed data
#     - Explained & cumulative variance (70–80% proof)
#     - Feature importance ranking
#     - Top original features per principal component
#     """

#     X_scaled = scale_features(X)

#     # ---- PCA FIT ----
#     pca = PCA(n_components=n_components, random_state=42)
#     X_pca = pca.fit_transform(X_scaled)

#     # ===============================
#     # PCA VARIANCE (70–80% REQUIREMENT)
#     # ===============================
#     explained_variance = pca.explained_variance_ratio_

#     pca_variance = pd.DataFrame({
#         "component": [f"PC{i+1}" for i in range(len(explained_variance))],
#         "explained_variance": explained_variance,
#         "cumulative_variance": np.cumsum(explained_variance)
#     })

#     pca_variance.to_csv("pca_variance.csv", index=False)

#     # ===============================
#     # PCA FEATURE IMPORTANCE (GLOBAL)
#     # ===============================
#     loadings = pd.DataFrame(
#         pca.components_.T,
#         index=X.columns,
#         columns=[f"PC{i+1}" for i in range(pca.n_components_)]
#     )

#     feature_importance = (
#         loadings.abs()
#         .mul(explained_variance, axis=1)
#         .sum(axis=1)
#         .reset_index()
#     )

#     feature_importance.columns = ["feature", "importance"]
#     feature_importance = feature_importance.sort_values(
#         by="importance", ascending=False
#     )

#     feature_importance.to_csv(
#         "pca_feature_importance.csv",
#         index=False
#     )

#     # =================================================
#     # TOP ORIGINAL FEATURES DRIVING EACH PC (INTERPRET)
#     # =================================================
#     top_features_per_pc = []

#     for pc in loadings.columns:
#         top_feats = (
#             loadings[pc]
#             .abs()
#             .sort_values(ascending=False)
#             .head(3)
#             .index
#             .tolist()
#         )

#         top_features_per_pc.append({
#             "component": pc,
#             "top_features": ", ".join(top_feats)
#         })

#     top_features_df = pd.DataFrame(top_features_per_pc)
#     top_features_df.to_csv(
#         "pca_top_features.csv",
#         index=False
#     )

#     return pca, X_pca


# -------------------------------------------------
# 5️⃣ UMAP (2D Visualization)
# -------------------------------------------------
def run_umap(X, n_components=2):
    X_scaled = scale_features(X)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    return reducer, X_umap
