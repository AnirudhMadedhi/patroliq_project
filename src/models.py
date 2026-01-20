import numpy as np

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
# 4️⃣ PCA (Dimensionality Reduction)
# -------------------------------------------------
def run_pca(X, n_components=3):
    X_scaled = scale_features(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


# -------------------------------------------------
# 5️⃣ UMAP (2D Visualization)
# -------------------------------------------------
def run_umap(X, n_components=2):
    X_scaled = scale_features(X)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    return reducer, X_umap
