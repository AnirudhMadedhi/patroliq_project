import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

from models import (
    run_kmeans,
    run_dbscan,
    run_hierarchical,
    run_pca,
    run_umap
)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv("data/sample_data.csv")
df = df.dropna(subset=["Latitude", "Longitude"])

# =================================================
# STEP 4 — GEOGRAPHIC CLUSTERING
# =================================================
FEATURES = ["Latitude", "Longitude"]
X = df[FEATURES]

mlflow.set_experiment("PatrolIQ_Clustering_Models")

with mlflow.start_run(run_name="Geographic_Clustering"):

    # ---------------- KMEANS ----------------
    with mlflow.start_run(run_name="KMeans", nested=True):
        model, labels, sil = run_kmeans(X, n_clusters=5)
        db = davies_bouldin_score(X, labels)

        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", 5)
        mlflow.log_metric("silhouette", sil)
        mlflow.log_metric("davies_bouldin", db)

        mlflow.sklearn.log_model(model, "model")

        # 🔹 LOG OUTPUTS FOR STREAMLIT
        np.save("geo_kmeans_labels.npy", labels)
        mlflow.log_artifact("geo_kmeans_labels.npy")

        centroids_df = pd.DataFrame(
            model.cluster_centers_,
            columns=["Latitude", "Longitude"]
        )
        centroids_df.to_csv("geo_kmeans_centroids.csv", index=False)
        mlflow.log_artifact("geo_kmeans_centroids.csv")

    # ---------------- DBSCAN ----------------
    with mlflow.start_run(run_name="DBSCAN", nested=True):
        model, labels, sil = run_dbscan(X)
        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_metric("silhouette", sil)

    # ---------------- HIERARCHICAL ----------------
    with mlflow.start_run(run_name="Hierarchical", nested=True):
        model, labels, sil = run_hierarchical(X, n_clusters=5)
        mlflow.log_param("algorithm", "Hierarchical")
        mlflow.log_param("n_clusters", 5)
        mlflow.log_metric("silhouette", sil)

    # ---------------- ELBOW METHOD ----------------
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.plot(range(2, 11), inertias)
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig("elbow.png")
    plt.close()
    mlflow.log_artifact("elbow.png")

    # ---------------- DENDROGRAM ----------------
    Z = linkage(X, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title("Hierarchical Dendrogram")
    plt.savefig("dendrogram.png")
    plt.close()
    mlflow.log_artifact("dendrogram.png")

# =================================================
# TEMPORAL FEATURE ENGINEERING
# =================================================
df["Date"] = pd.to_datetime(df["Date"])
df["Hour"] = df["Date"].dt.hour
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# =================================================
# STEP 4 — TEMPORAL PATTERN CLUSTERING (FULLY LOGGED)
# =================================================
mlflow.set_experiment("PatrolIQ_Temporal_Clustering")

with mlflow.start_run(run_name="Temporal_KMeans_Full"):
    X_time = df[["Hour", "DayOfWeek", "Month"]]
    scaler = StandardScaler()
    X_time_scaled = scaler.fit_transform(X_time)

    kmeans_time = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_time = kmeans_time.fit_predict(X_time_scaled)
    df["Temporal_Cluster"] = labels_time

    sil_score = silhouette_score(X_time_scaled, labels_time)

    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("features", "Hour, DayOfWeek, Month")
    mlflow.log_param("n_clusters", 4)
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.sklearn.log_model(kmeans_time, "temporal_kmeans_model")

    np.save("temporal_cluster_labels.npy", labels_time)
    mlflow.log_artifact("temporal_cluster_labels.npy")

    df.groupby("Temporal_Cluster")[["Hour", "DayOfWeek", "Month"]] \
      .mean().round(2).to_csv("temporal_cluster_summary.csv")
    mlflow.log_artifact("temporal_cluster_summary.csv")

    df["Hour"].value_counts().head(5).to_csv("peak_crime_hours.csv")
    mlflow.log_artifact("peak_crime_hours.csv")

    df["Month"].value_counts().head(3).to_csv("peak_crime_months.csv")
    mlflow.log_artifact("peak_crime_months.csv")

    df.groupby(["Temporal_Cluster", "Primary Type"]) \
      .size().reset_index(name="Count") \
      .to_csv("temporal_crime_profiles.csv", index=False)
    mlflow.log_artifact("temporal_crime_profiles.csv")

# =================================================
# STEP 5 — PCA
# =================================================
NUMERIC_FEATURES = df.select_dtypes(include="number").columns
X_num = df[NUMERIC_FEATURES]

mlflow.set_experiment("PatrolIQ_PCA")

with mlflow.start_run(run_name="PCA"):
    pca, X_pca = run_pca(X_num, n_components=3)

    mlflow.log_metric("explained_variance", pca.explained_variance_ratio_.sum())

    pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"]) \
      .to_csv("pca_transformed_data.csv", index=False)
    mlflow.log_artifact("pca_transformed_data.csv")

    pd.Series(abs(pca.components_[0]), index=NUMERIC_FEATURES) \
      .sort_values(ascending=False) \
      .to_csv("pca_feature_importance.csv")
    mlflow.log_artifact("pca_feature_importance.csv")

# =================================================
# STEP 5 — UMAP
# =================================================
mlflow.set_experiment("PatrolIQ_UMAP")

with mlflow.start_run(run_name="UMAP"):
    reducer, X_umap = run_umap(X_num)

    pd.DataFrame(X_umap, columns=["UMAP_1", "UMAP_2"]) \
      .to_csv("umap_coordinates.csv", index=False)
    mlflow.log_artifact("umap_coordinates.csv")

    plt.scatter(X_umap[:, 0], X_umap[:, 1], s=2)
    plt.title("UMAP Crime Pattern Visualization")
    plt.savefig("umap.png")
    plt.close()
    mlflow.log_artifact("umap.png")
