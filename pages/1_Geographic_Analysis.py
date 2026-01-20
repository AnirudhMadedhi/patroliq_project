import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from mlflow_utils import load_latest_artifact

st.title("📍 Geographic Crime Hotspot Analysis")

df = pd.read_csv("data/sample_data.csv").dropna(subset=["Latitude", "Longitude"])

labels = load_latest_artifact(
    "PatrolIQ_Clustering_Models",
    "KMeans",
    "geo_kmeans_labels.npy"
)

centroids = load_latest_artifact(
    "PatrolIQ_Clustering_Models",
    "KMeans",
    "geo_kmeans_centroids.csv"
)

df["Cluster"] = labels

m = folium.Map(
    location=[df.Latitude.mean(), df.Longitude.mean()],
    zoom_start=11
)

colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=2,
        color=colors[row["Cluster"] % len(colors)],
        fill=True
    ).add_to(m)

for _, row in centroids.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)

folium_static(m)

st.markdown("---")
st.subheader("🌳 Hierarchical Clustering – Crime Zone Relationships")

dendrogram_path = load_latest_artifact(
    experiment_name="PatrolIQ_Clustering_Models",
    run_name="Geographic_Clustering",
    artifact_path="dendrogram.png"
)

st.image(
    dendrogram_path,
    caption="Hierarchical Dendrogram Showing Relationships Between Crime Zones",
    use_column_width=True
)

