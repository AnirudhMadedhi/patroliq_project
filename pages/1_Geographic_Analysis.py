import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static

st.title("📍 Geographic Crime Hotspot Analysis")

# -------------------------------------------------
# Load base dataset
# -------------------------------------------------
df = pd.read_csv("data/sample_data.csv").dropna(
    subset=["Latitude", "Longitude"]
)

# -------------------------------------------------
# Load pre-generated clustering artifacts (NO MLflow)
# -------------------------------------------------
labels = np.load("geo_kmeans_labels.npy")
centroids = pd.read_csv("geo_kmeans_centroids.csv")

df["Cluster"] = labels

# -------------------------------------------------
# Folium Map – Crime Hotspots
# -------------------------------------------------
m = folium.Map(
    location=[df.Latitude.mean(), df.Longitude.mean()],
    zoom_start=11
)

colors = [
    "red", "blue", "green", "purple",
    "orange", "darkred", "cadetblue"
]

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=2,
        color=colors[row["Cluster"] % len(colors)],
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# Plot cluster centroids
for _, row in centroids.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)

folium_static(m)

# -------------------------------------------------
# Hierarchical Clustering – Dendrogram
# -------------------------------------------------
st.markdown("---")
st.subheader("🌳 Hierarchical Clustering – Crime Zone Relationships")

st.image(
    "dendrogram.png",
    caption="Hierarchical Dendrogram Showing Relationships Between Crime Zones",
    use_column_width=True
)
