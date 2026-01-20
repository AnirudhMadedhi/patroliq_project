import streamlit as st
from mlflow_utils import load_latest_artifact

st.title("⏰ Temporal Crime Pattern Analysis")

summary = load_latest_artifact(
    "PatrolIQ_Temporal_Clustering",
    "Temporal_KMeans_Full",
    "temporal_cluster_summary.csv"
)

peak_hours = load_latest_artifact(
    "PatrolIQ_Temporal_Clustering",
    "Temporal_KMeans_Full",
    "peak_crime_hours.csv"
)

peak_months = load_latest_artifact(
    "PatrolIQ_Temporal_Clustering",
    "Temporal_KMeans_Full",
    "peak_crime_months.csv"
)

profiles = load_latest_artifact(
    "PatrolIQ_Temporal_Clustering",
    "Temporal_KMeans_Full",
    "temporal_crime_profiles.csv"
)

st.subheader("Temporal Cluster Behavior Summary")
st.dataframe(summary)

st.subheader("Peak Crime Hours")
st.dataframe(peak_hours)

st.subheader("Peak Crime Months")
st.dataframe(peak_months)

st.subheader("Temporal Crime Profiles")
st.dataframe(profiles)
