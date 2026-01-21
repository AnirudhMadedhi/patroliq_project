import streamlit as st
import pandas as pd

st.title("⏰ Temporal Crime Pattern Analysis")

# -------------------------------------------------
# Load pre-generated temporal analysis artifacts
# -------------------------------------------------
summary = pd.read_csv("temporal_cluster_summary.csv")
peak_hours = pd.read_csv("peak_crime_hours.csv")
peak_months = pd.read_csv("peak_crime_months.csv")
profiles = pd.read_csv("temporal_crime_profiles.csv")

# -------------------------------------------------
# Display results
# -------------------------------------------------
st.subheader("Temporal Cluster Behavior Summary")
st.dataframe(summary)

st.subheader("Peak Crime Hours")
st.dataframe(peak_hours)

st.subheader("Peak Crime Months")
st.dataframe(peak_months)

st.subheader("Temporal Crime Profiles")
st.dataframe(profiles)
