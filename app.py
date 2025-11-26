import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_data, feature_engineer
from src.models import geographic_features, scale_features, run_kmeans, run_dbscan, run_agglomerative, do_pca, do_umap

st.set_page_config(layout='wide', page_title='PatrolIQ - Demo')

st.title('PatrolIQ - Smart Safety Analytics (Demo)')

st.sidebar.header('Controls')
data_path = st.sidebar.text_input('Data CSV path', value='data/sample_data.csv')
sample_n = st.sidebar.number_input('Use first N rows (0 = all)', min_value=0, value=5000, step=1000)
k = st.sidebar.slider('K (KMeans / Agglomerative)', 2, 12, 6)
eps = st.sidebar.number_input('DBSCAN eps (approx degrees)', min_value=0.001, max_value=1.0, value=0.01, step=0.001)

df = load_data(data_path, n_rows=sample_n if sample_n>0 else None)
df = feature_engineer(df)

st.subheader('Data snapshot')
st.dataframe(df.head(100))

st.subheader('Geographic clusters')

X = geographic_features(df)
Xs, scaler = scale_features(X)

km_model, km_labels = run_kmeans(Xs, k=k)
df['kmeans_cluster'] = km_labels.astype(int)

db_model, db_labels = run_dbscan(Xs, eps=eps, min_samples=10)
df['dbscan_cluster'] = db_labels.astype(int)

ag_model, ag_labels = run_agglomerative(Xs, k=k)
df['agg_cluster'] = ag_labels.astype(int)

st.markdown('**KMeans cluster map (scatter)**')
fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', color='kmeans_cluster', hover_data=['Primary Type','Description'], zoom=9, height=500)
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig, use_container_width=True)

st.markdown('**DBSCAN cluster map (scatter)**')
fig2 = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', color='dbscan_cluster', hover_data=['Primary Type','Description'], zoom=9, height=500)
fig2.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Dimensionality reduction')

st.markdown('PCA projection')
pca, pc = do_pca(Xs, n_components=2)
df['pca0'] = pc[:,0]; df['pca1'] = pc[:,1]
fig3 = px.scatter(df, x='pca0', y='pca1', color='kmeans_cluster', hover_data=['Primary Type','Hour'], height=500)
st.plotly_chart(fig3, use_container_width=True)

st.markdown('UMAP / t-SNE projection (may be slower)')
reducer, emb = do_umap(Xs, n_components=2)
df['emb0'] = emb[:,0]; df['emb1'] = emb[:,1]
fig4 = px.scatter(df, x='emb0', y='emb1', color='kmeans_cluster', hover_data=['Primary Type','Hour'], height=500)
st.plotly_chart(fig4, use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.markdown('Made for demo. Replace data with full Chicago CSV for real analysis.')
