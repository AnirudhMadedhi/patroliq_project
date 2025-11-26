import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

def geographic_features(df):
    X = df[['Latitude','Longitude']].dropna().values
    return X

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def run_kmeans(X, k=6, random_state=42):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def run_dbscan(X, eps=0.01, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels

def run_agglomerative(X, k=6):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return model, labels

def do_pca(X, n_components=2):
    p = PCA(n_components=n_components, random_state=42)
    comp = p.fit_transform(X)
    return p, comp

def do_umap(X, n_components=2):
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        emb = reducer.fit_transform(X)
        return reducer, emb
    else:
        # fallback to t-SNE
        ts = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=500)
        emb = ts.fit_transform(X)
        return ts, emb
