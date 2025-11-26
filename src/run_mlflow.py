import mlflow, os
import pandas as pd
from data_loader import load_data, feature_engineer
from models import geographic_features, scale_features, run_kmeans, do_pca, do_umap

def run_experiment(data_path='data/sample_data.csv', n_clusters=6):
    mlflow.set_experiment('patroliq_experiment')
    with mlflow.start_run():
        mlflow.log_param('data_path', data_path)
        mlflow.log_param('n_clusters', n_clusters)
        df = load_data(data_path)
        df = feature_engineer(df)
        X = geographic_features(df)
        Xs, scaler = scale_features(X)
        model, labels = run_kmeans(Xs, k=n_clusters)
        mlflow.log_metric('inertia', float(model.inertia_))
        pca, pc = do_pca(Xs, n_components=2)
        mlflow.log_metric('pca_explained_0', float(pca.explained_variance_ratio_[0]))
        # Save sample outputs
        out = 'outputs/clustered_sample.csv'
        os.makedirs('outputs', exist_ok=True)
        df_out = df.copy().iloc[:len(labels)].reset_index(drop=True)
        df_out['cluster'] = labels
        df_out.to_csv(out, index=False)
        mlflow.log_artifact(out)
        print('Run completed. Outputs written to', out)

if __name__ == '__main__':
    run_experiment()
