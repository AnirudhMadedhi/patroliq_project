import mlflow
import pandas as pd
import numpy as np

def load_latest_artifact(experiment_name, run_name, artifact_path):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        exp.experiment_id,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )

    run = runs[0]
    local_path = client.download_artifacts(run.info.run_id, artifact_path)

    if artifact_path.endswith(".csv"):
        return pd.read_csv(local_path)
    elif artifact_path.endswith(".npy"):
        return np.load(local_path)
    else:
        return local_path
