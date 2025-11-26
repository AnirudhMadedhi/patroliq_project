# PatrolIQ - Smart Safety Analytics (Minimal runnable version)
This is a lightweight, runnable skeleton of the PatrolIQ project you provided.
It contains code to generate a small synthetic dataset, run clustering and
dimensionality reduction, log experiments to MLflow locally, and run a Streamlit
app for exploration.

**What's included**
- `data/sample_data.csv` : small synthetic Chicago-like crime dataset (10k rows)
- `notebooks/analysis.ipynb` : placeholder (not included) - use scripts instead
- `src/data_loader.py` : functions to load and preprocess data
- `src/models.py` : clustering and dimensionality reduction utilities
- `src/run_mlflow.py` : example script to run experiments and log to mlflow
- `app.py` : Streamlit app to explore clusters and DR visuals
- `requirements.txt` : python dependencies
- `generate_sample_data.py` : create the sample dataset quickly
- `Dockerfile` : optional containerization (basic)
- `README.md` : this file

**How to run (basic)**
1. Create a virtualenv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Generate sample data (optional, already included):
   ```bash
   python generate_sample_data.py --n 10000
   ```
3. Run the example MLflow experiment (creates `mlruns/` folder):
   ```bash
   python src/run_mlflow.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

**Notes**
- This repository is a runnable skeleton for demonstration and evaluation purposes.
- Replace the sample data with the real Chicago dataset CSV (download from Chicago Data Portal)
  and update `config` if needed.
- MLflow is configured to use local file storage (`mlruns/`) by default.
