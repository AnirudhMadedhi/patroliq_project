import pandas as pd
import numpy as np

def load_data(path='data/sample_data.csv', sample_frac=None, n_rows=None):
    df = pd.read_csv(path, parse_dates=['Date'])
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    if n_rows is not None:
        df = df.head(n_rows)
    return df

def feature_engineer(df):
    df = df.copy()
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday','Sunday']).astype(int)
    # simple severity mapping
    sev_map = {'HOMICIDE':5,'ASSAULT':4,'ROBBERY':4,'BURGLARY':3,'THEFT':2,'NARCOTICS':1,'BATTERY':3,'CRIMINAL DAMAGE':2,'MOTOR VEHICLE THEFT':3}
    df['Crime_Severity'] = df['Primary Type'].map(sev_map).fillna(1).astype(int)
    return df
