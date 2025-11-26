import pandas as pd, numpy as np, argparse, os
from datetime import datetime, timedelta
rng = np.random.default_rng(42)

def generate(n=10000, out='data/sample_data.csv'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    base_date = datetime(2023,1,1)
    rows = []
    crime_types = ['THEFT','BATTERY','CRIMINAL DAMAGE','NARCOTICS','ASSAULT','BURGLARY','ROBBERY','MOTOR VEHICLE THEFT']
    location_desc = ['STREET','RESIDENCE','APARTMENT','SIDEWALK','PARKING LOT','GAS STATION']
    for i in range(n):
        dt = base_date + timedelta(minutes=int(rng.integers(0,60*24*365)))
        lat = 41.6 + rng.random() * 0.4  # Chicago approx 41.6 - 42.0
        lon = -87.9 + rng.random() * 0.4
        primary = rng.choice(crime_types, p=[0.30,0.2,0.15,0.1,0.08,0.07,0.06,0.04])
        desc = rng.choice(location_desc)
        arrest = bool(rng.random() < 0.07)
        domestic = bool(rng.random() < 0.05)
        beat = int(rng.integers(100, 999))
        district = int(rng.integers(1, 26))
        ward = int(rng.integers(1, 51))
        rows.append({
            'ID': i+1,
            'Date': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'Primary Type': primary,
            'Description': primary + ' - ' + desc,
            'Location Description': desc,
            'Latitude': lat,
            'Longitude': lon,
            'Arrest': arrest,
            'Domestic': domestic,
            'Beat': beat,
            'District': district,
            'Ward': ward
        })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f'Wrote {len(df)} rows to {out}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--out', type=str, default='data/sample_data.csv')
    args = p.parse_args()
    generate(args.n, args.out)
