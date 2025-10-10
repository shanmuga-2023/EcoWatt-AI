import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "energy_data.csv")

def generate_data(days=14):
    """Generate simulated hourly energy data."""
    rows = []
    now = datetime.now()
    start = now - timedelta(days=days)
    ts = start

    while ts <= now:
        hour = ts.hour
        base = 120 + 40 * np.sin((hour / 24) * 2 * np.pi) + np.random.normal(0, 5)
        temp = 20 + 10 * np.sin((ts.timetuple().tm_yday / 365.0) * 2 * np.pi) + np.random.normal(0, 2)
        solar = max(0, 100 * np.sin(((hour - 6) / 12) * np.pi) + np.random.normal(0, 8))
        grid_load = base - solar * 0.3 + np.random.normal(0, 3)
        consumption = base + np.random.normal(0, 6)
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": round(temp, 2),
            "solar_energy": round(solar, 2),
            "grid_load": round(grid_load, 2),
            "consumption": round(consumption, 2)
        })
        ts += timedelta(hours=1)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Data generated at: {CSV_PATH}")
    return df

def load_data():
    """Load energy dataset or create it if missing."""
    if not os.path.exists(CSV_PATH):
        print("⚠ energy_data.csv not found, generating new data...")
        return generate_data(days=14)
    df = pd.read_csv(CSV_PATH)
    return df
