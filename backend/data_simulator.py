import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "energy_data.csv")

def generate_initial(days=14):
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
    print(f"✅ Initial data generated: {CSV_PATH}")

def run_live_append(interval_seconds=5):
    print("🌍 Live simulation running... Press Ctrl+C to stop.")
    while True:
        try:
            now = datetime.now()
            hour = now.hour
            base = 120 + 40 * np.sin((hour / 24) * 2 * np.pi) + np.random.normal(0, 5)
            temp = 20 + np.random.normal(0, 1.5)
            solar = max(0, 100 * np.sin(((hour - 6) / 12) * np.pi) + np.random.normal(0, 8))
            grid_load = base - solar * 0.3 + np.random.normal(0, 3)
            consumption = base + np.random.normal(0, 6)
            row = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": round(temp, 2),
                "solar_energy": round(solar, 2),
                "grid_load": round(grid_load, 2),
                "consumption": round(consumption, 2)
            }
            df_new = pd.DataFrame([row])
            df_new.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
            print(f"[+] Added: {row['timestamp']} | consumption={row['consumption']:.2f} | solar={row['solar_energy']:.2f}")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("Simulation stopped.")
            break

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        generate_initial(days=14)
    run_live_append(interval_seconds=5)

