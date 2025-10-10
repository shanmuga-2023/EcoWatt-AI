import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "energy_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "demand_model.pkl")
ANOMALY_PATH = os.path.join(BASE_DIR, "anomaly_model.pkl")

def train_models():
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    df['hour'] = df['timestamp'].dt.hour
    df['consumption_lag1'] = df['consumption'].shift(1).fillna(method='bfill')
    df['consumption_lag24'] = df['consumption'].shift(24).fillna(method='bfill')

    features = ['hour', 'temperature', 'solar_energy', 'grid_load', 'consumption_lag1', 'consumption_lag24']
    X = df[features]
    y = df['consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"✅ Demand Model Trained | Test MSE: {mse:.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Saved model: {MODEL_PATH}")

    # Train anomaly model
    anom = IsolationForest(contamination=0.02, random_state=42)
    anom.fit(df[['consumption']])
    joblib.dump(anom, ANOMALY_PATH)
    print(f"💾 Saved anomaly model: {ANOMALY_PATH}")

if __name__ == "__main__":
    train_models()

