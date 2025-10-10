import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from energy_data import load_data

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "demand_model.pkl")

def train_demand_model():
    """Train RandomForest model for energy demand prediction."""
    df = load_data()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['hour'] = df['timestamp'].dt.hour
    df['consumption_lag1'] = df['consumption'].shift(1).fillna(method='bfill')
    df['consumption_lag24'] = df['consumption'].shift(24).fillna(method='bfill')

    X = df[['hour', 'temperature', 'solar_energy', 'grid_load', 'consumption_lag1', 'consumption_lag24']]
    y = df['consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    print(f"✅ Demand Model trained. MSE: {mse:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    return model

def load_demand_model():
    """Load trained demand model."""
    if not os.path.exists(MODEL_PATH):
        print("⚠ Model not found. Training new one...")
        return train_demand_model()
    model = joblib.load(MODEL_PATH)
    return model
