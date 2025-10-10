import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
import os
from energy_data import load_data

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "anomaly_model.pkl")

def train_anomaly_model():
    """Train IsolationForest to detect abnormal energy spikes."""
    df = load_data()
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(df[['consumption']])

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Anomaly model trained and saved to {MODEL_PATH}")
    return model

def load_anomaly_model():
    """Load anomaly detection model."""
    if not os.path.exists(MODEL_PATH):
        print("⚠ Anomaly model not found. Training new one...")
        return train_anomaly_model()
    model = joblib.load(MODEL_PATH)
    return model

def detect_anomalies(df):
    """Detect abnormal energy consumption values."""
    model = load_anomaly_model()
    preds = model.predict(df[['consumption']])
    df['is_anomaly'] = preds
    anomalies = df[df['is_anomaly'] == -1]
    print(f"⚠ Detected {len(anomalies)} anomalies.")
    return anomalies
