from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
from rl_optimizer import QLearningOptimizer  # ✅ New RL module

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "energy_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "demand_model.pkl")
ANOMALY_PATH = os.path.join(BASE_DIR, "anomaly_model.pkl")

app = FastAPI(title="EcoWatt AI - Adaptive Smart Energy Backend")

# ---------- GLOBAL MODELS ----------
demand_model = None
anom_model = None
optimizer = QLearningOptimizer()  # Adaptive AI Optimizer
optimizer.train(episodes=300)     # Pre-train RL agent for better demo visuals

# ---------- MODEL LOADER ----------
def load_models():
    global demand_model, anom_model
    if demand_model is None and os.path.exists(MODEL_PATH):
        demand_model = joblib.load(MODEL_PATH)
    if anom_model is None and os.path.exists(ANOMALY_PATH):
        anom_model = joblib.load(ANOMALY_PATH)

# ---------- REQUEST BODY ----------
class PredictRequest(BaseModel):
    temperature: float
    solar_energy: float
    grid_load: float
    hour: int = None
    lag1: float = None
    lag24: float = None

# ---------- STARTUP ----------
@app.on_event("startup")
def startup_event():
    load_models()
    print("✅ Models Loaded | Backend Ready with RL Optimizer")

# ---------- ROOT TEST ----------
@app.get("/")
def root():
    return {"message": "EcoWatt AI Backend Active", "optimizer_status": "Running"}

# ---------- DASHBOARD DATA ----------
@app.get("/dashboard_data")
def dashboard_data():
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    last24 = df.tail(24)

    avg_consumption = round(last24['consumption'].mean(), 2)
    avg_solar = round(last24['solar_energy'].mean(), 2)
    renewable_ratio = round((avg_solar / (avg_consumption + 1)) * 100, 2)

    anomalies = []
    if os.path.exists(ANOMALY_PATH):
        anom = joblib.load(ANOMALY_PATH)
        preds = anom.predict(df[['consumption']])
        anom_points = df[preds == -1].tail(10)
        anomalies = anom_points[['timestamp', 'consumption']].to_dict(orient='records')

    return {
        "avg_consumption": avg_consumption,
        "avg_solar": avg_solar,
        "renewable_ratio_percent": renewable_ratio,
        "anomalies": anomalies
    }

# ---------- DEMAND PREDICTION ----------
@app.post("/predict_demand")
def predict_demand(req: PredictRequest):
    load_models()
    if demand_model is None:
        return {"error": "Demand model not trained yet"}

    hour = req.hour if req.hour is not None else datetime.now().hour
    df = pd.read_csv(CSV_PATH).sort_values("timestamp")
    lag1 = req.lag1 or df['consumption'].iloc[-1]
    lag24 = req.lag24 or (df['consumption'].iloc[-24] if len(df) >= 24 else lag1)

    features = [[hour, req.temperature, req.solar_energy, req.grid_load, lag1, lag24]]
    pred = float(demand_model.predict(features)[0])
    return {"predicted_consumption": round(pred, 2)}

# ---------- RL-BASED OPTIMIZATION ----------
@app.post("/optimize")
def optimize_energy(req: PredictRequest):
    pred_resp = predict_demand(req)
    predicted = pred_resp.get("predicted_consumption", 0)
    solar = req.solar_energy

    learned = optimizer.optimize(solar, predicted)  # Q-learning decision
    renewable_used = learned['solar_used']
    grid_used = learned['grid_used']
    renewable_ratio = learned['renewable_ratio_percent']

    # CO₂ savings calculation
    co2_before = predicted * 0.8
    co2_after = grid_used * 0.8
    co2_saved = round(co2_before - co2_after, 2)

    return {
        "predicted_consumption": predicted,
        "renewable_used_kwh": renewable_used,
        "grid_used_kwh": grid_used,
        "renewable_ratio_percent": renewable_ratio,
        "co2_saved_kg": co2_saved
    }

# ---------- RL METRICS (for visualization) ----------
@app.get("/rl_metrics")
def rl_metrics():
    """Return the current Q-table and average reward per state."""
    q_table = optimizer.q_table.tolist()
    avg_rewards = np.mean(optimizer.q_table, axis=1).tolist()
    return {
        "q_table": q_table,
        "avg_rewards": avg_rewards,
        "actions": optimizer.actions,
        "episodes_trained": optimizer.training_episodes
    }
