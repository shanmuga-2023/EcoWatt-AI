import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from chatbot_rules import eco_reply   # ✅ NEW: Local EcoBot module

API_URL = "http://127.0.0.1:8000"

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(layout="wide", page_title="EcoWatt AI – Sustainable City Dashboard")

# ---------- AUTO REFRESH ----------
st_autorefresh(interval=30*1000, key="auto_refresh")

# ---------- LOAD CUSTOM CSS & LOGO ----------
logo_path = os.path.join("assets", "logo.png")
css_path = os.path.join("assets", "style.css")

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 4])
if os.path.exists(logo_path):
    col_logo.image(logo_path, width=100)
col_title.markdown("""
    <h1 style='color:#2E7D32;'>🌿 EcoWatt AI</h1>
    <h4 style='color:#388E3C;'>AI-Powered Energy Management for Sustainable Cities</h4>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controls")
selected_view = st.sidebar.radio("Choose View", ["Overview", "Predictions", "City Heatmap", "EcoBot Chat"])
if st.sidebar.button("🔄 Refresh"):
    st.experimental_rerun()

# ---------- FETCH DATA SAFELY ----------
try:
    resp = requests.get(f"{API_URL}/dashboard_data").json()
except:
    st.error("⚠️ Cannot connect to backend API. Please start the FastAPI server first.")
    st.stop()

# ---------- KPI METRICS ----------
col1, col2, col3 = st.columns(3)
col1.metric("Avg Consumption (last 24h)", f"{resp.get('avg_consumption',0)} kWh")
col2.metric("Avg Solar (last 24h)", f"{resp.get('avg_solar',0)} kWh")
col3.metric("Renewable Ratio (%)", f"{resp.get('renewable_ratio_percent',0)}%")

# ---------- RENEWABLE FORECAST ----------
st.markdown("### 🌞 Renewable Energy Forecast (Next 24 Hours)")
hours = pd.date_range(datetime.now(), periods=24, freq='H')
solar_forecast = [max(0, 80 * np.sin(((h.hour - 6) / 12) * np.pi) + np.random.normal(0, 5)) for h in hours]
temp_forecast = [25 + 8 * np.sin((h.hour / 24) * 2 * np.pi) + np.random.normal(0, 1.5) for h in hours]

forecast_df = pd.DataFrame({
    "Hour": [h.strftime("%H:%M") for h in hours],
    "Predicted Solar (kWh)": np.round(solar_forecast, 2),
    "Temperature (°C)": np.round(temp_forecast, 2)
})
fig_forecast = px.line(forecast_df, x="Hour", y="Predicted Solar (kWh)",
                       title="Predicted Solar Generation (Next 24 Hours)",
                       markers=True, color_discrete_sequence=["green"])
st.plotly_chart(fig_forecast, use_container_width=True)

# ---------- CARBON CREDIT CALCULATOR ----------
st.markdown("### 💰 Carbon Credit Savings Estimator")
avg_solar = resp.get("avg_solar", 0)
avg_cons = resp.get("avg_consumption", 1)
renewable_ratio = resp.get("renewable_ratio_percent", 0)
co2_saved = (avg_solar * 0.8)
credits_earned = co2_saved * 15

col1, col2 = st.columns(2)
col1.metric("CO₂ Saved (kg)", f"{round(co2_saved,2)}")
col2.metric("Carbon Credit Value (₹)", f"{round(credits_earned,2)}")
st.progress(min(1.0, renewable_ratio / 100.0))
st.caption(f"🌿 Renewable contribution: {renewable_ratio}% | Each carbon credit ≈ ₹15 saved")

# ---------- GAMIFIED SUSTAINABILITY SCORE ----------
st.markdown("### 🏅 City Sustainability Score")
score = (
    (renewable_ratio * 0.5) +
    (min(avg_solar / avg_cons, 1.0) * 100 * 0.3) +
    (min(co2_saved / 100, 1.0) * 100 * 0.2)
)
score = round(min(score, 100), 2)
if score >= 80:
    badge = "🥇 Excellent Sustainability"
    color = "green"
elif score >= 60:
    badge = "🥈 Good Progress"
    color = "orange"
else:
    badge = "🥉 Needs Improvement"
    color = "red"

st.markdown(f"<h2 style='color:{color}'>EcoWatt Score: {score}/100 – {badge}</h2>", unsafe_allow_html=True)
st.progress(score / 100)
st.caption("Calculated using renewable usage, CO₂ savings, and efficiency ratios.")

# ---------- CITY HEATMAP ----------
if selected_view == "City Heatmap":
    st.markdown("### 🗺 Smart City Energy Efficiency Map (Simulated)")
    np.random.seed(42)
    city_zones = [f"Zone-{i}" for i in range(1, 10)]
    consumption = np.random.uniform(80, 200, len(city_zones))
    renewable_share = np.random.uniform(20, 90, len(city_zones))
    efficiency = np.round((renewable_share / (consumption / 100)), 2)
    df_map = pd.DataFrame({
        "City Zone": city_zones,
        "Efficiency Index": efficiency,
        "Renewable %": renewable_share,
        "Consumption (kWh)": consumption
    })
    fig_map = px.density_heatmap(df_map, x="City Zone", y="Renewable %",
                                 z="Efficiency Index", color_continuous_scale="Greens",
                                 title="City-Wide Energy Efficiency Heatmap")
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Darker green = more efficient, renewable-heavy zones.")

# ---------- ANOMALIES ----------
if selected_view == "Overview":
    st.markdown("---")
    st.subheader("⚠ Detected Anomalies (Energy Spikes)")
    if resp.get("anomalies"):
        anom_df = pd.DataFrame(resp['anomalies'])
        st.dataframe(anom_df)
    else:
        st.success("No major anomalies detected in recent data.")

# ---------- PREDICTION / OPTIMIZATION ----------
if selected_view == "Predictions":
    st.markdown("---")
    st.subheader("🤖 Predict Next Hour & Optimize Energy Mix (Adaptive RL)")
    with st.form("predict_form"):
        temp = st.number_input("Temperature (°C)", value=30.0)
        solar = st.number_input("Available Solar Energy (kWh)", value=50.0)
        grid_load = st.number_input("Current Grid Load (kWh)", value=120.0)
        hour = st.slider("Hour of day", 0, 23, datetime.now().hour)
        submitted = st.form_submit_button("Predict & Optimize (AI)")
        if submitted:
            payload = {
                "temperature": float(temp),
                "solar_energy": float(solar),
                "grid_load": float(grid_load),
                "hour": int(hour)
            }
            pred = requests.post(f"{API_URL}/predict_demand", json=payload).json()
            opt = requests.post(f"{API_URL}/optimize", json=payload).json()
            st.success(f"Predicted Consumption: {pred.get('predicted_consumption','—')} kWh")
            st.metric("Renewable used (kWh)", opt.get("renewable_used_kwh","—"))
            st.metric("Grid used (kWh)", opt.get("grid_used_kwh","—"))
            st.metric("CO₂ saved (kg)", opt.get("co2_saved_kg","—"))
            mix_df = pd.DataFrame({
                "source": ["Renewable", "Grid"],
                "kwh": [opt.get("renewable_used_kwh", 0), opt.get("grid_used_kwh", 0)]
            })
            fig = px.pie(mix_df, names='source', values='kwh',
                         title="AI-Optimized Energy Mix (Reinforcement Learning)")
            st.plotly_chart(fig, use_container_width=True)

# ---------- ECOBOT CHAT ASSISTANT ----------
if selected_view == "EcoBot Chat":
    st.markdown("---")
    st.subheader("🤖 EcoBot – Your AI Sustainability Assistant")
    st.caption("Ask questions like *'How can I save more energy?'* or *'What is my carbon footprint?'*")

    user_q = st.text_input("Ask EcoBot:")
    if user_q:
        context = {
            "co2_saved": co2_saved,
            "renewable_ratio": renewable_ratio
        }
        answer = eco_reply(user_q, context)
        st.success(answer)

# ---------- FOOTER ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<center>
<p style='color:#2E7D32;font-size:14px;'>
© 2025 <b>EcoWatt AI</b> | AI for Sustainable Living 🌱 | Adaptive Optimizer + EcoBot
</p>
</center>
""", unsafe_allow_html=True)
