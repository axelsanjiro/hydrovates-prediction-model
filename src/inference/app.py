import os
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. INISIALISASI & LOAD MODEL
# ==========================================
app = FastAPI(title="Flood Risk Prediction API", version="1.0.0")

# Mengizinkan frontend (React/Vite) mengakses API ini (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Untuk production, ganti dengan domain frontend Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dari folder tempat script dijalankan
try:
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
except Exception as e:
    print(f"Warning: Model gagal dimuat. Pastikan rf_model.pkl dan xgb_model.pkl tersedia. Error: {e}")

# ==========================================
# 2. PYDANTIC SCHEMAS (API CONTRACT)
# ==========================================
class Location(BaseModel):
    latitude: float
    longitude: float

class PredictionMetrics(BaseModel):
    probability: float
    risk_score: int
    risk_level: str
    severity: str

class PredictRequest(BaseModel):
    latitude: float = Field(..., description="Latitude lokasi pengguna")
    longitude: float = Field(..., description="Longitude lokasi pengguna")
    forecast_hours: int = Field(24, description="Jam perkiraan ke depan")

class PredictResponse(BaseModel):
    location: Location
    prediction: PredictionMetrics
    factors: List[str]
    generated_at: str

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def fetch_current_weather(lat: float, lon: float) -> pd.DataFrame:
    """Mengambil data cuaca real-time & prakiraan Open-Meteo (sesuai kebutuhan PRD)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "precipitation,temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "past_days": 1, # Ambil data 24 jam ke belakang untuk menghitung akumulasi
        "forecast_days": 1,
        "timezone": "Asia/Jakarta"
    }
    
    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        # Sesuai AGENTS.md: Penanganan kegagalan API cuaca
        raise HTTPException(status_code=503, detail="Weather data is temporarily unavailable.")
        
    data = response.json()
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    return df

def generate_factors(features: dict) -> List[str]:
    """Menghasilkan explainability/alasan tingkat risiko sesuai AGENTS.md."""
    factors = []
    if features['precip_24h'] > 20:
        factors.append("High 24-hour rainfall accumulation")
    if features['precip_3h'] > 10:
        factors.append("Heavy short-term rainfall detected")
    if features['pressure_trend_3h'] < -1.5:
        factors.append("Significant drop in atmospheric pressure (potential storm)")
    if not factors:
        factors.append("Normal weather conditions")
    return factors

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.get("/health")
def health_check():
    """Endpoint untuk memastikan API berjalan."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/api/predict", response_model=PredictResponse)
def predict_flood_risk(request: PredictRequest):
    """
    Endpoint utama prediksi risiko banjir.
    """
    # 1. Fetch Data Cuaca
    df_weather = fetch_current_weather(request.latitude, request.longitude)
    
    # 2. Feature Engineering (Titik waktu saat ini)
    # Anggap waktu sekarang adalah baris terakhir dari data observasi "past_days"
    now_idx = 23  # Indeks jam ke-24 dalam rentang data
    
    df_weather['precip_1h'] = df_weather['precipitation']
    df_weather['precip_3h'] = df_weather['precipitation'].rolling(3, min_periods=1).sum()
    df_weather['precip_6h'] = df_weather['precipitation'].rolling(6, min_periods=1).sum()
    df_weather['precip_12h'] = df_weather['precipitation'].rolling(12, min_periods=1).sum()
    df_weather['precip_24h'] = df_weather['precipitation'].rolling(24, min_periods=1).sum()
    df_weather['pressure_trend_3h'] = df_weather['surface_pressure'] - df_weather['surface_pressure'].shift(3)
    
    current_features = df_weather.iloc[now_idx]
    
    # 3. Susun Vektor Input Sesuai Urutan Training
    feature_dict = {
        'precip_1h': current_features['precip_1h'],
        'precip_3h': current_features['precip_3h'],
        'precip_6h': current_features['precip_6h'],
        'precip_12h': current_features['precip_12h'],
        'precip_24h': current_features['precip_24h'],
        'temperature': current_features['temperature_2m'],
        'relative_humidity': current_features['relative_humidity_2m'],
        'surface_pressure': current_features['surface_pressure'],
        'pressure_trend_3h': current_features['pressure_trend_3h'] if pd.notna(current_features['pressure_trend_3h']) else 0.0,
        'wind_speed': current_features['wind_speed_10m']
    }
    
    X_input = pd.DataFrame([feature_dict])
    
    # 4. Prediksi Model
    try:
        rf_prob = rf_model.predict_proba(X_input)[0, 1]
        xgb_prob = xgb_model.predict_proba(X_input)[0, 1]
        ensemble_prob = float((rf_prob + xgb_prob) / 2)
    except Exception:
        # Sesuai AGENTS.md: Penanganan kegagalan sistem prediksi
        raise HTTPException(status_code=503, detail="Flood risk prediction is temporarily unavailable.")
    
    # 5. Klasifikasi Risiko (Mengikuti PRD Section 8)
    risk_score = int(ensemble_prob * 100)
    if risk_score <= 20:
        risk_level, severity = "very low", "none"
    elif risk_score <= 40:
        risk_level, severity = "low", "low"
    elif risk_score <= 60:
        risk_level, severity = "moderate", "moderate"
    elif risk_score <= 80:
        risk_level, severity = "high", "high"
    else:
        risk_level, severity = "critical", "severe"

    # 6. Susun Respons JSON Sesuai Kontrak AGENTS.md
    return PredictResponse(
        location=Location(latitude=request.latitude, longitude=request.longitude),
        prediction=PredictionMetrics(
            probability=round(ensemble_prob, 4),
            risk_score=risk_score,
            risk_level=risk_level,
            severity=severity
        ),
        factors=generate_factors(feature_dict),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )