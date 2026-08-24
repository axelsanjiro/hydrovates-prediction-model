import os
import re
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI DAN PATH
# ==========================================
RAWAN_FILE = 'data/Filedata Data Titik Rawan BencanaBanjir.csv'
KEJADIAN_FILE = 'data/Filedata Data Kejadian Bencana Banjir.csv'
OUTPUT_FILE = 'data/ml_flood_training_dataset.csv'

# Fallback koordinat pusat wilayah jika kelurahan tidak ada di katalog titik rawan
WILAYAH_COORDS = {
    'jakarta pusat': (-6.1805, 106.8284),
    'jakarta utara': (-6.1384, 106.8640),
    'jakarta barat': (-6.1674, 106.7637),
    'jakarta selatan': (-6.2615, 106.8106),
    'jakarta timur': (-6.2250, 106.9004),
    'kepulauan seribu': (-5.6122, 106.5614)
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def parse_water_level(val: str) -> float:
    """Mengonversi rentang ketinggian air (misal: '30 s.d 80 cm') menjadi nilai rata-rata kontinu (cm)."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).lower().replace('cm', '').strip()
    nums = [float(n) for n in re.findall(r'\d+', val_str)]
    if not nums:
        return 0.0
    return float(np.mean(nums[:2])) if len(nums) >= 2 else nums[0]

def categorize_severity(water_level_cm: float) -> str:
    """Mengklasifikasikan severity sesuai standar PRD."""
    if water_level_cm <= 20:
        return 'low'
    elif water_level_cm <= 50:
        return 'moderate'
    elif water_level_cm <= 100:
        return 'high'
    else:
        return 'critical'

def fetch_open_meteo_hourly(lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Mengambil data cuaca historis per jam dari Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,rain,temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Jakarta"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            df_hourly = pd.DataFrame(data['hourly'])
            df_hourly['time'] = pd.to_datetime(df_hourly['time'])
            return df_hourly
        else:
            print(f"API Error ({response.status_code}) for ({lat}, {lon})")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# ==========================================
# 3. PIPELINE UTAMA
# ==========================================
def main():
    print("Memuat dataset mentah...")
    df_rawan = pd.read_csv(RAWAN_FILE)
    df_kejadian = pd.read_csv(KEJADIAN_FILE)

    # 1. Normalisasi Titik Koordinat dari file titik rawan (skala 10^7)
    df_rawan['latitude_deg'] = df_rawan['latitude'] / 1e6
    df_rawan['longitude_deg'] = df_rawan['longitude'] / 1e6
    df_rawan['kelurahan_clean'] = df_rawan['kelurahan'].str.lower().str.strip()
    
    # Hitung rata-rata koordinat per kelurahan
    kelurahan_coords = df_rawan.groupby('kelurahan_clean')[['latitude_deg', 'longitude_deg']].mean().to_dict('index')

    # 2. Parsing Data Kejadian
    df_kejadian['kelurahan_clean'] = df_kejadian['kelurahan'].str.lower().str.strip()
    df_kejadian['wilayah_clean'] = df_kejadian['wilayah'].str.lower().str.strip()
    df_kejadian['water_level_cm'] = df_kejadian['jumlah_rata_rata_ketinggian_air'].apply(parse_water_level)
    df_kejadian['severity'] = df_kejadian['water_level_cm'].apply(categorize_severity)

    # Petakan koordinat ke tabel kejadian
    coords_list = []
    for _, row in df_kejadian.iterrows():
        kel = row['kelurahan_clean']
        wil = row['wilayah_clean']
        if kel in kelurahan_coords:
            coords_list.append((kelurahan_coords[kel]['latitude_deg'], kelurahan_coords[kel]['longitude_deg']))
        else:
            # Fallback ke pusat kota wilayah
            matched_wil = next((k for k in WILAYAH_COORDS if k in wil), 'jakarta pusat')
            coords_list.append(WILAYAH_COORDS[matched_wil])
            
    df_kejadian['latitude'] = [c[0] for c in coords_list]
    df_kejadian['longitude'] = [c[1] for c in coords_list]

    # Asumsikan periode 2024 untuk penarikan cuaca historis
    TARGET_YEAR = 2024
    
    training_rows = []
    unique_locations = df_kejadian.drop_duplicates(subset=['kelurahan_clean', 'bulan'])
    print(f"Mengekstraksi fitur cuaca dari Open-Meteo untuk {len(unique_locations)} kombinasi lokasi & bulan...")

    # Cache penarikan API per koordinat & bulan agar tidak redundant
    weather_cache = {}

    for idx, row in unique_locations.iterrows():
        month = int(row['bulan'])
        lat, lon = row['latitude'], row['longitude']
        kel = row['kelurahan_clean']
        water_level = row['water_level_cm']
        severity = row['severity']
        
        # Tentukan rentang tanggal bulan tersebut
        start_date = f"{TARGET_YEAR}-{month:02d}-01"
        end_day = 28 if month == 2 else (30 if month in [4, 6, 9, 11] else 31)
        end_date = f"{TARGET_YEAR}-{month:02d}-{end_day:02d}"

        cache_key = (round(lat, 3), round(lon, 3), month)
        if cache_key not in weather_cache:
            df_weather = fetch_open_meteo_hourly(lat, lon, start_date, end_date)
            weather_cache[cache_key] = df_weather
            time.sleep(0.2) # Menghormati rate limit Open-Meteo
        else:
            df_weather = weather_cache[cache_key]

        if df_weather is None or df_weather.empty:
            continue

        # ==========================================
        # FEATURE ENGINEERING (Rolling Accumulations)
        # ==========================================
        df_weather['precip_1h'] = df_weather['precipitation']
        df_weather['precip_3h'] = df_weather['precipitation'].rolling(3, min_periods=1).sum()
        df_weather['precip_6h'] = df_weather['precipitation'].rolling(6, min_periods=1).sum()
        df_weather['precip_12h'] = df_weather['precipitation'].rolling(12, min_periods=1).sum()
        df_weather['precip_24h'] = df_weather['precipitation'].rolling(24, min_periods=1).sum()
        df_weather['pressure_trend_3h'] = df_weather['surface_pressure'] - df_weather['surface_pressure'].shift(3)

        # 1. Sampel Positif (y = 1): Ambil jam dengan akumulasi curah hujan tertinggi di bulan tersebut
        peak_idx = df_weather['precip_24h'].idxmax()
        peak_row = df_weather.loc[peak_idx]

        training_rows.append({
            'timestamp': peak_row['time'],
            'kelurahan': kel,
            'latitude': lat,
            'longitude': lon,
            'precip_1h': peak_row['precip_1h'],
            'precip_3h': peak_row['precip_3h'],
            'precip_6h': peak_row['precip_6h'],
            'precip_12h': peak_row['precip_12h'],
            'precip_24h': peak_row['precip_24h'],
            'temperature': peak_row['temperature_2m'],
            'relative_humidity': peak_row['relative_humidity_2m'],
            'surface_pressure': peak_row['surface_pressure'],
            'pressure_trend_3h': peak_row['pressure_trend_3h'] if pd.notna(peak_row['pressure_trend_3h']) else 0.0,
            'wind_speed': peak_row['wind_speed_10m'],
            'flood_occurrence': 1,  # Target Klasifikasi
            'water_level_cm': water_level,  # Target Regresi / Severity
            'severity_level': severity
        })

        # 2. Sampel Negatif (y = 0): Ambil jam dengan curah hujan rendah / normal di bulan yang sama
        low_rain_df = df_weather[df_weather['precip_24h'] < 5.0]
        if not low_rain_df.empty:
            neg_row = low_rain_df.sample(1, random_state=42).iloc[0]
            training_rows.append({
                'timestamp': neg_row['time'],
                'kelurahan': kel,
                'latitude': lat,
                'longitude': lon,
                'precip_1h': neg_row['precip_1h'],
                'precip_3h': neg_row['precip_3h'],
                'precip_6h': neg_row['precip_6h'],
                'precip_12h': neg_row['precip_12h'],
                'precip_24h': neg_row['precip_24h'],
                'temperature': neg_row['temperature_2m'],
                'relative_humidity': neg_row['relative_humidity_2m'],
                'surface_pressure': neg_row['surface_pressure'],
                'pressure_trend_3h': neg_row['pressure_trend_3h'] if pd.notna(neg_row['pressure_trend_3h']) else 0.0,
                'wind_speed': neg_row['wind_speed_10m'],
                'flood_occurrence': 0,
                'water_level_cm': 0.0,
                'severity_level': 'none'
            })

    df_final = pd.DataFrame(training_rows)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSelesai! Dataset berhasil disimpan ke '{OUTPUT_FILE}' dengan total {len(df_final)} baris.")

if __name__ == '__main__':
    main()