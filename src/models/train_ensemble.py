import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Tuple, Dict, Any, List

# ==========================================
# 1. KONFIGURASI FITUR DAN TARGET
# ==========================================
FEATURES = [
    'precip_1h', 'precip_3h', 'precip_6h', 'precip_12h', 'precip_24h',
    'temperature', 'relative_humidity', 'surface_pressure', 
    'pressure_trend_3h', 'wind_speed'
]
TARGET = 'flood_occurrence'

# Ambang batas risiko berdasarkan PRD Section 8
RISK_THRESHOLDS = [
    (20, 'Very Low'),
    (40, 'Low'),
    (60, 'Moderate'),
    (80, 'High'),
    (100, 'Critical')
]

# ==========================================
# 2. MODUL FUNGSI
# ==========================================

def load_and_split_data(filepath: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Memuat data dan melakukan Time-based Split untuk mencegah temporal data leakage.
    Sesuai prinsip di AGENTS.md: dilarang menggunakan random train_test_split.
    """
    df = pd.read_csv(filepath)
    # Pastikan data diurutkan berdasarkan waktu dari yang paling lampau ke terbaru
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    
    # Memisahkan matriks fitur (X) dan target (y)
    X_train = df[FEATURES].iloc[:split_idx]
    y_train = df[TARGET].iloc[:split_idx]
    
    X_test = df[FEATURES].iloc[split_idx:]
    y_test = df[TARGET].iloc[split_idx:]
    
    print(f"Data di-split secara temporal.")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_ensemble_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestClassifier, xgb.XGBClassifier]:
    """
    Melatih Random Forest dan XGBoost (PRD 7.2 Baseline Ensemble).
    """
    # 1. Random Forest: Robust terhadap noise
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        class_weight='balanced', # Penting untuk recall
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # 2. XGBoost: Menangkap hubungan antarvariabel kompleks
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=1.5, # Memberi penalti lebih jika gagal menebak banjir (fokus pada Recall)
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    return rf_model, xgb_model

def evaluate_model(y_true: pd.Series, y_prob: np.ndarray, model_name: str) -> None:
    """Mencetak evaluasi sesuai metrik yang diminta PRD Section 14."""
    y_pred = (y_prob >= 0.5).astype(int)
    
    print(f"\n--- Evaluasi Model: {model_name} ---")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f} (Prioritas Utama MVP!)")
    print(f"F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def calculate_risk_metrics(flood_probability: float) -> Dict[str, Any]:
    """
    Mengonversi probabilitas mentah menjadi output frontend yang dimengerti user.
    """
    risk_score = int(flood_probability * 100)
    
    # Kalibrasi Risk Level
    risk_level = "Very Low"
    for threshold, level in RISK_THRESHOLDS:
        if risk_score <= threshold:
            risk_level = level
            break
            
    # Asumsi Severity untuk MVP berdasarkan probabilitas historis
    # (Bisa dikalibrasi ulang nantinya dengan regresi terpisah ke `water_level_cm`)
    severity = "none"
    if risk_score > 80:
        severity = "severe"
    elif risk_score > 60:
        severity = "moderate"
    elif risk_score > 40:
        severity = "low"
        
    return {
        "flood_probability": round(flood_probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "severity": severity
    }

# ==========================================
# 3. PIPELINE EKSEKUSI
# ==========================================
def main():
    # Gunakan dataset yang telah dibuat sebelumnya
    dataset_path = 'data/ml_flood_training_dataset.csv' # Ubah ke dataset _2.csv jika berada di path tersebut
    
    # 1. Split Data
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_path)
    
    # 2. Train Models
    print("\nMelatih Ensemble Models (RF + XGBoost)...")
    rf_model, xgb_model = train_ensemble_models(X_train, y_train)
    
    # 3. Dapatkan Prediksi Probabilitas
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Soft Voting Ensemble
    ensemble_probs = (rf_probs + xgb_probs) / 2
    
    # 4. Evaluasi Independen & Ensemble
    evaluate_model(y_test, rf_probs, "Random Forest")
    evaluate_model(y_test, xgb_probs, "XGBoost")
    evaluate_model(y_test, ensemble_probs, "Ensemble (RF + XGB)")
    
    # 5. Simulasi Output API (Menyerupai Section 'API Design' di AGENTS.md)
    print("\n--- Simulasi Output Payload API ---")
    sample_index = 5  # Ambil sampel acak dari data test untuk didemonstrasikan
    sample_prob = float(ensemble_probs[sample_index])
    
    result = calculate_risk_metrics(sample_prob)
    print(result)

    # 6. SIMPAN MODEL (Dibutuhkan untuk FastAPI)
    print("\nMenyimpan model untuk tahap produksi...")
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(xgb_model, 'xgb_model.pkl')
    print("Berhasil! File 'rf_model.pkl' dan 'xgb_model.pkl' telah dibuat di direktori saat ini.")

if __name__ == '__main__':
    main()