"""
train_inversion_model.py — BLOK 3: Arsitektur Inversi SVR Termodinamika
==================================================================
Q1 Thesis: Machine Learning Inversion Engine for CR-LIBS

Fungsi:
1. Memuat dataset sintetik termodinamika (X_train, y_train) dari HDF5.
2. Membangun Pipeline scikit-learn: Ekstraksi Fitur Fisika (White-Box) -> StandardScaler -> SVR.
3. Melatih MultiOutput SVR secara murni sebagai mesin inversi matematis.
4. Memvalidasi kekuatan prediktif (RMSE, R2 score).
5. Mengekspor (`joblib.dump`) model final ke disk untuk digunakan di eksperimen nyata.
"""

import h5py
import numpy as np
import time
import argparse
import joblib
import yaml
import os

from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(_CONFIG_PATH, 'r') as f:
    _CONFIG = yaml.safe_load(f)

from src.feature_extractor import PhysicsFeatureExtractor

def train_model(dataset_file: str, output_model: str):
    print("=== BLOK 3: Arsitektur Inversi SVR (Pysics Feature Version) ===")
    print(f"Memuat matrix dataset dari: {dataset_file}")
    
    try:
        with h5py.File(dataset_file, 'r') as f:
            X = f['spectra'][:]
            y = f['parameters'][:]
            columns = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f['parameters'].attrs['columns']]
    except FileNotFoundError:
        print(f"Error: Dataset {dataset_file} tidak ditemukan. Jalankan BLOK 2 terlebih dahulu.")
        return
        
    print(f"Dataset sukses dimat. Bentuk X (Spektrum): {X.shape}, y (Parameter): {y.shape}")
    
    n_samples = X.shape[0]
    if n_samples < 5:
        print("Peringatan: Jumlah sampel sangat kecil (kurang dari 5). Evaluasi Train/Test split akan diskip.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    print(f"Skema partisi data: {X_train.shape[0]} LATIH | {X_test.shape[0]} UJI.")
    
    # Rekonstruksi grid spektrum untuk Physical Extractor
    wl_min, wl_max = _CONFIG['instrument']['wl_range_nm']
    resolution = _CONFIG['instrument']['resolution']
    wavelengths = np.linspace(wl_min, wl_max, resolution, dtype=np.float64)
    
    # TAHAP 1: EKSTRAKSI FITUR FISIKA
    print(f"\n[Tahap 1] Dekomposisi Spektrum ke Fitur Termodinamika (White-Box)...")
    physics_extractor = PhysicsFeatureExtractor(wavelengths=wavelengths)
    
    X_train_phys = physics_extractor.fit_transform(X_train)
    X_test_phys = physics_extractor.transform(X_test)
    
    # TAHAP 2: Standarisasi Data
    print("[Tahap 2] Memusatkan & menskalakan data fitur fisik (StandardScaler)...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_phys)
    X_test_scaled = scaler_X.transform(X_test_phys)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # TAHAP 3: Pelatihan Mesin Inverse SVR
    print("[Tahap 3] Melatih SVR Multi-Output menggunakan Fitur Fisika Murni...")
    t0 = time.time()
    
    cfg_svr_k = _CONFIG['machine_learning']['svr_kernel']
    cfg_svr_c = _CONFIG['machine_learning']['svr_C']
    svr = SVR(kernel=cfg_svr_k, C=cfg_svr_c, gamma='scale', epsilon=0.01)
    model = MultiOutputRegressor(svr)
    
    model.fit(X_train_scaled, y_train_scaled)
    print(f"Optimalisasi Hyperplane SVR selesai dalam {time.time()-t0:.2f} detik.")
    
    # TAHAP 4: Validasi & Uji Empiris Internal
    print("\n[Validasi] Mengevaluasi model pada himpunan data uji (Test Set) independen...")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    print("\nMetrik Validasi Inversi Termodinamika (Berbasis Physics Features):")
    print("-" * 50)
    print(f"{'Parameter':<18} | {'R-squared (R²)':<14} | {'RMSE':<12}")
    print("-" * 50)
    for i, col in enumerate(columns):
        if len(y_test) > 1:
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            print(f"{col:<18} | {r2:<14.4f} | {rmse:<12.2e}")
        else:
            diff = abs(y_test[0, i] - y_pred[0, i])
            print(f"{col:<18} | {'N/A (n=1)':<14} | Dev=" + f"{diff:.2e}")
    print("-" * 50)
        
    # TAHAP 5: Persistensi Model
    print(f"\n[Menyimpan] Mengekspor *pipeline* mesin inversi lengkap ke: {output_model} ...")
    pipeline = {
        'physics_extractor': physics_extractor,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'model': model,
        'columns': columns
    }
    joblib.dump(pipeline, output_model)
    print("✅ BLOK 3 Selesai! Arsitektur Deterministik sukses menyingkirkan PCA (Black-Box).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blok 3: Latih Arsitektur ML Inversi Berbasis Logika Fisika.")
    parser.add_argument('--dataset', type=str, default='dataset_synthetic.h5', help='File data masuk')
    parser.add_argument('--out', type=str, default='model_inversi_svr.pkl', help='File model keluar')
    args = parser.parse_args()
    
    train_model(args.dataset, args.out)
