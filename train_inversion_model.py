"""
train_inversion_model.py — BLOK 3: Arsitektur Inversi SVR
==================================================================
Q1 Thesis: Machine Learning Inversion Engine for CR-LIBS

Fungsi:
1. Memuat dataset sintetik termodinamika (X_train, y_train) dari HDF5.
2. Membangun Pipeline scikit-learn: PCA -> StandardScaler -> SVR.
3. Melatih MultiOutput SVR secara murni sebagai mesin inversi matematis.
4. Memvalidasi kekuatan prediktif (RMSE, R2 score).
5. Mengekspor (`joblib.dump`) model final ke disk untuk digunakan di eksperimen nyata.
"""

import h5py
import numpy as np
import time
import argparse
import joblib

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def train_model(dataset_file: str, output_model: str, n_components: int = 20):
    print("=== BLOK 3: Arsitektur Inversi SVR ===")
    print(f"Memuat matrix dataset dari: {dataset_file}")
    
    try:
        with h5py.File(dataset_file, 'r') as f:
            X = f['spectra'][:]
            y = f['parameters'][:]
            # Dekode nama kolom dari bytes ke string jika perlu
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
        # Split train/test (80% training / 20% testing independen)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    print(f"Skema partisi data: {X_train.shape[0]} LATIH | {X_test.shape[0]} UJI.")
    
    # TAHAP 1: Reduksi Dimensi Spektrum menggunakan PCA
    # Resolusi asli spektrum (24.480 fitur) menyebabkan kompleksitas overfitting memori untuk SVR
    # Oleh karena itu, kompresi informasi linear dari dimensi tinggi mutlak diperlukan.
    print(f"\n[Tahap 1] Dekomposisi PCA (target komponen: {n_components})...")
    
    # Batasi n_components jika jumlah sampel terlalu kecil
    actual_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=actual_components)
    
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    var_exp = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total Varian Informasi Fisika yang dipertahankan: {var_exp:.2f}%")
    
    # TAHAP 2: Standarisasi Data (Wajib untuk performa algoritma RBF kernel SVM)
    print("[Tahap 2] Memusatkan & menskalakan data (StandardScaler)...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_pca)
    X_test_scaled = scaler_X.transform(X_test_pca)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # TAHAP 3: Pelatihan Mesin Inverse SVR
    print("[Tahap 3] Melatih SVR Multi-Output (Kernel=RBF)...")
    t0 = time.time()
    
    # C=100 (penalti keras pada margin), epsilon=0.01 (tube margin ketat)
    svr = SVR(kernel='rbf', C=100.0, gamma='scale', epsilon=0.01)
    model = MultiOutputRegressor(svr)
    
    model.fit(X_train_scaled, y_train_scaled)
    print(f"Optimalisasi Hyperplane SVR selesai dalam {time.time()-t0:.2f} detik.")
    
    # TAHAP 4: Validasi & Uji Empiris Internal
    print("\n[Validasi] Mengevaluasi model pada himpunan data uji (Test Set) independen...")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    print("\nMetrik Validasi Inversi Termodinamika:")
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
        
    # TAHAP 5: Persistensi Model (Mengekspor kecerdasan buatan ke disk)
    print(f"\n[Menyimpan] Mengekspor *pipeline* mesin inversi lengkap ke: {output_model} ...")
    pipeline = {
        'pca': pca,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'model': model,
        'columns': columns
    }
    joblib.dump(pipeline, output_model)
    print("✅ BLOK 3 Selesai! Forward model kini telah dipetakan sukses ke Inverse Engine SVR.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blok 3: Latih Arsitektur ML Inversi dari data HDF5 sintetik.")
    parser.add_argument('--dataset', type=str, default='dataset_synthetic.h5', help='File data masuk')
    parser.add_argument('--out', type=str, default='model_inversi_svr.pkl', help='File model keluar')
    parser.add_argument('--pca', type=int, default=20, help='Jumlah Komponen Principal untuk Spektrum (default: 20)')
    args = parser.parse_args()
    
    train_model(args.dataset, args.out, n_components=args.pca)
