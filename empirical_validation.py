"""
empirical_validation.py — BLOK 4: Validasi Eksperimental & Koreksi McWhirter
==================================================================
Q1 Thesis: The Empirical Truth — Deconvoluting Real LIBS Spectra

Fungsi:
1. Mengimpor data spektrum LIBS mentah (CSV).
2. Prapemrosesan (koreksi baseline & interpolasi) agar identik dengan resolusi model.
3. Ekstraksi suhu (T_e) dan densitas (n_e) menggunakan SVR Inverter dari Blok 3.
4. Evaluasi Kriteria McWhirter secara matematis untuk menjatuhkan asumsi LTE.
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(_CONFIG_PATH, 'r') as f:
    _CONFIG = yaml.safe_load(f)

from libs_physics import SIMULATION_CONFIG

def load_and_preprocess_experimental_data(csv_path: str, target_wavelengths: np.ndarray) -> np.ndarray:
    print(f"[Prep] Membaca data eksperimen dari: {csv_path}")
    # Gunakan separator cerdas untuk melahap .csv (koma), .asc (spasi/tab), atau .txt (titik koma)
    df = pd.read_csv(csv_path, sep=r'[\s,;]+', engine='python', header=None, skiprows=lambda x: x < 5 if 'asc' in csv_path.lower() else 0)
    
    # Jika skrip salah mendeteksi header karena format ASC, paksakan penamaan
    if not isinstance(df.iloc[0, 0], (int, float, np.float64)): 
        df = df.iloc[1:].reset_index(drop=True)
    
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df.columns = [f"Col_{i}" for i in range(df.shape[1])]
    
    # Kolom 0 otomatis panjang gelombang, Kolom 1 intensitas (standar ASCII Andor)
    I_raw = df.iloc[:, 1].values
    wl_raw = df.iloc[:, 0].values
    
    # 1. Koreksi baseline / latar (pendekatan kuantil terbawah)
    baseline = np.percentile(I_raw, 5) 
    I_corr = I_raw - baseline
    I_corr[I_corr < 0] = 0
    
    # 2. Interpolasi ke grid resolusi model komputasi murni
    print(f"[Prep] Interpolasi spektrum ke resolusi termodinamika ({len(target_wavelengths)} piksel)...")
    I_interp = np.interp(target_wavelengths, wl_raw, I_corr)
    
    # 3. Normalisasi L-Infinity (Maksimum intensitas = 1.0)
    m = I_interp.max()
    if m > 0:
        I_interp /= m
        
    return I_interp

def run_empirical_validation(model_pkl: str, csv_path: str):
    print("=== BLOK 4: Validasi Eksperimental & Koreksi McWhirter ===")
    
    # --- 1. Memuat Model SVR dari BLOK 3 ---
    if not os.path.exists(model_pkl):
        print(f"Error: Model {model_pkl} tidak ditemukan. Harap jalankan Blok 3.")
        return
        
    print(f"[Inversi] Memuat AI pipeline regresi: {model_pkl}")
    pipeline = joblib.load(model_pkl)
    
    pca = pipeline['pca']
    scaler_X = pipeline['scaler_X']
    scaler_y = pipeline['scaler_y']
    model = pipeline['model']
    cols = pipeline['columns']
    
    # Mendefinisikan grid termodinamika internal
    wl_min, wl_max = SIMULATION_CONFIG["wl_range_nm"]
    N_pts = SIMULATION_CONFIG["resolution"]
    target_wl = np.linspace(wl_min, wl_max, N_pts, dtype=np.float64)
    
    # --- 2. Input Spektrum Eksperimen Nyata ---
    if not os.path.exists(csv_path):
        print(f"\nPeringatan: Data spektrum {csv_path} tidak ditemukan!")
        print(">> Menciptakan file MOCK 'data_eksperimen_mock.csv' untuk kerangka demo...")
        
        # Membuat kurva palsu: 2 puncak spektral dominan + derau noise tinggi
        mock_wl = np.linspace(wl_min - 10, wl_max + 10, 3000)
        mock_I = np.exp(-((mock_wl - 250)/0.7)**2) * 5000 + np.exp(-((mock_wl - 400)/1.2)**2) * 3500
        mock_I += np.random.normal(200, 50, len(mock_wl))  # Noise
        
        pd.DataFrame({'Wavelength': mock_wl, 'Intensity': mock_I}).to_csv(csv_path, index=False)
        print(">> MOCK data diciptakan.\n")
        
    I_exp = load_and_preprocess_experimental_data(csv_path, target_wl)
    
    # --- 3. Dekonvolusi Spektrum (Ekstraksi) ---
    print("\n[Inversi] Memproses data piksel masif melalui ruang hiperdimensi SVR...")
    X_features = I_exp.reshape(1, -1)
    
    # Proyeksi PCA & Standarisasi
    X_pca = pca.transform(X_features)
    X_scaled = scaler_X.transform(X_pca)
    
    # Prediksi Inverse
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    
    print("\n>>> HASIL TRANSLASI FISIKA (PARAMETER PLASMA INTI) <<<")
    results = dict(zip(cols, y_pred))
    for k, v in results.items():
        print(f"    {k:<16} = {v:.2e}")
        
    # --- 3.5. Komparasi dengan Ground Truth Konvensional (Saha-Boltzmann) ---
    import re
    sample_id = None
    filename = os.path.basename(csv_path)
    match = re.search(r'S(\d+)', filename, re.IGNORECASE)
    if match:
        sample_id = f"S{match.group(1).upper()}"
        
    legacy_pkl = "data/ground_truth_legacy.pkl"
    if sample_id and os.path.exists(legacy_pkl):
        try:
            ground_truth = joblib.load(legacy_pkl)
            if sample_id in ground_truth:
                print(f"\n[Benchmarking] Ditemukan Ground Truth historis untuk '{sample_id}'!")
                saha_Te = ground_truth[sample_id]['T_e_K']
                saha_ne = ground_truth[sample_id]['n_e_cm3']
                
                svr_Te = results.get('T_e_core_K', 0)
                svr_ne = results.get('n_e_core_cm3', 0)
                
                err_Te = abs(svr_Te - saha_Te) / saha_Te * 100 if pd.notna(saha_Te) and saha_Te > 0 else float('nan')
                err_ne = abs(svr_ne - saha_ne) / saha_ne * 100 if pd.notna(saha_ne) and saha_ne > 0 else float('nan')
                
                print("-" * 75)
                print(f"   === ADU MEKANIK: AI (SVR) vs MANUSIA (Saha-Boltzmann) ===")
                print(f"   {'Parameter':<12} | {'SVR (Blok 3)':<17} | {'Saha (Konvensional)':<20} | {'Error (%)':<10}")
                print(f"   {'Suhu (Te)':<12} | {svr_Te:<17.0f} | {saha_Te:<20.0f} | {err_Te:.2f}%")
                print(f"   {'Densit(ne)':<12} | {svr_ne:<17.2e} | {saha_ne:<20.2e} | {err_ne:.2f}%")
                print("-" * 75)
        except Exception as e:
            print(f"[Warning] Gagal memuat fitur perbandingan: {e}")
            
    # --- 4. Pembuktian Kriteria McWhirter ---
    print("\n[Fisika] Evaluasi Hukum Termodinamika (Kriteria McWhirter)...")
    
    T_core = results.get('T_e_core_K', 12000.0)
    ne_core = results.get('n_e_core_cm3', 1e17)
    
    # Asumsi beda energi transisi optik ditarik secara dinamis dari Pusat YAML
    delta_E_eV = _CONFIG['plasma_target']['mcwhirter_delta_E_eV'] 
    
    # McWhirter: N_e >= 1.6e12 * T_e^{1/2} * (dE)^3
    # Dengan T_e dalam K, dan dE dalam eV
    mcwhirter_limit = 1.6e12 * np.sqrt(T_core) * (delta_E_eV ** 3)
    
    print("-" * 60)
    print(f"Temperatur Elektron (T_e): {T_core:.0f} K")
    print(f"Transisi Optik Terlebar (ΔE): {delta_E_eV} eV")
    print(f"Densitas Plasma Terukur (n_e): {ne_core:.2e} cm^-3")
    print(f"Batas Minimum McWhirter:       {mcwhirter_limit:.2e} cm^-3")
    
    # Keputusan Akhir
    if ne_core >= mcwhirter_limit:
        print("\nKesimpulan: Plasma MEMENUHI syarat Local Thermodynamic Equilibrium (LTE).")
        print("Metodologi Saha/Boltzmann tradisional masih sah digunakan.")
    else:
        print("\nKesimpulan: Plasma GAGAL memenuhi syarat LTE!")
        print(">> [BINTANG THESIS]: Asumsi Keseimbangan Termal pada sampel ini terbukti adalah ILLUSI.")
        print(">> Paradigma Collisional-Radiative (CR) dari Tesis Q1 mutlak diperlukan.")
    print("-" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blok 4: Validasi Termodinamika Spektrum Empiris")
    parser.add_argument('--model', type=str, default='model_inversi_svr.pkl', help='File ML SVR (.pkl)')
    parser.add_argument('--csv', type=str, default='data_eksperimen_mock.csv', help='File data CSV eksperimental')
    args = parser.parse_args()
    
    run_empirical_validation(args.model, args.csv)
