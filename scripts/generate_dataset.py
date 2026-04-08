"""
generate_dataset.py — BLOK 2: Pabrik Dataset Sintetik
==================================================================
Q1 Thesis: Parallelized Synthetic Dataset Generator for CR-LIBS

Fungsi:
1. Menyapu ruang parameter (T_core, n_e_core, dll) secara stokastik (Random Uniform).
2. Mengeksekusi model fisika ODE `libs_physics` secara paralel memanfaatkan seluruh core CPU.
3. Menambahkan derau (noise) eksperimental Gaussian pada level 1% untuk realistis.
4. Menyimpan output berdimensi masif ke dalam format HDF5 yang efisien.
"""

import os

# =========================================================================
# MAC OS (APPLE SILICON M1) MULTIPROCESSING DEADLOCK FIX:
# Matikan paksa inner-threading C-level dari Numpy/Scipy (Accelerate/OpenBLAS).
# Jika ini tidak diset, numpy Radau solver akan 'hang' berjam-jam ketika 
# diletakkan di dalam modul multiprocessing Pool 'spawn' di macOS.
# Set ini harus *SEBELUM* impor numpy!
# =========================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import h5py
import time
import os
import argparse
import multiprocessing as mp

from src.libs_physics import DataFetcher, PhysicsCalculator, PlasmaZoneParams, TwoZonePlasma

import yaml

# Memuat Konfigurasi Gateway
_BASE_DIR = os.environ.get(
    "LIBS_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")
with open(_CONFIG_PATH, 'r') as f:
    _CONFIG = yaml.safe_load(f)

# Variabel global untuk setiap Process-Worker agar tidak di-pickle ulang dari master
_fetcher = None
_elements = None
_fwhm_nm = _CONFIG['instrument']['fwhm_nm']
_noise_level = _CONFIG['instrument']['noise_level']

def init_worker(elements, fwhm_nm_unused):
    """Inisialisasi memori per-core CPU untuk DataFetcher yang berukuran besar."""
    global _fetcher, _elements
    _fetcher = DataFetcher()  
    _elements = elements

def simulate_single_spectrum(args):
    """Fungsi pekerja: menerima kombinasi parameter, merakit spektrum, mengembalikan array."""
    idx, T_core, T_shell, n_e_core, n_e_shell = args
    
    # Ketebalan geometris (dipertahankan dari YAML GATEWAY)
    d_core_m = _CONFIG['monte_carlo_synthesizer']['core']['thickness_m']
    d_shell_m = _CONFIG['monte_carlo_synthesizer']['shell']['thickness_m']
    
    core = PlasmaZoneParams(T_e_K=T_core, T_i_K=T_core*0.8, n_e_cm3=n_e_core, thickness_m=d_core_m, label='Core')
    shell = PlasmaZoneParams(T_e_K=T_shell, T_i_K=T_shell*0.8, n_e_cm3=n_e_shell, thickness_m=d_shell_m, label='Shell')
    
    try:
        # Eksekusi Blok 1
        model = TwoZonePlasma(core, shell, _elements, _fetcher)
        wl, I_raw, meta = model.run()
        
        # Broadening Instrumen
        if _fwhm_nm > 0.0:
            I_sim = PhysicsCalculator.instrumental_broadening(I_raw, wl, _fwhm_nm)
        else:
            I_sim = I_raw
            
        # Normalisasi Maksimum ke 1.0
        m = I_sim.max()
        if m > 0.0:
            I_sim = I_sim / m
            
        # Injeksi Derau Sintetik: 1% atau sesuai config
        noise = np.random.normal(0, _noise_level, size=I_sim.shape)
        I_sim = np.clip(I_sim + noise, 0.0, 1.0)
        
        # Bungkus hasil (Float32 untuk efisiensi penyimpanan HDF5)
        theta = np.array([T_core, T_shell, n_e_core, n_e_shell], dtype=np.float32)
        return idx, theta, I_sim.astype(np.float32)
        
    except Exception as e:
        # Mencegah worker mati akibat matriks Jacobian yang infinitely stiff
        print(f"[Worker] Singularity dihindari pada matriks (idx={idx}, Tc={T_core:.0f}K, nc={n_e_core:.1e}): {str(e)}")
        return idx, None, None

def generate_dataset(n_samples: int, output_file: str, num_workers: int = None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    print(f"=== BLOK 2: Mesin Sintesis Dataset CR-LIBS ===")
    print(f"Target Sampel      : {n_samples} spektrum sintetik")
    print(f"Unit Pekerja (CPU) : {num_workers} cores aktif")
    print(f"File Output (HDF5) : {output_file}")
    
    # Rentang Domain Parameter Termodinamika dari config.yaml (dipaksa ke float agar aman dari string-parsing PyYAML)
    bounds = {
        'T_core': tuple(map(float, _CONFIG['monte_carlo_synthesizer']['core']['T_range_K'])),
        'T_shell': tuple(map(float, _CONFIG['monte_carlo_synthesizer']['shell']['T_range_K'])),
        'ne_core': tuple(map(float, _CONFIG['monte_carlo_synthesizer']['core']['ne_range_cm3'])),
        'ne_shell': tuple(map(float, _CONFIG['monte_carlo_synthesizer']['shell']['ne_range_cm3']))
    }
    
    # Material komposisi langsung diproyeksikan dari config.yaml
    els = _CONFIG['plasma_target']['elements']
    fracs = _CONFIG['plasma_target']['fractions']
    elements = [(el, 1, frac) for el, frac in zip(els, fracs)]
    
    # Sampling parameter secara seragam
    np.random.seed(42)
    tasks = []
    
    d_core_m = _CONFIG['monte_carlo_synthesizer']['core']['thickness_m']
    d_shell_m = _CONFIG['monte_carlo_synthesizer']['shell']['thickness_m']
    
    for i in range(n_samples):
        Tc = np.random.uniform(*bounds['T_core'])
        Ts = np.random.uniform(*bounds['T_shell'])
        if Ts > Tc: Ts = Tc * 0.8
        
        nc = np.random.uniform(*bounds['ne_core'])
        ns = np.random.uniform(*bounds['ne_shell'])
        
        # Override tasks parameter if needed, but the worker currently uses hardcoded thickness
        # Wait, the worker uses hardcoded thickness in generate_dataset.py. We should pass it via args if we want it dynamic.
        tasks.append((i, Tc, Ts, nc, ns))
    
    # Resolusi bawaan grid wavelength
    resolution = _CONFIG['instrument']['resolution']
    
    # =========================================================================
    # SOLUSI MULTI-CORE (APPLE M1 / LINUX):
    # Proses anak (Workers) JANGAN sampai mewarisi file HDF5 yang sedang terbuka.
    # Maka, Pool diciptakan TERLEBIH DAHULU di RAM, baru file HDF5 dibuka.
    # =========================================================================
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(elements, 0.5)) as pool:
        
        # Eksekusi generator stokastik
        iterator = pool.imap_unordered(simulate_single_spectrum, tasks)
        
        # Menulis ke HDF5 secara akumulatif (Append) oleh MASTER PROCESS SAJA
        with h5py.File(output_file, 'a') as f:
            # Cek apakah dataset lama sudah ada
            if 'parameters' in f and 'spectra' in f:
                dset_theta = f['parameters']
                dset_spectra = f['spectra']
                current_size = dset_theta.shape[0]
                
                # Memperpanjang 
                dset_theta.resize((current_size + n_samples, 4))
                dset_spectra.resize((current_size + n_samples, resolution))
                write_idx = current_size
                print(f"  [Resume] Dataset lama terdeteksi ({current_size} baris). Menambahkan {n_samples} baris baru...")
                
            else:
                dset_theta = f.create_dataset("parameters", shape=(n_samples, 4), maxshape=(None, 4), dtype=np.float32)
                dset_spectra = f.create_dataset("spectra", shape=(n_samples, resolution), maxshape=(None, resolution), dtype=np.float32, 
                                                compression="gzip", compression_opts=4)
                
                dset_theta.attrs['columns'] = ['T_e_core_K', 'T_e_shell_K', 'n_e_core_cm3', 'n_e_shell_cm3']
                dset_spectra.attrs['description'] = 'Normalized synthetic spectra (Instrumental FWHM 0.5nm, 1% Gaussian noise)'
                write_idx = 0
                print(f"  [Baru] Membuat dataset HDF5 kosong untuk {n_samples} baris awal...")
            
            t0 = time.time()
            completed = 0
            completed_success = 0
            
            # Menyerap hasil kerja dari multi-core dan menuliskannya secara Serial
            for idx, theta, I_sim in iterator:
                completed += 1
                
                if theta is not None:
                    dset_theta[write_idx] = theta
                    dset_spectra[write_idx] = I_sim
                    write_idx += 1
                    completed_success += 1
                
                progress_marker = max(1, n_samples // 10)
                if completed % progress_marker == 0 or completed == n_samples:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (n_samples - completed) / rate
                    print(f"  [{completed}/{n_samples}] ... {(completed/n_samples)*100:.1f}% | Laju: {rate:.1f} spec/detik | ETA: {eta:.1f} dtk")
                        
            # Memotong (resize) jika ada sample yang fail
            target_max_size = current_size + completed_success if 'current_size' in locals() else completed_success
            if write_idx < dset_theta.shape[0]:
                dset_theta.resize((write_idx, 4))
                dset_spectra.resize((write_idx, resolution))
                print(f"  [Cleanup] Dataset disusutkan (membuang baris gagal/kosong) menjadi: {write_idx} baris.")
                    
    fs = os.path.getsize(output_file)/1024/1024
    print(f"\n✅ Selesai! Pabrik data ditutup.")
    print(f"Dataset masif berhasil diformat ke: {output_file} ({fs:.1f} MB)")
    print(f"Waktu komputasi bersih: {time.time()-t0:.1f} detik.")


if __name__ == '__main__':
    # Ekstraksi fallback default samplerate dari YAML
    import yaml
    with open(_CONFIG_PATH, 'r') as f:
        _cfg = yaml.safe_load(f)
    default_samples = _cfg['monte_carlo_synthesizer']['generator_samples']
    
    parser = argparse.ArgumentParser(description="Blok 2: Ciptakan Dataset Sintetik LIBS secara paralel")
    parser.add_argument('--samples', type=int, default=default_samples, help='Jumlah sampel')
    parser.add_argument(
        '--out',
        type=str,
        default=os.path.join(_BASE_DIR, 'data', 'dataset_synthetic.h5'),
        help='Nama file output HDF5',
    )
    parser.add_argument('--cores', type=int, default=None, help='Jumlah core CPU (-1 untuk auto)')
    parser.add_argument(
        '--base-dir',
        type=str,
        default=_BASE_DIR,
        help='Base directory project (override LIBS_BASE_DIR)',
    )
    args = parser.parse_args()
    
    if args.base_dir != _BASE_DIR:
        _BASE_DIR = args.base_dir
    generate_dataset(n_samples=args.samples, output_file=args.out, num_workers=args.cores)
