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

import numpy as np
import h5py
import time
import os
import argparse
import multiprocessing as mp

from libs_physics import DataFetcher, PlasmaZoneParams, TwoZonePlasma, instrumental_broadening

# Variabel global untuk setiap Process-Worker agar tidak di-pickle ulang dari master
_fetcher = None
_elements = None
_fwhm_nm = 0.5

def init_worker(elements, fwhm_nm):
    """Inisialisasi memori per-core CPU untuk DataFetcher yang berukuran besar."""
    global _fetcher, _elements, _fwhm_nm
    _fetcher = DataFetcher()  
    _elements = elements
    _fwhm_nm = fwhm_nm

def simulate_single_spectrum(args):
    """Fungsi pekerja: menerima kombinasi parameter, merakit spektrum, mengembalikan array."""
    idx, T_core, T_shell, n_e_core, n_e_shell = args
    
    # Ketebalan geometris (dipertahankan konstan dalam studi ini)
    d_core_m = 1e-3
    d_shell_m = 2e-3
    
    core = PlasmaZoneParams(T_e_K=T_core, T_i_K=T_core*0.8, n_e_cm3=n_e_core, thickness_m=d_core_m, label='Core')
    shell = PlasmaZoneParams(T_e_K=T_shell, T_i_K=T_shell*0.8, n_e_cm3=n_e_shell, thickness_m=d_shell_m, label='Shell')
    
    try:
        # Eksekusi Blok 1
        model = TwoZonePlasma(core, shell, _elements, _fetcher)
        wl, I_raw, meta = model.run()
        
        # Broadening Instrumen
        if _fwhm_nm > 0.0:
            I_sim = instrumental_broadening(I_raw, wl, _fwhm_nm)
        else:
            I_sim = I_raw
            
        # Normalisasi Maksimum ke 1.0
        m = I_sim.max()
        if m > 0.0:
            I_sim = I_sim / m
            
        # Injeksi Derau Sintetik: 1% dari rentang maksimal
        noise = np.random.normal(0, 0.01, size=I_sim.shape)
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
    
    # Rentang Domain Parameter Termodinamika
    bounds = {
        'T_core': (10000.0, 20000.0),
        'T_shell': (5000.0, 10000.0),
        'ne_core': (1e17, 1e18),
        'ne_shell': (1e15, 1e16)
    }
    
    # Kita pertahankan komposisi elemen Si, Al, Fe untuk realisme meteorit/tanah
    elements = [('Si', 1, 0.25), ('Al', 1, 0.25), ('Fe', 1, 0.5)]
    
    # Sampling parameter secara seragam
    np.random.seed(42)
    tasks = []
    for i in range(n_samples):
        Tc = np.random.uniform(*bounds['T_core'])
        Ts = np.random.uniform(*bounds['T_shell'])
        # Hukum Fisika: Shell tidak bolah lebih panas dari Core
        if Ts > Tc: Ts = Tc * 0.8
        
        nc = np.random.uniform(*bounds['ne_core'])
        ns = np.random.uniform(*bounds['ne_shell'])
        
        tasks.append((i, Tc, Ts, nc, ns))
    
    # Resolusi bawaan grid wavelength
    resolution = 24480
    
    # Menulis ke HDF5 secara real-time
    with h5py.File(output_file, 'w') as f:
        # Deklarasi array di disk (chunked array agar bisa kompresi GZIP tanpa memenuhi RAM)
        dset_theta = f.create_dataset("parameters", shape=(n_samples, 4), maxshape=(None, 4), dtype=np.float32)
        dset_spectra = f.create_dataset("spectra", shape=(n_samples, resolution), maxshape=(None, resolution), dtype=np.float32, 
                                        compression="gzip", compression_opts=4)
        
        dset_theta.attrs['columns'] = ['T_e_core_K', 'T_e_shell_K', 'n_e_core_cm3', 'n_e_shell_cm3']
        dset_spectra.attrs['description'] = 'Normalized synthetic spectra (Instrumental FWHM 0.5nm, 1% Gaussian noise)'
        
        t0 = time.time()
        completed = 0
        write_idx = 0
        
        # Eksekusi Paralel Pool
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(elements, 0.5)) as pool:
            for idx, theta, I_sim in pool.imap_unordered(simulate_single_spectrum, tasks):
                completed += 1
                
                if theta is not None:
                    # Tulis berurutan ke sub-blok HDF5 membuang celah nol
                    dset_theta[write_idx] = theta
                    dset_spectra[write_idx] = I_sim
                    write_idx += 1
                
                progress_marker = max(1, n_samples // 10)
                if completed % progress_marker == 0 or completed == n_samples:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (n_samples - completed) / rate
                    print(f"  [{completed}/{n_samples}] ... {(completed/n_samples)*100:.1f}% | Laju: {rate:.1f} spec/detik | ETA: {eta:.1f} dtk")
                    
        # Memotong (resize) array riil dengan data sukses agar terbebas dari artefak baris-nol
        if write_idx < n_samples:
            dset_theta.resize((write_idx, 4))
            dset_spectra.resize((write_idx, resolution))
            print(f"  [Cleanup] Memotong array ke dataset berukuran sukses: {write_idx} dari target awal {n_samples}")
                    
    fs = os.path.getsize(output_file)/1024/1024
    print(f"\n✅ Selesai! Pabrik data ditutup.")
    print(f"Dataset masif berhasil diformat ke: {output_file} ({fs:.1f} MB)")
    print(f"Waktu komputasi bersih: {time.time()-t0:.1f} detik.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blok 2: Ciptakan Dataset Sintetik LIBS secara paralel")
    parser.add_argument('--samples', type=int, default=20, help='Jumlah sampel (def: 20 uji coba)')
    parser.add_argument('--out', type=str, default='dataset_synthetic.h5', help='Nama file output HDF5')
    parser.add_argument('--cores', type=int, default=None, help='Jumlah core CPU (-1 untuk auto)')
    args = parser.parse_args()
    
    generate_dataset(n_samples=args.samples, output_file=args.out, num_workers=args.cores)
