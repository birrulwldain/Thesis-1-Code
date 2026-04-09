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
from dataclasses import dataclass

import src.libs_physics as libs_physics
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
_fwhm_nm = _CONFIG['instrument']['fwhm_nm']
_noise_level = _CONFIG['instrument']['noise_level']

SPECTRAL_TIER_CONFIGS = {
    "L": {"resolution": 4000, "description": "Low pragmatic resolution"},
    "M": {"resolution": 8000, "description": "Medium pragmatic resolution"},
    "H": {"resolution": 16000, "description": "High pragmatic resolution"},
}

@dataclass
class DatasetGroupSpec:
    name: str
    description: str
    element_bounds_pct: list[tuple[str, tuple[float, float]]]


def init_worker(resolution: int):
    """Inisialisasi memori per-core CPU untuk DataFetcher yang berukuran besar."""
    global _fetcher
    libs_physics.SIMULATION_CONFIG["resolution"] = int(resolution)
    _fetcher = DataFetcher()  


def _load_group_spec(group_name: str) -> DatasetGroupSpec:
    profile = _CONFIG["synthetic_dataset_profiles"]["volcanic_soil"]["groups"][group_name]
    element_bounds = [
        (elem, tuple(map(float, bounds)))
        for elem, bounds in profile["elements"].items()
    ]
    return DatasetGroupSpec(
        name=group_name,
        description=str(profile.get("description", "")),
        element_bounds_pct=element_bounds,
    )


def _sample_bounded_simplex_pct(
    element_bounds_pct: list[tuple[str, tuple[float, float]]],
    rng: np.random.Generator,
) -> np.ndarray:
    lowers = np.asarray([bounds[0] for _, bounds in element_bounds_pct], dtype=np.float64) / 100.0
    uppers = np.asarray([bounds[1] for _, bounds in element_bounds_pct], dtype=np.float64) / 100.0
    if lowers.sum() > 1.0 + 1e-9 or uppers.sum() < 1.0 - 1e-9:
        raise ValueError("Komposisi bounds tidak feasible untuk simplex 100%.")

    order = rng.permutation(len(element_bounds_pct))
    x = np.zeros(len(element_bounds_pct), dtype=np.float64)
    remaining = 1.0
    remaining_lower = lowers.sum()
    remaining_upper = uppers.sum()

    for pos, idx in enumerate(order[:-1]):
        remaining_lower -= lowers[idx]
        remaining_upper -= uppers[idx]
        lo = max(lowers[idx], remaining - remaining_upper)
        hi = min(uppers[idx], remaining - remaining_lower)
        if hi < lo:
            raise RuntimeError("Gagal sampling komposisi bounded simplex.")
        value = rng.uniform(lo, hi)
        x[idx] = value
        remaining -= value

    last_idx = int(order[-1])
    x[last_idx] = remaining
    if x[last_idx] < lowers[last_idx] - 1e-9 or x[last_idx] > uppers[last_idx] + 1e-9:
        raise RuntimeError("Komposisi terakhir keluar dari bounds.")
    x = np.clip(x, lowers, uppers)
    x = x / x.sum()
    return (x * 100.0).astype(np.float32)

def simulate_single_spectrum(args):
    """Fungsi pekerja: menerima kombinasi parameter, merakit spektrum, mengembalikan array."""
    idx, T_core, T_shell, n_e_core, n_e_shell, elements, theta = args
    
    # Ketebalan geometris (dipertahankan dari YAML GATEWAY)
    d_core_m = _CONFIG['monte_carlo_synthesizer']['core']['thickness_m']
    d_shell_m = _CONFIG['monte_carlo_synthesizer']['shell']['thickness_m']
    
    core = PlasmaZoneParams(T_e_K=T_core, T_i_K=T_core*0.8, n_e_cm3=n_e_core, thickness_m=d_core_m, label='Core')
    shell = PlasmaZoneParams(T_e_K=T_shell, T_i_K=T_shell*0.8, n_e_cm3=n_e_shell, thickness_m=d_shell_m, label='Shell')
    
    try:
        # Eksekusi Blok 1
        model = TwoZonePlasma(core, shell, elements, _fetcher)
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
        return idx, theta, I_sim.astype(np.float32)
        
    except Exception as e:
        # Mencegah worker mati akibat matriks Jacobian yang infinitely stiff
        print(f"[Worker] Singularity dihindari pada matriks (idx={idx}, Tc={T_core:.0f}K, nc={n_e_core:.1e}): {str(e)}")
        return idx, None, None

def generate_dataset(
    n_samples: int,
    output_file: str,
    dataset_group: str,
    spectral_tier: str,
    num_workers: int = None,
):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    print(f"=== BLOK 2: Mesin Sintesis Dataset CR-LIBS ===")
    print(f"Target Sampel      : {n_samples} spektrum sintetik")
    print(f"Unit Pekerja (CPU) : {num_workers} cores aktif")
    print(f"File Output (HDF5) : {output_file}")
    print(f"Grup Dataset       : {dataset_group}")
    print(f"Tier Resolusi      : {spectral_tier}")
    
    # Rentang Domain Parameter Termodinamika dari config.yaml (dipaksa ke float agar aman dari string-parsing PyYAML)
    volcanic_profile = _CONFIG["synthetic_dataset_profiles"]["volcanic_soil"]
    group_spec = _load_group_spec(dataset_group)
    bounds = {
        'T_core': tuple(map(float, volcanic_profile['plasma']['T_core_range_K'])),
        'T_shell': tuple(map(float, _CONFIG['monte_carlo_synthesizer']['shell']['T_range_K'])),
        'ne_core': tuple(map(float, volcanic_profile['plasma']['ne_core_range_cm3'])),
        'ne_shell': tuple(map(float, _CONFIG['monte_carlo_synthesizer']['shell']['ne_range_cm3']))
    }

    composition_columns = [f"comp_{elem}_pct" for elem, _ in group_spec.element_bounds_pct]
    parameter_columns = [
        "T_e_core_K",
        "T_e_shell_K",
        "n_e_core_cm3",
        "n_e_shell_cm3",
        *composition_columns,
    ]
    
    # Sampling parameter secara seragam
    rng = np.random.default_rng(42)
    tasks = []

    for i in range(n_samples):
        Tc = rng.uniform(*bounds['T_core'])
        Ts = rng.uniform(*bounds['T_shell'])
        if Ts > Tc:
            Ts = Tc * 0.8

        nc = rng.uniform(*bounds['ne_core'])
        ns = rng.uniform(*bounds['ne_shell'])
        composition_pct = _sample_bounded_simplex_pct(group_spec.element_bounds_pct, rng)
        elements = [
            (elem, 1, float(frac_pct) / 100.0)
            for (elem, _), frac_pct in zip(group_spec.element_bounds_pct, composition_pct)
        ]
        theta = np.asarray([Tc, Ts, nc, ns, *composition_pct.tolist()], dtype=np.float32)
        tasks.append((i, Tc, Ts, nc, ns, elements, theta))
    
    tier_config = SPECTRAL_TIER_CONFIGS[spectral_tier]
    resolution = int(tier_config["resolution"])
    libs_physics.SIMULATION_CONFIG["resolution"] = resolution
    
    # =========================================================================
    # SOLUSI MULTI-CORE (APPLE M1 / LINUX):
    # Proses anak (Workers) JANGAN sampai mewarisi file HDF5 yang sedang terbuka.
    # Maka, Pool diciptakan TERLEBIH DAHULU di RAM, baru file HDF5 dibuka.
    # =========================================================================
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(resolution,)) as pool:
        
        # Eksekusi generator stokastik
        iterator = pool.imap_unordered(simulate_single_spectrum, tasks)
        
        # Menulis ke HDF5 secara akumulatif (Append) oleh MASTER PROCESS SAJA
        with h5py.File(output_file, 'a') as f:
            # Cek apakah dataset lama sudah ada
            if 'parameters' in f and 'spectra' in f:
                dset_theta = f['parameters']
                dset_spectra = f['spectra']
                current_size = dset_theta.shape[0]
                existing_columns = list(dset_theta.attrs.get('columns', []))
                if existing_columns != parameter_columns:
                    raise ValueError(
                        "Kolom dataset HDF5 yang ada tidak cocok dengan grup dataset yang diminta."
                    )
                existing_group = dset_theta.attrs.get('dataset_group', '')
                existing_tier = dset_theta.attrs.get('spectral_tier', '')
                if str(existing_group) != group_spec.name:
                    raise ValueError("Dataset HDF5 yang ada berasal dari group komposisi yang berbeda.")
                if str(existing_tier) != spectral_tier:
                    raise ValueError("Dataset HDF5 yang ada berasal dari tier resolusi yang berbeda.")
                if dset_spectra.shape[1] != resolution:
                    raise ValueError("Resolusi spektrum HDF5 yang ada tidak cocok dengan tier yang diminta.")
                
                # Memperpanjang 
                dset_theta.resize((current_size + n_samples, len(parameter_columns)))
                dset_spectra.resize((current_size + n_samples, resolution))
                write_idx = current_size
                print(f"  [Resume] Dataset lama terdeteksi ({current_size} baris). Menambahkan {n_samples} baris baru...")
                
            else:
                dset_theta = f.create_dataset(
                    "parameters",
                    shape=(n_samples, len(parameter_columns)),
                    maxshape=(None, len(parameter_columns)),
                    dtype=np.float32,
                )
                dset_spectra = f.create_dataset("spectra", shape=(n_samples, resolution), maxshape=(None, resolution), dtype=np.float32, 
                                                compression="gzip", compression_opts=4)
                
                dset_theta.attrs['columns'] = parameter_columns
                dset_theta.attrs['dataset_group'] = group_spec.name
                dset_theta.attrs['description'] = group_spec.description
                dset_theta.attrs['spectral_tier'] = spectral_tier
                dset_theta.attrs['spectral_resolution'] = resolution
                dset_spectra.attrs['description'] = 'Normalized synthetic spectra (Instrumental FWHM 0.5nm, 1% Gaussian noise)'
                dset_spectra.attrs['spectral_tier'] = spectral_tier
                dset_spectra.attrs['spectral_resolution'] = resolution
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
                dset_theta.resize((write_idx, len(parameter_columns)))
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
        '--dataset-group',
        type=str,
        default='A',
        choices=['A', 'B'],
        help='Grup dataset: A tanpa trace element, B dengan trace element',
    )
    parser.add_argument(
        '--spectral-tier',
        type=str,
        default='L',
        choices=['L', 'M', 'H'],
        help='Tier resolusi spektrum: L=4000, M=8000, H=16000 titik',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
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
    output_file = args.out or os.path.join(
        _BASE_DIR,
        'data',
        f'dataset_synthetic_{args.dataset_group}_{args.spectral_tier}.h5',
    )
    generate_dataset(
        n_samples=args.samples,
        output_file=output_file,
        dataset_group=args.dataset_group,
        spectral_tier=args.spectral_tier,
        num_workers=args.cores,
    )
