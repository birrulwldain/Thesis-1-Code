"""
empirical_validation.py — BLOK 4: End-to-End Empirical Validation
==================================================================
Validates an experimental spectrum using:
1. Phase 1 thermodynamic inversion
2. Phase 2 geometric RTE fitting
3. McWhirter criterion evaluation
"""

from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np
import pandas as pd
import yaml

from src.libs_inversion import HierarchicalPIInverter, Phase2GeometrySolver
from src.libs_physics import SIMULATION_CONFIG


_ENV_BASE_DIR = os.environ.get("LIBS_BASE_DIR")
if _ENV_BASE_DIR and os.path.exists(os.path.join(_ENV_BASE_DIR, "config.yaml")):
    _BASE_DIR = _ENV_BASE_DIR
else:
    _BASE_DIR = _ROOT
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f)


def load_and_preprocess_experimental_data(csv_path: str, target_wavelengths: np.ndarray) -> np.ndarray:
    print(f"[Prep] Membaca data eksperimen dari: {csv_path}")
    df = pd.read_csv(
        csv_path,
        sep=r"[\s,;]+",
        engine="python",
        header=None,
        skiprows=lambda x: x < 5 if "asc" in csv_path.lower() else 0,
    )
    if not isinstance(df.iloc[0, 0], (int, float, np.float64)):
        df = df.iloc[1:].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = [f"Col_{i}" for i in range(df.shape[1])]

    wl_raw = df.iloc[:, 0].values
    I_raw = df.iloc[:, 1].values
    baseline = np.percentile(I_raw, 5)
    I_corr = I_raw - baseline
    I_corr[I_corr < 0] = 0

    print(f"[Prep] Interpolasi spektrum ke resolusi model ({len(target_wavelengths)} piksel)...")
    I_interp = np.interp(target_wavelengths, wl_raw, I_corr)
    m = float(np.max(I_interp))
    if m > 0:
        I_interp = I_interp / m
    return I_interp.astype(np.float32)


def _run_legacy_pipeline(pipeline: dict, spectrum: np.ndarray) -> dict[str, float]:
    physics_extractor = pipeline.get("physics_extractor", None)
    if not physics_extractor:
        physics_extractor = pipeline.get("pca")
    scaler_X = pipeline["scaler_X"]
    scaler_y = pipeline["scaler_y"]
    model = pipeline["model"]
    cols = pipeline["columns"]

    X_features = spectrum.reshape(1, -1)
    X_phys = physics_extractor.transform(X_features)
    X_scaled = scaler_X.transform(X_phys)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    return dict(zip(cols, y_pred))


def _run_pi_pipeline(pipeline: dict, spectrum: np.ndarray) -> dict[str, float]:
    inverter = HierarchicalPIInverter.from_checkpoint(pipeline)
    thermo = inverter.predict_thermo(spectrum)
    geometry_solver = Phase2GeometrySolver(inverter.config)
    geometry = geometry_solver.fit(
        observed_spectrum=spectrum,
        core_temp_K=float(thermo["T_e_K"][0]),
        core_ne_cm3=float(thermo["n_e_cm3"][0]),
    )
    return {
        "T_e_core_K": float(thermo["T_e_K"][0]),
        "n_e_core_cm3": float(thermo["n_e_cm3"][0]),
        "d_core_m": geometry.core_thickness_m,
        "d_shell_m": geometry.shell_thickness_m,
        "tau_shell_max": geometry.tau_shell_max,
        "geometry_mse": geometry.mse,
    }


def _print_ground_truth_benchmark(csv_path: str, results: dict[str, float]) -> None:
    sample_id = None
    filename = os.path.basename(csv_path)
    match = re.search(r"S(\d+)", filename, re.IGNORECASE)
    if match:
        sample_id = f"S{match.group(1).upper()}"

    legacy_pkl = "data/ground_truth_legacy.pkl"
    if not sample_id or not os.path.exists(legacy_pkl):
        return

    try:
        ground_truth = joblib.load(legacy_pkl)
        if sample_id not in ground_truth:
            return
        print(f"\n[Benchmarking] Ditemukan Ground Truth historis untuk '{sample_id}'!")
        saha_Te = ground_truth[sample_id]["T_e_K"]
        saha_ne = ground_truth[sample_id]["n_e_cm3"]
        pred_Te = results.get("T_e_core_K", 0.0)
        pred_ne = results.get("n_e_core_cm3", 0.0)
        err_Te = abs(pred_Te - saha_Te) / saha_Te * 100 if pd.notna(saha_Te) and saha_Te > 0 else float("nan")
        err_ne = abs(pred_ne - saha_ne) / saha_ne * 100 if pd.notna(saha_ne) and saha_ne > 0 else float("nan")

        print("-" * 75)
        print("   === ADU MEKANIK: PI Inverter vs Saha-Boltzmann ===")
        print(f"   {'Parameter':<12} | {'PI Inverter':<17} | {'Saha':<20} | {'Error (%)':<10}")
        print(f"   {'Suhu (Te)':<12} | {pred_Te:<17.0f} | {saha_Te:<20.0f} | {err_Te:.2f}%")
        print(f"   {'Densit(ne)':<12} | {pred_ne:<17.2e} | {saha_ne:<20.2e} | {err_ne:.2f}%")
        print("-" * 75)
    except Exception as exc:
        print(f"[Warning] Gagal memuat fitur perbandingan: {exc}")


def _print_mcwhirter(results: dict[str, float]) -> None:
    print("\n[Fisika] Evaluasi Hukum Termodinamika (Kriteria McWhirter)...")
    T_core = results.get("T_e_core_K", 12000.0)
    ne_core = results.get("n_e_core_cm3", 1e17)
    delta_E_eV = _CONFIG["plasma_target"]["mcwhirter_delta_E_eV"]
    mcwhirter_limit = 1.6e12 * np.sqrt(T_core) * (delta_E_eV ** 3)

    print("-" * 60)
    print(f"Temperatur Elektron (T_e): {T_core:.0f} K")
    print(f"Transisi Optik Terlebar (ΔE): {delta_E_eV} eV")
    print(f"Densitas Plasma Terukur (n_e): {ne_core:.2e} cm^-3")
    print(f"Batas Minimum McWhirter:       {mcwhirter_limit:.2e} cm^-3")
    if ne_core >= mcwhirter_limit:
        print("\nKesimpulan: Plasma MEMENUHI syarat LTE.")
    else:
        print("\nKesimpulan: Plasma GAGAL memenuhi syarat LTE.")
        print(">> Paradigma Collisional-Radiative (CR) mutlak diperlukan.")
    print("-" * 60)


def run_empirical_validation(model_pkl: str, csv_path: str) -> dict[str, float]:
    print("=== BLOK 4: Validasi Eksperimental End-to-End ===")
    if not os.path.exists(model_pkl):
        raise FileNotFoundError(f"Model {model_pkl} tidak ditemukan. Harap jalankan Blok 3.")

    print(f"[Inversi] Memuat model: {model_pkl}")
    pipeline = joblib.load(model_pkl)

    wl_min, wl_max = SIMULATION_CONFIG["wl_range_nm"]
    n_pts = SIMULATION_CONFIG["resolution"]
    target_wl = np.linspace(wl_min, wl_max, n_pts, dtype=np.float64)

    if not os.path.exists(csv_path):
        print(f"\nPeringatan: Data spektrum {csv_path} tidak ditemukan.")
        print(">> Membuat MOCK data untuk demo...")
        mock_wl = np.linspace(wl_min - 10, wl_max + 10, 3000)
        mock_I = np.exp(-((mock_wl - 250) / 0.7) ** 2) * 5000 + np.exp(-((mock_wl - 400) / 1.2) ** 2) * 3500
        mock_I += np.random.normal(200, 50, len(mock_wl))
        pd.DataFrame({"Wavelength": mock_wl, "Intensity": mock_I}).to_csv(csv_path, index=False)
        print(">> MOCK data diciptakan.\n")

    spectrum = load_and_preprocess_experimental_data(csv_path, target_wl)

    model_type = pipeline.get("model_type", "legacy_svr")
    if model_type == "hierarchical_pi_inverter_phase1":
        print("\n[Inversi] Menjalankan Phase 1 + Phase 2 pada PI inverter...")
        results = _run_pi_pipeline(pipeline, spectrum)
    else:
        print("\n[Inversi] Menjalankan pipeline legacy SVR...")
        results = _run_legacy_pipeline(pipeline, spectrum)

    print("\n>>> HASIL TRANSLASI FISIKA <<<")
    for key, value in results.items():
        print(f"    {key:<16} = {value:.4e}")

    _print_ground_truth_benchmark(csv_path, results)
    _print_mcwhirter(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blok 4: Validasi termodinamika + geometri spektrum empiris"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(_BASE_DIR, "artifacts", "models", "model_inversi_pi.pkl"),
        help="File model inversi (.pkl)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(_BASE_DIR, "data", "data_eksperimen_mock.csv"),
        help="File data CSV/ASC eksperimental",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=_BASE_DIR,
        help="Base directory project (override LIBS_BASE_DIR)",
    )
    args = parser.parse_args()
    if args.base_dir != _BASE_DIR:
        _BASE_DIR = args.base_dir
    run_empirical_validation(args.model, args.csv)
