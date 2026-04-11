from __future__ import annotations

import os

import h5py
import numpy as np

from src.core.contracts import DatasetBundle, SplitBundle


def _decode_columns(raw_columns) -> list[str]:
    return [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in raw_columns]


def load_dataset_bundle(dataset_file: str) -> DatasetBundle:
    print(f"Memuat dataset sintetik dari: {dataset_file}")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset {dataset_file} tidak ditemukan.")

    with h5py.File(dataset_file, "r") as handle:
        if "spectra" not in handle or "parameters" not in handle:
            raise KeyError("Dataset HDF5 wajib memiliki dataset 'spectra' dan 'parameters'.")
        spectra = handle["spectra"][:]
        parameters = handle["parameters"][:]
        raw_columns = handle["parameters"].attrs.get("columns")
        if raw_columns is None:
            raise KeyError("Atribut 'columns' tidak ditemukan pada dataset 'parameters'.")

    columns = _decode_columns(raw_columns)
    if spectra.ndim != 2 or parameters.ndim != 2:
        raise ValueError("Dataset 'spectra' dan 'parameters' harus berupa matriks 2D.")
    if spectra.shape[0] != parameters.shape[0]:
        raise ValueError("Jumlah sampel antara 'spectra' dan 'parameters' harus sama.")

    try:
        temp_idx = columns.index("T_e_core_K")
        ne_idx = columns.index("n_e_core_cm3")
    except ValueError as exc:
        raise KeyError(
            "Dataset harus memiliki kolom 'T_e_core_K' dan 'n_e_core_cm3' pada atribut columns."
        ) from exc

    temperatures_K = parameters[:, temp_idx].astype(np.float32)
    electron_densities_cm3 = parameters[:, ne_idx].astype(np.float32)
    composition_columns = [col for col in columns if str(col).startswith("comp_")]
    compositions = None
    if composition_columns:
        comp_indices = [columns.index(col) for col in composition_columns]
        compositions = parameters[:, comp_indices].astype(np.float32)
        row_sums = np.clip(compositions.sum(axis=1, keepdims=True), 1e-8, None)
        compositions = compositions / row_sums

    print(
        f"Dataset sukses dimuat. Spektra={spectra.shape}, "
        f"parameter termodinamika={(temperatures_K.shape[0], 2)}"
    )
    if composition_columns:
        print(f"Komposisi unsur terdeteksi: {len(composition_columns)} kolom")
    return DatasetBundle(
        spectra=spectra.astype(np.float32),
        temperatures_K=temperatures_K,
        electron_densities_cm3=electron_densities_cm3,
        parameter_columns=columns,
        compositions=compositions,
        composition_columns=composition_columns or None,
    )


def build_dataset_split(dataset: DatasetBundle, random_state: int = 42) -> SplitBundle:
    n_samples = dataset.spectra.shape[0]
    indices = np.arange(n_samples, dtype=int)
    if n_samples < 5:
        print("Peringatan: dataset kecil; evaluasi dilakukan pada data latih.")
        return SplitBundle(train_indices=indices, test_indices=indices, has_holdout=False)

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(indices)
    split_at = max(1, int(round(0.8 * n_samples)))
    split_at = min(split_at, n_samples - 1)
    return SplitBundle(
        train_indices=shuffled[:split_at],
        test_indices=shuffled[split_at:],
        has_holdout=True,
    )

