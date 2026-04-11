from __future__ import annotations

import numpy as np

from src.core.contracts import DatasetBundle, SplitBundle
from src.mrmr import MRMRConfig, compute_mrmr_indices


def build_prefilter_indices(input_dim: int, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.arange(input_dim, dtype=int)
    indices = np.arange(0, input_dim, stride, dtype=int)
    if indices[-1] != input_dim - 1:
        indices = np.concatenate([indices, np.asarray([input_dim - 1], dtype=int)])
    return indices


def apply_mrmr_selection(
    dataset: DatasetBundle,
    split: SplitBundle,
    *,
    n_features: int,
    pool_size: int,
    sample_size: int | None,
    redundancy_weight: float,
    random_state: int,
    score_mode: str,
    prefilter_stride: int,
) -> tuple[DatasetBundle, np.ndarray, dict[str, int | float | None | str]]:
    y_targets = np.stack([dataset.temperatures_K, dataset.electron_densities_cm3], axis=1)
    prefilter_indices = build_prefilter_indices(dataset.spectra.shape[1], int(prefilter_stride))
    x_train = dataset.spectra[split.train_indices][:, prefilter_indices]
    y_train = y_targets[split.train_indices]
    print(
        f"[mRMR] Prefilter stride={int(prefilter_stride)} "
        f"menurunkan kandidat awal dari {dataset.spectra.shape[1]} ke {len(prefilter_indices)} fitur."
    )
    config = MRMRConfig(
        n_features=int(n_features),
        pool_size=int(pool_size),
        sample_size=int(sample_size) if sample_size is not None else None,
        redundancy_weight=float(redundancy_weight),
        random_state=int(random_state),
        score_mode=str(score_mode),
    )
    selected_local = compute_mrmr_indices(x_train, y_train, config)
    selected_indices = prefilter_indices[selected_local]
    selected_dataset = DatasetBundle(
        spectra=dataset.spectra[:, selected_indices],
        temperatures_K=dataset.temperatures_K,
        electron_densities_cm3=dataset.electron_densities_cm3,
        parameter_columns=dataset.parameter_columns,
        compositions=dataset.compositions,
        composition_columns=dataset.composition_columns,
    )
    params = {
        "n_features": int(n_features),
        "pool_size": int(pool_size),
        "sample_size": int(sample_size) if sample_size is not None else None,
        "redundancy_weight": float(redundancy_weight),
        "random_state": int(random_state),
        "score_mode": str(score_mode),
        "prefilter_stride": int(prefilter_stride),
    }
    return selected_dataset, selected_indices, params

