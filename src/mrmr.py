"""
mrmr.py — Minimal mRMR feature selection utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_regression


@dataclass
class MRMRConfig:
    n_features: int
    pool_size: int = 2048
    sample_size: int | None = 2000
    redundancy_weight: float = 1.0
    random_state: int = 42
    n_neighbors: int = 3
    score_mode: str = "mifs"
    miq_epsilon: float = 1e-6


def _subsample_for_mi(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_size is None or sample_size <= 0 or sample_size >= X.shape[0]:
        return X, y
    idx = rng.choice(X.shape[0], size=sample_size, replace=False)
    return X[idx], y[idx]


def _compute_relevance(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_neighbors: int = 3,
) -> np.ndarray:
    if y.ndim == 1:
        return mutual_info_regression(X, y, random_state=random_state, n_neighbors=n_neighbors)
    relevance = np.zeros(X.shape[1], dtype=np.float64)
    for i in range(y.shape[1]):
        relevance += mutual_info_regression(
            X, y[:, i], random_state=random_state, n_neighbors=n_neighbors
        )
    return relevance / float(y.shape[1])


def compute_mrmr_indices(
    X: np.ndarray,
    y: np.ndarray,
    config: MRMRConfig,
    *,
    progress: bool = True,
) -> np.ndarray:
    if config.n_features <= 0:
        raise ValueError("mRMR n_features harus > 0.")
    if config.pool_size <= 0:
        raise ValueError("mRMR pool_size harus > 0.")
    score_mode = config.score_mode.lower()
    if score_mode not in {"mifs", "miq"}:
        raise ValueError("mRMR score_mode harus 'mifs' atau 'miq'.")

    rng = np.random.default_rng(config.random_state)
    X_mi, y_mi = _subsample_for_mi(X, y, config.sample_size, rng)
    relevance = _compute_relevance(
        X_mi, y_mi, random_state=config.random_state, n_neighbors=config.n_neighbors
    )

    pool_size = min(config.pool_size, X.shape[1])
    if pool_size < config.n_features:
        pool_size = config.n_features

    candidate_idx = np.argsort(relevance)[::-1][:pool_size]
    selected: list[int] = []
    if progress:
        print(
            f"[mRMR] Memilih {config.n_features} fitur dari kandidat {pool_size} "
            f"(sample MI={X_mi.shape[0]} baris)."
        )

    for step in range(config.n_features):
        if not selected:
            best = int(candidate_idx[0])
            selected.append(best)
            continue

        best_score = -np.inf
        best_idx = None
        for idx in candidate_idx:
            if idx in selected:
                continue
            redundancy = 0.0
            for sel in selected:
                mi = mutual_info_regression(
                    X_mi[:, [idx]],
                    X_mi[:, sel],
                    random_state=config.random_state,
                    n_neighbors=config.n_neighbors,
                )
                redundancy += float(mi[0])
            redundancy /= float(len(selected))
            if score_mode == "miq":
                score = float(relevance[idx]) / (redundancy + config.miq_epsilon)
            else:
                score = float(relevance[idx]) - config.redundancy_weight * redundancy
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        if best_idx is None:
            break
        selected.append(best_idx)
        if progress and ((step + 1) % 50 == 0 or (step + 1) == config.n_features):
            print(f"[mRMR] Progress: {step + 1}/{config.n_features} fitur terpilih")

    return np.asarray(selected, dtype=int)
