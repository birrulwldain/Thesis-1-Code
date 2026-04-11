from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetBundle:
    spectra: np.ndarray
    temperatures_K: np.ndarray
    electron_densities_cm3: np.ndarray
    parameter_columns: list[str]
    compositions: np.ndarray | None = None
    composition_columns: list[str] | None = None


@dataclass
class SplitBundle:
    train_indices: np.ndarray
    test_indices: np.ndarray
    has_holdout: bool


@dataclass
class ReportContext:
    dataset_code: str
    dataset_file: str
    model_output: str
    model_type: str
    pipeline: str
    feature_mode: str
    sample_count: int
    input_dim: int
    train_count: int
    test_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    physics_informed: bool
    target_thermo_count: int = 2
    target_composition_count: int = 0
    composition_columns: list[str] | None = None
    extra_model_lines: list[tuple[str, object]] | None = None

