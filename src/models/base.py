from __future__ import annotations

from typing import Protocol

import numpy as np


class ExperimentModel(Protocol):
    model_name: str

    def fit(self, dataset_file: str, output_model: str, report_file: str | None = None, **kwargs) -> dict:
        ...

    def predict(self, spectra: np.ndarray) -> dict[str, np.ndarray]:
        ...

    def save(self, output_model: str) -> None:
        ...

    @classmethod
    def load(cls, path: str):
        ...

