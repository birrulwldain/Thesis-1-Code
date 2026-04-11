from __future__ import annotations

from typing import Any


class SVRModel:
    model_name = "svr"

    def __init__(self, project_config: dict[str, Any], **kwargs) -> None:
        self.project_config = project_config
        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        raise NotImplementedError("SVRModel belum dimigrasikan ke registry baru.")

    def predict(self, spectra):
        raise NotImplementedError("SVRModel belum dimigrasikan ke registry baru.")

    def save(self, output_model: str) -> None:
        raise NotImplementedError("SVRModel belum dimigrasikan ke registry baru.")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("SVRModel belum dimigrasikan ke registry baru.")

