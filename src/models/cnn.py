from __future__ import annotations

from typing import Any


class CNNModel:
    model_name = "cnn"

    def __init__(self, project_config: dict[str, Any], **kwargs) -> None:
        self.project_config = project_config
        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        raise NotImplementedError("CNNModel belum diimplementasikan.")

    def predict(self, spectra):
        raise NotImplementedError("CNNModel belum diimplementasikan.")

    def save(self, output_model: str) -> None:
        raise NotImplementedError("CNNModel belum diimplementasikan.")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("CNNModel belum diimplementasikan.")

