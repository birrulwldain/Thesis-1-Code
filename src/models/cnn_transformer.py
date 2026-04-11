from __future__ import annotations

from typing import Any


class CNNTransformerModel:
    model_name = "cnn_transformer"

    def __init__(self, project_config: dict[str, Any], **kwargs) -> None:
        self.project_config = project_config
        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        raise NotImplementedError("CNNTransformerModel belum diimplementasikan.")

    def predict(self, spectra):
        raise NotImplementedError("CNNTransformerModel belum diimplementasikan.")

    def save(self, output_model: str) -> None:
        raise NotImplementedError("CNNTransformerModel belum diimplementasikan.")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("CNNTransformerModel belum diimplementasikan.")

