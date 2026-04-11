from __future__ import annotations

from typing import Any

from src.models.cnn import CNNModel
from src.models.cnn_transformer import CNNTransformerModel
from src.models.pi import PIModel
from src.models.svr import SVRModel


MODEL_REGISTRY = {
    "pi": PIModel,
    "svr": SVRModel,
    "cnn": CNNModel,
    "cnn_transformer": CNNTransformerModel,
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def create_model(
    model_name: str,
    *,
    project_config: dict[str, Any],
    model_params: dict[str, Any] | None = None,
    training_params: dict[str, Any] | None = None,
    preprocessing_params: dict[str, Any] | None = None,
):
    normalized = str(model_name).strip().lower()
    try:
        model_cls = MODEL_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. Pilihan: {', '.join(available_models())}"
        ) from exc
    return model_cls(
        project_config,
        model_params=model_params,
        training_params=training_params,
        preprocessing_params=preprocessing_params,
    )

