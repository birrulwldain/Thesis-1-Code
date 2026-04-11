from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_pipeline_artifact(output_model: str, payload: dict[str, Any]) -> None:
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_model)
    print(f"[Save] Artefak model disimpan ke: {output_model}")


def load_pipeline_artifact(path: str) -> dict[str, Any]:
    return joblib.load(path)

