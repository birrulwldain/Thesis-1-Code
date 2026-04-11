from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.config import ensure_parent_dir, get_base_dir, load_config_tree, resolve_path
from src.models.registry import create_model


def _nested_get(data: dict[str, Any], *keys: str, default=None):
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_output_path(
    candidate: str | None,
    *,
    default_dir: str,
    default_name: str,
    base_dir: str,
    config_dir: str,
) -> str:
    if candidate:
        resolved = resolve_path(candidate, base_dir=base_dir, config_dir=config_dir)
        assert resolved is not None
        return resolved
    return str((Path(base_dir) / default_dir / default_name).resolve())


def run_training_job(
    *,
    dataset: str | None = None,
    model_name: str | None = None,
    config_path: str | None = None,
    output_model: str | None = None,
    report_file: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    use_mrmr: bool | None = None,
    mrmr_features: int | None = None,
    mrmr_pool: int | None = None,
    mrmr_sample: int | None = None,
    mrmr_redundancy_weight: float | None = None,
    mrmr_random_state: int | None = None,
    mrmr_score_mode: str | None = None,
    mrmr_prefilter_stride: int | None = None,
) -> dict[str, float]:
    project_config, loaded_path = load_config_tree(config_path)
    base_dir = str(get_base_dir())
    config_dir = str(loaded_path.parent)

    selected_model = model_name or _nested_get(project_config, "model", "name", default="pi")
    dataset_path = dataset or _nested_get(project_config, "data", "dataset")
    if dataset_path is None:
        raise ValueError("Dataset path belum ditentukan lewat CLI atau config.")
    resolved_dataset = resolve_path(dataset_path, base_dir=base_dir, config_dir=config_dir)
    assert resolved_dataset is not None

    model_cfg = dict(_nested_get(project_config, "model", "params", default={}) or {})
    training_cfg = dict(_nested_get(project_config, "training", default={}) or {})
    preprocessing_cfg = dict(_nested_get(project_config, "preprocessing", default={}) or {})
    output_cfg = dict(_nested_get(project_config, "outputs", default={}) or {})

    model_basename = Path(resolved_dataset).stem.replace("dataset_synthetic_", f"model_inversi_{selected_model}_")
    model_out = _resolve_output_path(
        output_model or output_cfg.get("model"),
        default_dir="artifacts/models",
        default_name=f"{model_basename}.pkl",
        base_dir=base_dir,
        config_dir=config_dir,
    )
    report_out = _resolve_output_path(
        report_file or output_cfg.get("report"),
        default_dir="artifacts/reports",
        default_name=f"{Path(model_out).stem}_report.txt",
        base_dir=base_dir,
        config_dir=config_dir,
    )
    ensure_parent_dir(model_out)
    ensure_parent_dir(report_out)

    model = create_model(
        selected_model,
        project_config=project_config,
        model_params=model_cfg,
        training_params=training_cfg,
        preprocessing_params=preprocessing_cfg,
    )
    extra_fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if use_mrmr is not None:
        extra_fit_kwargs["use_mrmr"] = use_mrmr
    if mrmr_features is not None:
        extra_fit_kwargs["mrmr_features"] = mrmr_features
    if mrmr_pool is not None:
        extra_fit_kwargs["mrmr_pool"] = mrmr_pool
    if mrmr_sample is not None:
        extra_fit_kwargs["mrmr_sample"] = mrmr_sample
    if mrmr_redundancy_weight is not None:
        extra_fit_kwargs["mrmr_redundancy_weight"] = mrmr_redundancy_weight
    if mrmr_random_state is not None:
        extra_fit_kwargs["mrmr_random_state"] = mrmr_random_state
    if mrmr_score_mode is not None:
        extra_fit_kwargs["mrmr_score_mode"] = mrmr_score_mode
    if mrmr_prefilter_stride is not None:
        extra_fit_kwargs["mrmr_prefilter_stride"] = mrmr_prefilter_stride
    return model.fit(resolved_dataset, model_out, report_out, **extra_fit_kwargs)

