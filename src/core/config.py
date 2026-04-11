from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import yaml


def get_base_dir() -> Path:
    env = os.environ.get("LIBS_BASE_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file harus berupa mapping YAML: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_tree(config_path: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], Path]:
    base_dir = get_base_dir()
    path = Path(config_path).resolve() if config_path else (base_dir / "configs" / "base.yaml")
    data = _load_yaml(path)

    merged: dict[str, Any] = {}
    base_ref = data.pop("base_config", None)
    if base_ref:
        parent_cfg, _ = load_config_tree(path.parent / str(base_ref))
        merged = _deep_merge(merged, parent_cfg)

    legacy_ref = data.pop("legacy_config_path", None)
    if legacy_ref:
        legacy_path = (path.parent / str(legacy_ref)).resolve()
        merged = _deep_merge(merged, _load_yaml(legacy_path))

    merged = _deep_merge(merged, data)
    merged.setdefault("_meta", {})
    merged["_meta"]["config_path"] = str(path)
    merged["_meta"]["base_dir"] = str(base_dir)
    return merged, path


def resolve_path(value: str | None, *, base_dir: str | os.PathLike[str], config_dir: str | os.PathLike[str] | None = None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if config_dir is not None:
        candidate = Path(config_dir) / path
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path(base_dir) / path).resolve())


def ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

