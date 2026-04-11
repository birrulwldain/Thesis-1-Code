from __future__ import annotations

import numpy as np

from src.core.contracts import DatasetBundle, SplitBundle


def verdict_from_rel_rmse(rel_pct: float) -> str:
    if not np.isfinite(rel_pct):
        return "tidak tersedia"
    if rel_pct <= 10.0:
        return "baik"
    if rel_pct <= 20.0:
        return "cukup"
    return "lemah"


def evaluate_thermo_predictions(
    predictions: dict[str, np.ndarray],
    dataset: DatasetBundle,
    split: SplitBundle,
) -> dict[str, float]:
    indices = split.test_indices
    y_true_temp = dataset.temperatures_K[indices]
    y_true_ne = dataset.electron_densities_cm3[indices]
    y_pred_temp = np.asarray(predictions["T_e_K"], dtype=np.float32)
    y_pred_ne = np.asarray(predictions["n_e_cm3"], dtype=np.float32)

    rmse_temp = float(np.sqrt(np.mean((y_true_temp - y_pred_temp) ** 2)))
    rmse_ne = float(np.sqrt(np.mean((y_true_ne - y_pred_ne) ** 2)))
    temp_span = float(np.max(y_true_temp) - np.min(y_true_temp)) if len(y_true_temp) > 1 else 0.0
    ne_span = float(np.max(y_true_ne) - np.min(y_true_ne)) if len(y_true_ne) > 1 else 0.0
    rel_rmse_temp = (rmse_temp / temp_span * 100.0) if temp_span > 0.0 else float("nan")
    rel_rmse_ne = (rmse_ne / ne_span * 100.0) if ne_span > 0.0 else float("nan")

    metrics = {
        "rmse_T_e_core_K": rmse_temp,
        "rmse_n_e_core_cm3": rmse_ne,
        "rel_rmse_T_e_core_K_pct": rel_rmse_temp,
        "rel_rmse_n_e_core_cm3_pct": rel_rmse_ne,
    }

    if dataset.compositions is not None and dataset.composition_columns and "composition" in predictions:
        y_true_comp = dataset.compositions[indices]
        y_pred_comp = np.asarray(predictions["composition"], dtype=np.float32)
        comp_rmse = np.sqrt(np.mean((y_true_comp - y_pred_comp) ** 2, axis=0))
        metrics["rmse_composition_mean"] = float(np.mean(comp_rmse))
        metrics["rmse_composition_mean_pct"] = float(np.mean(comp_rmse * 100.0))
    return metrics


def print_metrics_summary(metrics: dict[str, float], *, has_holdout: bool) -> None:
    print("\nMetrik Validasi Inversi Termodinamika:")
    print("-" * 92)
    print(f"{'Parameter':<18} | {'RMSE (%)':<14} | {'Verdict':<10}")
    print("-" * 92)
    temp_rel = metrics.get("rel_rmse_T_e_core_K_pct", float("nan"))
    ne_rel = metrics.get("rel_rmse_n_e_core_cm3_pct", float("nan"))
    print(f"{'T_e_core_K':<18} | {temp_rel:<14.2f} | {verdict_from_rel_rmse(temp_rel):<10}")
    print(f"{'n_e_core_cm3':<18} | {ne_rel:<14.2f} | {verdict_from_rel_rmse(ne_rel):<10}")
    if not has_holdout:
        print("[Validasi] Tidak ada holdout set independen; metrik di atas memakai data latih.")
    if "rmse_composition_mean_pct" in metrics:
        comp_rel = metrics["rmse_composition_mean_pct"]
        print(f"{'composition_mean':<18} | {comp_rel:<14.2f} | {verdict_from_rel_rmse(comp_rel):<10}")
    print("-" * 92)

