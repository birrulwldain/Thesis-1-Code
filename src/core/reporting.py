from __future__ import annotations

from pathlib import Path
import numpy as np

from src.core.contracts import ReportContext
from src.core.metrics import verdict_from_rel_rmse


def infer_dataset_code(dataset_file: str) -> str:
    dataset_base = Path(dataset_file).stem
    if dataset_base.startswith("dataset_synthetic_"):
        return dataset_base.replace("dataset_synthetic_", "", 1)
    return dataset_base


def write_training_report(
    report_file: str,
    context: ReportContext,
    metrics: dict[str, float],
) -> None:
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "LAPORAN ILMIAH TRAINING",
        "=" * 72,
        "",
        "Metadata Eksperimen",
        "-" * 72,
        f"Dataset code             : {context.dataset_code}",
        f"Dataset file             : {context.dataset_file}",
        f"Model output             : {context.model_output}",
        f"Model type               : {context.model_type}",
        f"Pipeline                 : {context.pipeline}",
        f"Feature mode             : {context.feature_mode}",
        "",
        "Konfigurasi Data",
        "-" * 72,
        f"Jumlah sampel            : {context.sample_count}",
        f"Dimensi spektrum input   : {context.input_dim}",
        f"Jumlah target termo      : {context.target_thermo_count}",
        f"Jumlah target komposisi  : {context.target_composition_count}",
        f"Split train/test         : {context.train_count}/{context.test_count}",
        "",
        "Konfigurasi Model",
        "-" * 72,
        f"Epochs                   : {context.epochs}",
        f"Batch size               : {context.batch_size}",
        f"Learning rate            : {context.learning_rate}",
        f"Physics-informed         : {'ya' if context.physics_informed else 'tidak'}",
    ]
    for key, value in context.extra_model_lines or []:
        lines.append(f"{key:<25}: {value}")

    lines.extend(
        [
            "",
            "Metrik Validasi",
            "-" * 72,
            f"RMSE T_e_core (%)        : {metrics.get('rel_rmse_T_e_core_K_pct', float('nan')):.2f}",
            f"RMSE n_e_core (%)        : {metrics.get('rel_rmse_n_e_core_cm3_pct', float('nan')):.2f}",
        ]
    )
    if "rmse_composition_mean_pct" in metrics:
        lines.append(f"RMSE komposisi rata2 (%) : {metrics['rmse_composition_mean_pct']:.2f}")

    lines.extend(
        [
            "",
            "Interpretasi",
            "-" * 72,
            f"Verdict T_e              : {verdict_from_rel_rmse(metrics.get('rel_rmse_T_e_core_K_pct', float('nan')))}",
            f"Verdict n_e              : {verdict_from_rel_rmse(metrics.get('rel_rmse_n_e_core_cm3_pct', float('nan')))}",
        ]
    )
    if "rmse_composition_mean_pct" in metrics:
        lines.append(f"Verdict komposisi        : {verdict_from_rel_rmse(metrics['rmse_composition_mean_pct'])}")
    if context.composition_columns:
        lines.extend(
            [
                "",
                "Komponen Komposisi",
                "-" * 72,
                ", ".join(context.composition_columns),
            ]
        )
    lines.append("")

    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"[Report] Laporan ilmiah disimpan ke: {report_file}")

