from __future__ import annotations

from dataclasses import asdict
from typing import Any
import os

import numpy as np

from src.core.config import ensure_parent_dir
from src.core.contracts import ReportContext
from src.core.metrics import evaluate_thermo_predictions, print_metrics_summary
from src.core.reporting import infer_dataset_code, write_training_report
from src.core.serialization import load_pipeline_artifact, save_pipeline_artifact
from src.data.feature_selection import apply_mrmr_selection
from src.data.io import build_dataset_split, load_dataset_bundle
from src.libs_inversion import (
    HierarchicalInversionConfig,
    HierarchicalPIInverter,
    ThermoPhaseDataset,
)


class PIModel:
    model_name = "pi"

    def __init__(
        self,
        project_config: dict[str, Any],
        *,
        model_params: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
        preprocessing_params: dict[str, Any] | None = None,
    ) -> None:
        self.project_config = project_config
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.preprocessing_params = preprocessing_params or {}
        self.inverter: HierarchicalPIInverter | None = None
        self.dataset = None
        self.selected_indices: np.ndarray | None = None
        self.mrmr_params: dict[str, Any] | None = None
        self.metrics: dict[str, float] | None = None

    def _build_training_config(self, *, input_dim: int, composition_dim: int, overrides: dict[str, Any]) -> HierarchicalInversionConfig:
        merged = dict(self.model_params)
        merged.update(self.training_params)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return HierarchicalInversionConfig.from_config(
            input_dim=input_dim,
            use_composition_head=composition_dim > 0,
            composition_dim=composition_dim,
            **merged,
        )

    def _build_phase_dataset(self, dataset, indices: np.ndarray, config: HierarchicalInversionConfig) -> ThermoPhaseDataset:
        return ThermoPhaseDataset(
            spectra=dataset.spectra[indices],
            temperatures_K=dataset.temperatures_K[indices],
            electron_densities_cm3=dataset.electron_densities_cm3[indices],
            config=config,
            compositions=dataset.compositions[indices] if dataset.compositions is not None else None,
            tau_values=np.zeros(len(indices), dtype=np.float32),
        )

    def fit(
        self,
        dataset_file: str,
        output_model: str,
        report_file: str | None = None,
        **kwargs,
    ) -> dict[str, float]:
        print("=== BLOK 3: Hierarchical PI Inversion (Phase 1) ===")
        dataset = load_dataset_bundle(dataset_file)
        split = build_dataset_split(dataset)
        print(f"Skema partisi data: {len(split.train_indices)} LATIH | {len(split.test_indices)} UJI.")

        preprocessing = dict(self.preprocessing_params)
        cli_mrmr = bool(kwargs.pop("use_mrmr", False))
        mrmr_cfg = dict(preprocessing.get("mrmr", {}))
        mrmr_enabled = cli_mrmr or bool(mrmr_cfg.get("enabled", False))
        if mrmr_enabled:
            mrmr_cfg.update(
                {
                    "enabled": True,
                    "n_features": kwargs.pop("mrmr_features", mrmr_cfg.get("n_features", 1024)),
                    "pool_size": kwargs.pop("mrmr_pool", mrmr_cfg.get("pool_size", 2048)),
                    "sample_size": kwargs.pop("mrmr_sample", mrmr_cfg.get("sample_size", 2000)),
                    "redundancy_weight": kwargs.pop(
                        "mrmr_redundancy_weight", mrmr_cfg.get("redundancy_weight", 1.0)
                    ),
                    "random_state": kwargs.pop("mrmr_random_state", mrmr_cfg.get("random_state", 42)),
                    "score_mode": kwargs.pop("mrmr_score_mode", mrmr_cfg.get("score_mode", "mifs")),
                    "prefilter_stride": kwargs.pop(
                        "mrmr_prefilter_stride", mrmr_cfg.get("prefilter_stride", 1)
                    ),
                }
            )
            dataset, self.selected_indices, self.mrmr_params = apply_mrmr_selection(
                dataset,
                split,
                n_features=int(mrmr_cfg["n_features"]),
                pool_size=int(mrmr_cfg["pool_size"]),
                sample_size=int(mrmr_cfg["sample_size"]) if mrmr_cfg["sample_size"] is not None else None,
                redundancy_weight=float(mrmr_cfg["redundancy_weight"]),
                random_state=int(mrmr_cfg["random_state"]),
                score_mode=str(mrmr_cfg["score_mode"]),
                prefilter_stride=int(mrmr_cfg["prefilter_stride"]),
            )
            print(f"[mRMR] Dimensi input turun menjadi {dataset.spectra.shape[1]} fitur.")

        composition_dim = dataset.compositions.shape[1] if dataset.compositions is not None else 0
        config = self._build_training_config(
            input_dim=dataset.spectra.shape[1],
            composition_dim=composition_dim,
            overrides=kwargs,
        )
        train_dataset = self._build_phase_dataset(dataset, split.train_indices, config)
        inverter = HierarchicalPIInverter(config)
        history = inverter.fit(train_dataset, epochs=config.epochs)
        print(f"Loss epoch terakhir: {history.losses[-1]:.4e}")

        predictions = inverter.predict_thermo(dataset.spectra[split.test_indices])
        metrics = evaluate_thermo_predictions(predictions, dataset, split)
        print_metrics_summary(metrics, has_holdout=split.has_holdout)

        self.inverter = inverter
        self.dataset = dataset
        self.metrics = metrics
        self.save(output_model)
        if report_file is None:
            report_file = os.path.splitext(output_model)[0] + "_report.txt"
        self._write_report(report_file, dataset_file, output_model, split, config)
        return metrics

    def predict(self, spectra: np.ndarray) -> dict[str, np.ndarray]:
        if self.inverter is None:
            raise RuntimeError("Model belum dimuat atau dilatih.")
        return self.inverter.predict_thermo(spectra)

    def save(self, output_model: str) -> None:
        if self.inverter is None or self.dataset is None:
            raise RuntimeError("Tidak ada model terlatih untuk disimpan.")
        ensure_parent_dir(output_model)
        payload = {
            "model_type": "hierarchical_pi_inverter_phase1",
            "model_family": self.model_name,
            "config": asdict(self.inverter.config),
            "state_dict": self.inverter.model.state_dict(),
            "metrics": self.metrics or {},
            "targets": ["T_e_core_K", "n_e_core_cm3"],
            "composition_columns": list(self.dataset.composition_columns or []),
        }
        if self.selected_indices is not None:
            payload["selected_indices"] = self.selected_indices.astype(int).tolist()
        if self.mrmr_params is not None:
            payload["mrmr"] = dict(self.mrmr_params)
        save_pipeline_artifact(output_model, payload)

    def _write_report(
        self,
        report_file: str,
        dataset_file: str,
        output_model: str,
        split,
        config: HierarchicalInversionConfig,
    ) -> None:
        if self.dataset is None or self.metrics is None:
            raise RuntimeError("Report tidak bisa ditulis sebelum training selesai.")
        context = ReportContext(
            dataset_code=infer_dataset_code(dataset_file),
            dataset_file=dataset_file,
            model_output=output_model,
            model_type="hierarchical_pi_inverter_phase1",
            pipeline="mRMR + PI" if self.selected_indices is not None else "PI",
            feature_mode="mrmr" if self.selected_indices is not None else "plain",
            sample_count=self.dataset.spectra.shape[0],
            input_dim=self.dataset.spectra.shape[1],
            train_count=len(split.train_indices),
            test_count=len(split.test_indices),
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            physics_informed=True,
            target_composition_count=0 if self.dataset.composition_columns is None else len(self.dataset.composition_columns),
            composition_columns=self.dataset.composition_columns,
            extra_model_lines=self._extra_model_lines(),
        )
        write_training_report(report_file, context, self.metrics)

    def _extra_model_lines(self) -> list[tuple[str, object]]:
        if self.selected_indices is None or self.mrmr_params is None:
            return [("mRMR", "tidak digunakan")]
        return [
            ("mRMR score mode", self.mrmr_params.get("score_mode")),
            ("Fitur terpilih", len(self.selected_indices)),
            ("mRMR pool size", self.mrmr_params.get("pool_size")),
            ("mRMR sample size", self.mrmr_params.get("sample_size")),
            ("mRMR prefilter stride", self.mrmr_params.get("prefilter_stride")),
        ]

    @classmethod
    def load(cls, path: str) -> "PIModel":
        checkpoint = load_pipeline_artifact(path)
        instance = cls(project_config={})
        instance.inverter = HierarchicalPIInverter.from_checkpoint(checkpoint)
        selected = checkpoint.get("selected_indices")
        if selected is not None:
            instance.selected_indices = np.asarray(selected, dtype=int)
        instance.metrics = checkpoint.get("metrics", {})
        return instance

