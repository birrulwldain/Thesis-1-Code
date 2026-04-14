from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR as SklearnSVR

from src.core.config import ensure_parent_dir
from src.core.contracts import ReportContext
from src.core.metrics import evaluate_thermo_predictions, print_metrics_summary
from src.core.reporting import infer_dataset_code, write_training_report
from src.core.serialization import load_pipeline_artifact, save_pipeline_artifact
from src.data.io import build_dataset_split, load_dataset_bundle


class SVRModel:
    model_name = "svr"

    def __init__(self, project_config: dict[str, Any], **kwargs) -> None:
        self.project_config = project_config
        self.model_params = dict(kwargs.get("model_params") or {})
        self.training_params = dict(kwargs.get("training_params") or {})
        self.preprocessing_params = dict(kwargs.get("preprocessing_params") or {})

        self.physics_extractor: Pipeline | None = None
        self.scaler_X: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        self.model: MultiOutputRegressor | None = None
        self.dataset = None
        self.metrics: dict[str, float] | None = None
        self.columns = ["T_e_core_K", "n_e_core_cm3"]
        self.prediction_keys = ["T_e_K", "n_e_cm3"]

    def _resolve_pca_components(self, input_dim: int, train_size: int) -> int | None:
        candidate = (
            self.preprocessing_params.get("pca", {}).get("n_components")
            or self.model_params.get("pca_components")
            or self.training_params.get("pca_components")
        )
        if candidate is None:
            candidate = min(64, input_dim)
        candidate = int(candidate)
        max_allowed = max(1, min(input_dim, train_size - 1 if train_size > 1 else 1))
        if candidate >= max_allowed:
            return None
        return max(1, min(candidate, max_allowed - 1))

    def _build_feature_extractor(self, *, input_dim: int, train_size: int) -> Pipeline:
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
        n_components = self._resolve_pca_components(input_dim, train_size)
        if n_components is not None:
            steps.append(("pca", PCA(n_components=n_components, random_state=42)))
        return Pipeline(steps)

    def _build_svr_estimator(self, **overrides: Any) -> MultiOutputRegressor:
        merged = dict(self.model_params)
        merged.update(self.training_params)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        base_estimator = SklearnSVR(
            kernel=str(merged.get("kernel", "rbf")),
            C=float(merged.get("C", 10.0)),
            epsilon=float(merged.get("epsilon", 0.05)),
            gamma=merged.get("gamma", "scale"),
            degree=int(merged.get("degree", 3)),
            coef0=float(merged.get("coef0", 0.0)),
            tol=float(merged.get("tol", 1e-3)),
            cache_size=float(merged.get("cache_size", 200.0)),
            shrinking=bool(merged.get("shrinking", True)),
        )
        return MultiOutputRegressor(base_estimator, n_jobs=int(merged.get("n_jobs", 1)))

    def _predict_array(self, spectra: np.ndarray) -> np.ndarray:
        if self.physics_extractor is None or self.scaler_X is None or self.scaler_y is None or self.model is None:
            raise RuntimeError("Model belum dimuat atau dilatih.")
        features = self.physics_extractor.transform(np.asarray(spectra, dtype=np.float32))
        scaled_features = self.scaler_X.transform(features)
        scaled_predictions = self.model.predict(scaled_features)
        return self.scaler_y.inverse_transform(np.asarray(scaled_predictions, dtype=np.float32))

    def fit(self, *args, **kwargs):
        if not args:
            raise TypeError("fit() membutuhkan minimal path dataset.")

        dataset_file = str(args[0])
        output_model = str(args[1]) if len(args) > 1 else str(Path(dataset_file).with_suffix(".pkl"))
        report_file = str(args[2]) if len(args) > 2 and args[2] is not None else None

        dataset = load_dataset_bundle(dataset_file)
        split = build_dataset_split(dataset)
        self.dataset = dataset

        input_dim = int(dataset.spectra.shape[1])
        train_size = int(len(split.train_indices))
        self.physics_extractor = self._build_feature_extractor(input_dim=input_dim, train_size=train_size)

        train_spectra = dataset.spectra[split.train_indices]
        test_spectra = dataset.spectra[split.test_indices]
        train_targets = np.column_stack(
            [dataset.temperatures_K[split.train_indices], dataset.electron_densities_cm3[split.train_indices]]
        )
        test_targets = np.column_stack(
            [dataset.temperatures_K[split.test_indices], dataset.electron_densities_cm3[split.test_indices]]
        )

        train_features = self.physics_extractor.fit_transform(train_spectra)
        self.scaler_X = StandardScaler()
        scaled_train_features = self.scaler_X.fit_transform(train_features)
        self.scaler_y = StandardScaler()
        scaled_train_targets = self.scaler_y.fit_transform(train_targets)

        self.model = self._build_svr_estimator(**kwargs)
        self.model.fit(scaled_train_features, scaled_train_targets)

        scaled_predictions = self.model.predict(
            self.scaler_X.transform(self.physics_extractor.transform(test_spectra))
        )
        prediction_array = self.scaler_y.inverse_transform(np.asarray(scaled_predictions, dtype=np.float32))
        predictions = {
            self.prediction_keys[0]: prediction_array[:, 0],
            self.prediction_keys[1]: prediction_array[:, 1],
            self.columns[0]: prediction_array[:, 0],
            self.columns[1]: prediction_array[:, 1],
        }
        metrics = evaluate_thermo_predictions(predictions, dataset, split)
        print_metrics_summary(metrics, has_holdout=split.has_holdout)

        self.metrics = metrics
        self.save(output_model)
        if report_file is None:
            report_file = os.path.splitext(output_model)[0] + "_report.txt"
        self._write_report(report_file, dataset_file, output_model, split)
        return metrics

    def predict(self, spectra):
        predictions = self._predict_array(np.asarray(spectra, dtype=np.float32))
        return {
            self.prediction_keys[0]: predictions[:, 0],
            self.prediction_keys[1]: predictions[:, 1],
            self.columns[0]: predictions[:, 0],
            self.columns[1]: predictions[:, 1],
        }

    def save(self, output_model: str) -> None:
        if self.physics_extractor is None or self.scaler_X is None or self.scaler_y is None or self.model is None:
            raise RuntimeError("Tidak ada model terlatih untuk disimpan.")
        ensure_parent_dir(output_model)
        payload = {
            "model_type": "legacy_svr",
            "model_family": self.model_name,
            "config": {
                "model_params": self.model_params,
                "training_params": self.training_params,
                "preprocessing_params": self.preprocessing_params,
            },
            "physics_extractor": self.physics_extractor,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "model": self.model,
            "metrics": self.metrics or {},
            "columns": self.columns,
        }
        save_pipeline_artifact(output_model, payload)

    def _write_report(self, report_file: str, dataset_file: str, output_model: str, split) -> None:
        if self.dataset is None or self.metrics is None:
            raise RuntimeError("Report tidak bisa ditulis sebelum training selesai.")

        extra_lines: list[tuple[str, object]] = [
            ("PCA components", "tanpa PCA"),
            ("SVR kernel", self.model_params.get("kernel", "rbf")),
            ("SVR C", self.model_params.get("C", 10.0)),
            ("SVR epsilon", self.model_params.get("epsilon", 0.05)),
        ]
        if self.physics_extractor is not None:
            pca = self.physics_extractor.named_steps.get("pca")
            if pca is not None:
                extra_lines[0] = ("PCA components", pca.n_components_)

        context = ReportContext(
            dataset_code=infer_dataset_code(dataset_file),
            dataset_file=dataset_file,
            model_output=output_model,
            model_type="legacy_svr",
            pipeline="PCA + SVR" if self.physics_extractor and "pca" in self.physics_extractor.named_steps else "SVR",
            feature_mode="pca" if self.physics_extractor and "pca" in self.physics_extractor.named_steps else "plain",
            sample_count=self.dataset.spectra.shape[0],
            input_dim=self.dataset.spectra.shape[1],
            train_count=len(split.train_indices),
            test_count=len(split.test_indices),
            epochs=int(self.training_params.get("epochs", 1)),
            batch_size=int(self.training_params.get("batch_size", len(split.train_indices))),
            learning_rate=float(self.training_params.get("learning_rate", 1.0)),
            physics_informed=False,
            extra_model_lines=extra_lines,
        )
        write_training_report(report_file, context, self.metrics)

    @classmethod
    def load(cls, path: str):
        checkpoint = load_pipeline_artifact(path)
        instance = cls(
            project_config={},
            model_params=dict(checkpoint.get("config", {}).get("model_params", {})),
            training_params=dict(checkpoint.get("config", {}).get("training_params", {})),
            preprocessing_params=dict(checkpoint.get("config", {}).get("preprocessing_params", {})),
        )
        instance.physics_extractor = checkpoint.get("physics_extractor") or checkpoint.get("pca")
        instance.scaler_X = checkpoint.get("scaler_X")
        instance.scaler_y = checkpoint.get("scaler_y")
        instance.model = checkpoint.get("model")
        instance.metrics = checkpoint.get("metrics", {})
        instance.columns = list(checkpoint.get("columns", instance.columns))
        return instance

