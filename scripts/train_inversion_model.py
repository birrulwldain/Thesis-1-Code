"""
train_inversion_model.py — BLOK 3: Hierarchical PI Inversion Training
=====================================================================
Q1 Thesis: Phase-1 thermodynamic training pipeline for the PyTorch-based
physics-informed inverter defined in src.libs_inversion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import h5py
import joblib
import numpy as np
import os
import yaml

from src.mrmr import MRMRConfig, compute_mrmr_indices

from src.libs_inversion import (
    HierarchicalInversionConfig,
    HierarchicalPIInverter,
    ThermoPhaseDataset,
)


_BASE_DIR = os.environ.get(
    "LIBS_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f)


@dataclass
class DatasetBundle:
    spectra: np.ndarray
    temperatures_K: np.ndarray
    electron_densities_cm3: np.ndarray
    parameter_columns: list[str]


@dataclass
class SplitBundle:
    train_indices: np.ndarray
    test_indices: np.ndarray
    has_holdout: bool


class PIInversionTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config

    @staticmethod
    def _decode_columns(raw_columns) -> list[str]:
        return [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in raw_columns]

    def load_dataset(self, dataset_file: str) -> DatasetBundle:
        print(f"Memuat dataset sintetik dari: {dataset_file}")
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(
                f"Dataset {dataset_file} tidak ditemukan. Jalankan BLOK 2 terlebih dahulu."
            )

        with h5py.File(dataset_file, "r") as f:
            if "spectra" not in f or "parameters" not in f:
                raise KeyError("Dataset HDF5 wajib memiliki dataset 'spectra' dan 'parameters'.")
            spectra = f["spectra"][:]
            parameters = f["parameters"][:]
            raw_columns = f["parameters"].attrs.get("columns")
            if raw_columns is None:
                raise KeyError("Atribut 'columns' tidak ditemukan pada dataset 'parameters'.")

        columns = self._decode_columns(raw_columns)
        if spectra.ndim != 2 or parameters.ndim != 2:
            raise ValueError("Dataset 'spectra' dan 'parameters' harus berupa matriks 2D.")
        if spectra.shape[0] != parameters.shape[0]:
            raise ValueError("Jumlah sampel antara 'spectra' dan 'parameters' harus sama.")

        try:
            temp_idx = columns.index("T_e_core_K")
            ne_idx = columns.index("n_e_core_cm3")
        except ValueError as exc:
            raise KeyError(
                "Dataset harus memiliki kolom 'T_e_core_K' dan 'n_e_core_cm3' pada atribut columns."
            ) from exc

        temperatures_K = parameters[:, temp_idx].astype(np.float32)
        electron_densities_cm3 = parameters[:, ne_idx].astype(np.float32)

        print(
            f"Dataset sukses dimuat. Spektra={spectra.shape}, "
            f"parameter termodinamika={(temperatures_K.shape[0], 2)}"
        )
        return DatasetBundle(
            spectra=spectra.astype(np.float32),
            temperatures_K=temperatures_K,
            electron_densities_cm3=electron_densities_cm3,
            parameter_columns=columns,
        )

    def split_dataset(self, dataset: DatasetBundle) -> SplitBundle:
        n_samples = dataset.spectra.shape[0]
        indices = np.arange(n_samples, dtype=int)
        if n_samples < 5:
            print("Peringatan: dataset kecil; evaluasi dilakukan pada data latih.")
            return SplitBundle(
                train_indices=indices,
                test_indices=indices,
                has_holdout=False,
            )

        rng = np.random.default_rng(42)
        shuffled = rng.permutation(indices)
        split_at = max(1, int(round(0.8 * n_samples)))
        split_at = min(split_at, n_samples - 1)
        return SplitBundle(
            train_indices=shuffled[:split_at],
            test_indices=shuffled[split_at:],
            has_holdout=True,
        )

    @staticmethod
    def build_prefilter_indices(input_dim: int, stride: int) -> np.ndarray:
        if stride <= 1:
            return np.arange(input_dim, dtype=int)
        indices = np.arange(0, input_dim, stride, dtype=int)
        if indices[-1] != input_dim - 1:
            indices = np.concatenate([indices, np.asarray([input_dim - 1], dtype=int)])
        return indices


    def build_training_config(
        self,
        input_dim: int,
        epochs: int | None,
        batch_size: int | None,
        learning_rate: float | None,
    ) -> HierarchicalInversionConfig:
        overrides = {}
        if epochs is not None:
            overrides["epochs"] = epochs
        if batch_size is not None:
            overrides["batch_size"] = batch_size
        if learning_rate is not None:
            overrides["learning_rate"] = learning_rate
        return HierarchicalInversionConfig.from_config(input_dim=input_dim, **overrides)

    def build_phase_dataset(
        self,
        dataset: DatasetBundle,
        indices: np.ndarray,
        inversion_config: HierarchicalInversionConfig,
    ) -> ThermoPhaseDataset:
        return ThermoPhaseDataset(
            spectra=dataset.spectra[indices],
            temperatures_K=dataset.temperatures_K[indices],
            electron_densities_cm3=dataset.electron_densities_cm3[indices],
            config=inversion_config,
            tau_values=np.zeros(len(indices), dtype=np.float32),
        )

    def evaluate_model(
        self,
        inverter: HierarchicalPIInverter,
        dataset: DatasetBundle,
        split: SplitBundle,
    ) -> dict[str, float]:
        indices = split.test_indices
        preds = inverter.predict_thermo(dataset.spectra[indices])
        y_true_temp = dataset.temperatures_K[indices]
        y_true_ne = dataset.electron_densities_cm3[indices]
        y_pred_temp = preds["T_e_K"].astype(np.float32)
        y_pred_ne = preds["n_e_cm3"].astype(np.float32)

        rmse_temp = float(np.sqrt(np.mean((y_true_temp - y_pred_temp) ** 2)))
        rmse_ne = float(np.sqrt(np.mean((y_true_ne - y_pred_ne) ** 2)))
        temp_span = float(np.max(y_true_temp) - np.min(y_true_temp)) if len(y_true_temp) > 1 else 0.0
        ne_span = float(np.max(y_true_ne) - np.min(y_true_ne)) if len(y_true_ne) > 1 else 0.0
        rel_rmse_temp = (rmse_temp / temp_span * 100.0) if temp_span > 0.0 else float("nan")
        rel_rmse_ne = (rmse_ne / ne_span * 100.0) if ne_span > 0.0 else float("nan")

        def verdict(rel_pct: float) -> str:
            if not np.isfinite(rel_pct):
                return "tidak tersedia"
            if rel_pct <= 10.0:
                return "baik"
            if rel_pct <= 20.0:
                return "cukup"
            return "lemah"

        print("\nMetrik Validasi Inversi Termodinamika (Phase 1):")
        print("-" * 92)
        print(f"{'Parameter':<18} | {'RMSE':<14} | {'Rel. RMSE (%)':<14} | {'Verdict':<10}")
        print("-" * 92)
        print(f"{'T_e_core_K':<18} | {rmse_temp:<14.2e} | {rel_rmse_temp:<14.2f} | {verdict(rel_rmse_temp):<10}")
        print(f"{'n_e_core_cm3':<18} | {rmse_ne:<14.2e} | {rel_rmse_ne:<14.2f} | {verdict(rel_rmse_ne):<10}")
        if not split.has_holdout:
            print("[Validasi] Tidak ada holdout set independen; metrik di atas memakai data latih.")
        print("-" * 92)
        print(
            f"[Interpretasi] Span uji: T_e={temp_span:.2e} K, "
            f"n_e={ne_span:.2e} cm^-3"
        )

        return {
            "rmse_T_e_core_K": rmse_temp,
            "rmse_n_e_core_cm3": rmse_ne,
            "rel_rmse_T_e_core_K_pct": rel_rmse_temp,
            "rel_rmse_n_e_core_cm3_pct": rel_rmse_ne,
        }

    def save_pipeline(
        self,
        output_model: str,
        inverter: HierarchicalPIInverter,
        inversion_config: HierarchicalInversionConfig,
        metrics: dict[str, float],
        selected_indices: np.ndarray | None = None,
        mrmr_params: dict | None = None,
    ) -> None:
        print(f"\n[Menyimpan] Mengekspor model PI inverter ke: {output_model} ...")
        pipeline = {
            "model_type": "hierarchical_pi_inverter_phase1",
            "config": asdict(inversion_config),
            "state_dict": inverter.model.state_dict(),
            "metrics": metrics,
            "targets": ["T_e_core_K", "n_e_core_cm3"],
        }
        if selected_indices is not None:
            pipeline["selected_indices"] = selected_indices.astype(int).tolist()
        if mrmr_params is not None:
            pipeline["mrmr"] = mrmr_params
        joblib.dump(pipeline, output_model)
        print("✅ BLOK 3 selesai. Model PI inverter berhasil disimpan.")

    def train(
        self,
        dataset_file: str,
        output_model: str,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        use_mrmr: bool = False,
        mrmr_features: int | None = None,
        mrmr_pool: int = 2048,
        mrmr_sample: int | None = 2000,
        mrmr_redundancy_weight: float = 1.0,
        mrmr_random_state: int = 42,
        mrmr_score_mode: str = "mifs",
        mrmr_prefilter_stride: int = 1,
    ) -> None:
        print("=== BLOK 3: Hierarchical PI Inversion (Phase 1) ===")
        dataset = self.load_dataset(dataset_file)
        split = self.split_dataset(dataset)
        print(
            f"Skema partisi data: {len(split.train_indices)} LATIH | "
            f"{len(split.test_indices)} UJI."
        )

        selected_indices = None
        if use_mrmr:
            if mrmr_features is None:
                mrmr_features = 1024
            y_targets = np.stack(
                [dataset.temperatures_K, dataset.electron_densities_cm3], axis=1
            )
            prefilter_indices = self.build_prefilter_indices(
                dataset.spectra.shape[1], int(mrmr_prefilter_stride)
            )
            X_train = dataset.spectra[split.train_indices][:, prefilter_indices]
            y_train = y_targets[split.train_indices]
            print(
                f"[mRMR] Prefilter stride={int(mrmr_prefilter_stride)} "
                f"menurunkan kandidat awal dari {dataset.spectra.shape[1]} ke {len(prefilter_indices)} fitur."
            )
            mrmr_cfg = MRMRConfig(
                n_features=int(mrmr_features),
                pool_size=int(mrmr_pool),
                sample_size=int(mrmr_sample) if mrmr_sample is not None else None,
                redundancy_weight=float(mrmr_redundancy_weight),
                random_state=int(mrmr_random_state),
                score_mode=str(mrmr_score_mode),
            )
            selected_local = compute_mrmr_indices(X_train, y_train, mrmr_cfg)
            selected_indices = prefilter_indices[selected_local]
            dataset = DatasetBundle(
                spectra=dataset.spectra[:, selected_indices],
                temperatures_K=dataset.temperatures_K,
                electron_densities_cm3=dataset.electron_densities_cm3,
                parameter_columns=dataset.parameter_columns,
            )
            print(f"[mRMR] Dimensi input turun menjadi {dataset.spectra.shape[1]} fitur.")

        inversion_config = self.build_training_config(
            input_dim=dataset.spectra.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        train_dataset = self.build_phase_dataset(dataset, split.train_indices, inversion_config)
        inverter = HierarchicalPIInverter(inversion_config)
        history = inverter.fit(train_dataset, epochs=inversion_config.epochs)
        print(f"Loss epoch terakhir: {history.losses[-1]:.4e}")

        metrics = self.evaluate_model(inverter, dataset, split)
        mrmr_params = None
        if use_mrmr and selected_indices is not None:
            mrmr_params = {
                "n_features": int(mrmr_features),
                "pool_size": int(mrmr_pool),
                "sample_size": int(mrmr_sample) if mrmr_sample is not None else None,
                "redundancy_weight": float(mrmr_redundancy_weight),
                "random_state": int(mrmr_random_state),
                "prefilter_stride": int(mrmr_prefilter_stride),
            }
        self.save_pipeline(
            output_model,
            inverter,
            inversion_config,
            metrics,
            selected_indices=selected_indices,
            mrmr_params=mrmr_params,
        )


def train_model(
    dataset_file: str,
    output_model: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    use_mrmr: bool = False,
    mrmr_features: int | None = None,
    mrmr_pool: int = 2048,
    mrmr_sample: int | None = 2000,
    mrmr_redundancy_weight: float = 1.0,
    mrmr_random_state: int = 42,
    mrmr_score_mode: str = "mifs",
    mrmr_prefilter_stride: int = 1,
) -> None:
    trainer = PIInversionTrainer(_CONFIG)
    trainer.train(
        dataset_file=dataset_file,
        output_model=output_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_mrmr=use_mrmr,
        mrmr_features=mrmr_features,
        mrmr_pool=mrmr_pool,
        mrmr_sample=mrmr_sample,
        mrmr_redundancy_weight=mrmr_redundancy_weight,
        mrmr_random_state=mrmr_random_state,
        mrmr_score_mode=mrmr_score_mode,
        mrmr_prefilter_stride=mrmr_prefilter_stride,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blok 3: Latih arsitektur Hierarchical PI Inversion Phase 1."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(_BASE_DIR, "data", "dataset_synthetic.h5"),
        help="File HDF5 sintetik masuk",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_BASE_DIR, "data", "model_inversi_pi.pkl"),
        help="File model keluar",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=_BASE_DIR,
        help="Base directory project (override LIBS_BASE_DIR)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--mrmr", action="store_true", help="Aktifkan seleksi fitur mRMR")
    parser.add_argument("--mrmr-features", type=int, default=None, help="Jumlah fitur mRMR")
    parser.add_argument("--mrmr-pool", type=int, default=2048, help="Jumlah kandidat fitur mRMR")
    parser.add_argument("--mrmr-sample", type=int, default=2000, help="Subsample baris untuk MI")
    parser.add_argument(
        "--mrmr-redundancy-weight",
        type=float,
        default=1.0,
        help="Bobot penalti redundansi mRMR",
    )
    parser.add_argument(
        "--mrmr-score-mode",
        type=str,
        default="mifs",
        help="Skor mRMR: 'mifs' (relevance - redundancy) atau 'miq' (relevance / redundancy)",
    )
    parser.add_argument(
        "--mrmr-prefilter-stride",
        type=int,
        default=1,
        help="Ambil setiap k titik sebelum mRMR untuk menurunkan biaya komputasi",
    )
    parser.add_argument("--mrmr-random-state", type=int, default=42, help="Seed mRMR")
    args = parser.parse_args()

    if args.base_dir != _BASE_DIR:
        _BASE_DIR = args.base_dir

    train_model(
        dataset_file=args.dataset,
        output_model=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_mrmr=args.mrmr,
        mrmr_features=args.mrmr_features,
        mrmr_pool=args.mrmr_pool,
        mrmr_sample=args.mrmr_sample,
        mrmr_redundancy_weight=args.mrmr_redundancy_weight,
        mrmr_random_state=args.mrmr_random_state,
        mrmr_score_mode=args.mrmr_score_mode,
        mrmr_prefilter_stride=args.mrmr_prefilter_stride,
    )
