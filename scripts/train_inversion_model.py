"""Compatibility wrapper for the generic experiment training runner."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.runner import run_training_job


_BASE_DIR = os.environ.get(
    "LIBS_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


def train_model(
    dataset_file: str,
    output_model: str,
    report_file: str | None = None,
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
) -> dict[str, float]:
    return run_training_job(
        dataset=dataset_file,
        model_name="pi",
        output_model=output_model,
        report_file=report_file,
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
    parser = argparse.ArgumentParser(description="Blok 3: Latih arsitektur Hierarchical PI Inversion Phase 1.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(_BASE_DIR, "data", "processed", "dataset_synthetic.h5"),
        help="File HDF5 sintetik masuk",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_BASE_DIR, "artifacts", "models", "model_inversi_pi.pkl"),
        help="File model keluar",
    )
    parser.add_argument("--report-out", type=str, default=None, help="File laporan ilmiah .txt; default mengikuti nama model")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--mrmr", action="store_true", help="Aktifkan seleksi fitur mRMR")
    parser.add_argument("--mrmr-features", type=int, default=None, help="Jumlah fitur mRMR")
    parser.add_argument("--mrmr-pool", type=int, default=2048, help="Jumlah kandidat fitur mRMR")
    parser.add_argument("--mrmr-sample", type=int, default=2000, help="Subsample baris untuk MI")
    parser.add_argument("--mrmr-redundancy-weight", type=float, default=1.0, help="Bobot penalti redundansi mRMR")
    parser.add_argument("--mrmr-score-mode", type=str, default="mifs", help="Skor mRMR")
    parser.add_argument("--mrmr-prefilter-stride", type=int, default=1, help="Ambil setiap k titik sebelum mRMR")
    parser.add_argument("--mrmr-random-state", type=int, default=42, help="Seed mRMR")
    args = parser.parse_args()

    train_model(
        dataset_file=args.dataset,
        output_model=args.out,
        report_file=args.report_out,
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
