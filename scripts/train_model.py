from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.runner import run_training_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runner generik untuk eksperimen training model.")
    parser.add_argument("--dataset", type=str, default=None, help="Path dataset HDF5.")
    parser.add_argument("--model", type=str, default=None, help="Nama model registry: pi, svr, cnn, cnn_transformer.")
    parser.add_argument("--config", type=str, default=None, help="File config eksperimen YAML.")
    parser.add_argument("--out", type=str, default=None, help="Path model output.")
    parser.add_argument("--report-out", type=str, default=None, help="Path report output.")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--mrmr", action="store_true", help="Aktifkan seleksi fitur mRMR.")
    parser.add_argument("--mrmr-features", type=int, default=None, help="Jumlah fitur mRMR.")
    parser.add_argument("--mrmr-pool", type=int, default=None, help="Ukuran pool kandidat mRMR.")
    parser.add_argument("--mrmr-sample", type=int, default=None, help="Subsample baris untuk mutual information.")
    parser.add_argument("--mrmr-redundancy-weight", type=float, default=None, help="Bobot penalti redundansi.")
    parser.add_argument("--mrmr-score-mode", type=str, default=None, help="Mode score mRMR.")
    parser.add_argument("--mrmr-prefilter-stride", type=int, default=None, help="Stride prefilter mRMR.")
    parser.add_argument("--mrmr-random-state", type=int, default=None, help="Seed mRMR.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_training_job(
        dataset=args.dataset,
        model_name=args.model,
        config_path=args.config,
        output_model=args.out,
        report_file=args.report_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_mrmr=args.mrmr if args.mrmr else None,
        mrmr_features=args.mrmr_features,
        mrmr_pool=args.mrmr_pool,
        mrmr_sample=args.mrmr_sample,
        mrmr_redundancy_weight=args.mrmr_redundancy_weight,
        mrmr_random_state=args.mrmr_random_state,
        mrmr_score_mode=args.mrmr_score_mode,
        mrmr_prefilter_stride=args.mrmr_prefilter_stride,
    )
