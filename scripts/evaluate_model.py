from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _base_dir() -> str:
    return _ROOT


if __name__ == "__main__":
    base_dir = _base_dir()
    os.environ["LIBS_BASE_DIR"] = base_dir
    from empirical_validation import run_empirical_validation

    parser = argparse.ArgumentParser(description="Runner tipis untuk evaluasi model tersimpan.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(base_dir, "artifacts", "models", "model_inversi_pi.pkl"),
        help="Path model tersimpan.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(base_dir, "data", "data_eksperimen_mock.csv"),
        help="Path CSV/ASC eksperimen.",
    )
    args = parser.parse_args()
    run_empirical_validation(args.model, args.csv)
