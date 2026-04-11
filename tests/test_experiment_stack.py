from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.core.config import load_config_tree
from src.core.contracts import ReportContext
from src.core.reporting import write_training_report
from src.data.io import load_dataset_bundle
from src.models.pi import PIModel
from src.models.registry import available_models, create_model


def test_registry_exposes_expected_model_names():
    names = available_models()
    assert "pi" in names
    assert "cnn_transformer" in names
    assert "svr" in names


def test_config_tree_loads_legacy_base():
    config, path = load_config_tree("configs/base.yaml")
    assert path.name == "base.yaml"
    assert "instrument" in config
    assert config["data"]["processed_dir"] == "data/processed"


def test_report_writer_outputs_standardized_fields(tmp_path: Path):
    report_path = tmp_path / "report.txt"
    context = ReportContext(
        dataset_code="A_L",
        dataset_file="data/processed/dataset_synthetic_A_L.h5",
        model_output="artifacts/models/model.pkl",
        model_type="hierarchical_pi_inverter_phase1",
        pipeline="PI",
        feature_mode="plain",
        sample_count=20,
        input_dim=64,
        train_count=16,
        test_count=4,
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
        physics_informed=True,
        target_composition_count=2,
        composition_columns=["comp_Si_pct", "comp_Al_pct"],
        extra_model_lines=[("mRMR", "tidak digunakan")],
    )
    write_training_report(
        str(report_path),
        context,
        {
            "rel_rmse_T_e_core_K_pct": 9.5,
            "rel_rmse_n_e_core_cm3_pct": 25.0,
            "rmse_composition_mean_pct": 2.1,
        },
    )
    text = report_path.read_text(encoding="utf-8")
    assert "Model type" in text
    assert "Pipeline" in text
    assert "Verdict komposisi" in text


def test_dataset_loader_normalizes_compositions(tmp_path: Path):
    dataset_path = tmp_path / "dataset.h5"
    spectra = np.ones((3, 4), dtype=np.float32)
    params = np.array(
        [
            [9000.0, 5000.0, 2e17, 1e16, 60.0, 40.0],
            [10000.0, 5500.0, 2.5e17, 1.1e16, 55.0, 45.0],
            [11000.0, 6000.0, 3e17, 1.2e16, 50.0, 50.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(dataset_path, "w") as handle:
        dset_params = handle.create_dataset("parameters", data=params)
        dset_params.attrs["columns"] = [
            "T_e_core_K",
            "T_e_shell_K",
            "n_e_core_cm3",
            "n_e_shell_cm3",
            "comp_Si_pct",
            "comp_Al_pct",
        ]
        handle.create_dataset("spectra", data=spectra)

    bundle = load_dataset_bundle(str(dataset_path))
    assert bundle.compositions is not None
    assert np.allclose(bundle.compositions.sum(axis=1), np.ones(3))


def test_pi_model_can_be_created_from_registry():
    model = create_model("pi", project_config={})
    assert isinstance(model, PIModel)

