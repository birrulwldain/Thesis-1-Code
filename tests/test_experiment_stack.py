from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.core.config import load_config_tree
from src.core.contracts import ReportContext
from src.core.reporting import write_training_report
from src.data.io import load_dataset_bundle
from src.models.cnn_transformer import CNNTransformerModel
from src.models.pi import PIModel
from src.models.svr import SVRModel
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


def test_svr_model_trains_and_roundtrips(tmp_path: Path):
    dataset_path = tmp_path / "dataset.h5"
    spectra = np.stack(
        [
            np.linspace(0.0, 1.0, 12, dtype=np.float32),
            np.linspace(1.0, 0.0, 12, dtype=np.float32),
            np.full(12, 0.25, dtype=np.float32),
            np.full(12, 0.5, dtype=np.float32),
            np.linspace(0.2, 0.8, 12, dtype=np.float32),
            np.linspace(0.8, 0.2, 12, dtype=np.float32),
        ],
        axis=0,
    )
    params = np.array(
        [
            [9000.0, 5000.0],
            [9800.0, 5300.0],
            [10200.0, 5600.0],
            [10800.0, 5900.0],
            [11400.0, 6200.0],
            [12000.0, 6500.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(dataset_path, "w") as handle:
        dset_params = handle.create_dataset("parameters", data=params)
        dset_params.attrs["columns"] = ["T_e_core_K", "n_e_core_cm3"]
        handle.create_dataset("spectra", data=spectra)

    model = create_model("svr", project_config={}, model_params={"pca_components": 2, "C": 1.0, "epsilon": 0.1})
    assert isinstance(model, SVRModel)

    output_model = tmp_path / "svr.pkl"
    report_file = tmp_path / "svr_report.txt"
    metrics = model.fit(str(dataset_path), str(output_model), str(report_file))

    assert output_model.exists()
    assert report_file.exists()
    assert "rel_rmse_T_e_core_K_pct" in metrics

    loaded = SVRModel.load(str(output_model))
    predictions = loaded.predict(spectra[:2])
    assert predictions["T_e_core_K"].shape == (2,)
    assert predictions["n_e_core_cm3"].shape == (2,)


def test_cnn_transformer_model_trains_and_roundtrips(tmp_path: Path):
    dataset_path = tmp_path / "dataset.h5"
    n_samples = 8
    n_points = 24
    spectra = np.stack(
        [np.roll(np.linspace(0.0, 1.0, n_points, dtype=np.float32), i) for i in range(n_samples)],
        axis=0,
    )
    params = np.array(
        [
            [9000.0 + 400.0 * i, 5.0e3 + 120.0 * i]
            for i in range(n_samples)
        ],
        dtype=np.float32,
    )
    with h5py.File(dataset_path, "w") as handle:
        dset_params = handle.create_dataset("parameters", data=params)
        dset_params.attrs["columns"] = ["T_e_core_K", "n_e_core_cm3"]
        handle.create_dataset("spectra", data=spectra)

    model = create_model(
        "cnn_transformer",
        project_config={},
        model_params={
            "encoder_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "conv_channels": 8,
            "kernel_size": 3,
        },
        training_params={
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
        },
    )
    assert isinstance(model, CNNTransformerModel)

    output_model = tmp_path / "cnn_transformer.pkl"
    report_file = tmp_path / "cnn_transformer_report.txt"
    metrics = model.fit(str(dataset_path), str(output_model), str(report_file))

    assert output_model.exists()
    assert report_file.exists()
    assert "rel_rmse_T_e_core_K_pct" in metrics
    assert hasattr(model.model, "mono_encoder")
    assert hasattr(model.model, "poly_encoder")
    assert hasattr(model.model, "cross_attention")
    assert hasattr(model.model, "decoder")

    loaded = CNNTransformerModel.load(str(output_model))
    predictions = loaded.predict(spectra[:3])
    assert predictions["T_e_K"].shape == (3,)
    assert predictions["n_e_cm3"].shape == (3,)
    assert predictions["T_e_core_K"].shape == (3,)
    assert predictions["n_e_core_cm3"].shape == (3,)
    assert predictions["mono_reconstruction"].shape == (3, n_points)

