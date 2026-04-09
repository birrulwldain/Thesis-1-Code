import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.libs_inversion import (
    HierarchicalInversionConfig,
    HierarchicalPIInverter,
    Phase2GeometrySolver,
    SobolevConstraintLoss,
    ThermoPhaseDataset,
    ThermoInversionNet,
)


def make_synthetic_dataset(
    n_samples: int = 12,
    n_points: int = 64,
    use_composition_head: bool = False,
    composition_dim: int = 0,
):
    np.random.seed(7)
    torch.manual_seed(7)
    config = HierarchicalInversionConfig.from_config(
        input_dim=n_points,
        hidden_dims=(64, 32),
        epochs=4,
        batch_size=4,
        use_composition_head=use_composition_head,
        composition_dim=composition_dim,
        learning_rate=2e-3,
        sobolev_weight=0.05,
        equality_weight=2.0,
        inequality_weight=0.1,
    )
    grid = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    temperatures = np.linspace(9000.0, 15000.0, n_samples, dtype=np.float32)
    densities = np.linspace(1.2e17, 4.5e17, n_samples, dtype=np.float32)
    spectra = []
    compositions = []
    for i, (temp, ne) in enumerate(zip(temperatures, densities)):
        center = 0.2 + 0.6 * i / max(1, n_samples - 1)
        width = 0.04 + 0.01 * (i / max(1, n_samples - 1))
        amplitude = 0.7 + 0.3 * (temp - temperatures.min()) / (temperatures.max() - temperatures.min())
        density_term = 0.05 * ((np.log10(ne) - 17.0) / 1.0) * np.sin(6.0 * np.pi * grid)
        spectrum = amplitude * np.exp(-((grid - center) ** 2) / (2.0 * width ** 2)) + density_term
        spectrum = np.clip(spectrum, 0.0, None)
        spectra.append(spectrum.astype(np.float32))
        if use_composition_head:
            frac = np.array([0.6 + 0.02 * np.sin(i), 0.4 - 0.02 * np.sin(i)], dtype=np.float32)
            frac = np.clip(frac, 1e-4, None)
            frac = frac / frac.sum()
            compositions.append(frac)

    spectra = np.asarray(spectra, dtype=np.float32)
    kwargs = {}
    if use_composition_head:
        kwargs["compositions"] = np.asarray(compositions, dtype=np.float32)
    dataset = ThermoPhaseDataset(
        spectra=spectra,
        temperatures_K=temperatures,
        electron_densities_cm3=densities,
        config=config,
        **kwargs,
    )
    return config, dataset, spectra, temperatures, densities


def test_target_normalization_roundtrip():
    config, _, _, temperatures, densities = make_synthetic_dataset()
    t = torch.as_tensor(temperatures, dtype=torch.float32)
    n = torch.as_tensor(densities, dtype=torch.float32)

    t_norm = HierarchicalPIInverter.normalize_temperature_static(t, config.temperature_bounds_K)
    n_norm = HierarchicalPIInverter.normalize_density_static(n, config.electron_density_bounds_cm3)
    t_back = HierarchicalPIInverter.denormalize_temperature_static(t_norm, config.temperature_bounds_K)
    n_back = HierarchicalPIInverter.denormalize_density_static(n_norm, config.electron_density_bounds_cm3)

    assert torch.allclose(t, t_back, rtol=1e-4, atol=1e-3)
    assert torch.allclose(n, n_back, rtol=1e-4, atol=1e10)


def test_composition_projection_stays_on_simplex():
    config, _, spectra, _, _ = make_synthetic_dataset(
        use_composition_head=True,
        composition_dim=2,
    )
    net = ThermoInversionNet(config)
    inputs = torch.as_tensor(spectra[:3], dtype=torch.float32)
    outputs = net(inputs)
    composition = outputs["composition"]

    assert composition.shape == (3, 2)
    assert torch.all(composition >= 0.0)
    assert torch.allclose(composition.sum(dim=1), torch.ones(3), atol=1e-6)


def test_sobolev_loss_produces_expected_jacobian_shape():
    config, dataset, spectra, _, _ = make_synthetic_dataset()
    net = ThermoInversionNet(config)
    loss_fn = SobolevConstraintLoss(config)

    inputs = torch.as_tensor(spectra[:2], dtype=torch.float32).requires_grad_(True)
    batch = {
        "spectrum": torch.stack([dataset[0]["spectrum"], dataset[1]["spectrum"]], dim=0),
        "target": torch.stack([dataset[0]["target"], dataset[1]["target"]], dim=0),
        "jacobian_target": torch.stack([dataset[0]["jacobian_target"], dataset[1]["jacobian_target"]], dim=0),
    }
    predictions = net(inputs)
    inverter = HierarchicalPIInverter(config)
    decoded = inverter.decode_outputs(predictions["thermo"])
    _, metrics = loss_fn(predictions, batch, inputs, decoded)
    jacobian = loss_fn._compute_output_jacobian(predictions["thermo"], inputs)

    assert jacobian.shape == (2, config.output_dim, config.input_dim)
    assert "sobolev_loss" in metrics


def test_training_loss_decreases_on_small_synthetic_dataset():
    config, dataset, _, _, _ = make_synthetic_dataset()
    torch.manual_seed(11)
    inverter = HierarchicalPIInverter(config)
    history = inverter.fit(dataset, epochs=4)

    assert len(history.losses) == 4
    assert min(history.losses[1:]) <= history.losses[0]


def test_predict_thermo_returns_physical_ranges():
    config, dataset, spectra, _, _ = make_synthetic_dataset()
    inverter = HierarchicalPIInverter(config)
    inverter.fit(dataset, epochs=3)
    preds = inverter.predict_thermo(spectra[:3])

    assert preds["T_e_K"].shape == (3,)
    assert preds["n_e_cm3"].shape == (3,)
    assert np.all(preds["T_e_K"] >= config.temperature_bounds_K[0])
    assert np.all(preds["T_e_K"] <= config.temperature_bounds_K[1])
    assert np.all(preds["n_e_cm3"] >= config.electron_density_bounds_cm3[0])
    assert np.all(preds["n_e_cm3"] <= config.electron_density_bounds_cm3[1])


def test_predict_with_composition_head_returns_simplex_outputs():
    config, dataset, spectra, _, _ = make_synthetic_dataset(
        use_composition_head=True,
        composition_dim=2,
    )
    inverter = HierarchicalPIInverter(config)
    inverter.fit(dataset, epochs=2)
    preds = inverter.predict_thermo(spectra[:3])

    assert "composition" in preds
    assert preds["composition"].shape == (3, 2)
    assert np.all(preds["composition"] >= 0.0)
    assert np.allclose(preds["composition"].sum(axis=1), np.ones(3), atol=1e-5)


def test_phase2_geometry_solver_returns_physical_ranges():
    config, dataset, spectra, _, _ = make_synthetic_dataset()
    inverter = HierarchicalPIInverter(config)
    inverter.fit(dataset, epochs=2)
    thermo = inverter.predict_thermo(spectra[:1])

    solver = Phase2GeometrySolver(config)
    result = solver.fit(
        observed_spectrum=spectra[0],
        core_temp_K=float(thermo["T_e_K"][0]),
        core_ne_cm3=float(thermo["n_e_cm3"][0]),
    )

    assert config.core_thickness_bounds_m[0] <= result.core_thickness_m <= config.core_thickness_bounds_m[1]
    assert config.shell_thickness_bounds_m[0] <= result.shell_thickness_m <= config.shell_thickness_bounds_m[1]
    assert result.tau_shell_max >= 0.0
    assert result.simulated_spectrum.shape == spectra[0].shape
