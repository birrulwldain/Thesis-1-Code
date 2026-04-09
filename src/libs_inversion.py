"""
libs_inversion.py — Hierarchical Physics-Informed Inversion Engine
==================================================================
Q1 Thesis: Phase-1 thermodynamic inversion for CR-LIBS spectra.

Design principles:
  - Forward physics stays in libs_physics.py.
  - Inversion is implemented as a PyTorch-based, physics-informed regressor.
  - Phase 1 predicts thermodynamic parameters (T_e, n_e) with optional
    composition branch; Phase 2 interfaces are prepared as stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os

import numpy as np
import yaml

from src.libs_physics import (
    DataFetcher,
    PhysicsCalculator,
    PlasmaZoneParams,
    TwoZonePlasma,
)

try:
    import torch
    from torch import Tensor
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    Tensor = Any
    nn = None

    class Dataset:  # type: ignore[override]
        """Fallback base class when torch is unavailable."""

    class DataLoader:  # type: ignore[override]
        """Fallback placeholder when torch is unavailable."""


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f)


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for src.libs_inversion. Install the 'torch' package "
            "to use the hierarchical PI inverter."
        )


@dataclass
class HierarchicalInversionConfig:
    """Configuration for the physics-informed Phase-1 inverter."""

    input_dim: int
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    output_dim: int = 2
    composition_dim: int = 0
    use_composition_head: bool = False
    shell_temp_factor: float = 0.5
    shell_density_factor: float = 0.1
    core_ion_temp_factor: float = 0.8
    shell_ion_temp_factor: float = 0.8
    core_thickness_bounds_m: Tuple[float, float] = (1e-6, 1e-5)
    shell_thickness_bounds_m: Tuple[float, float] = (1e-6, 1e-5)
    geometry_grid_size: int = 8
    temperature_bounds_K: Tuple[float, float] = tuple(
        map(float, _CONFIG["monte_carlo_synthesizer"]["core"]["T_range_K"])
    )
    electron_density_bounds_cm3: Tuple[float, float] = tuple(
        map(float, _CONFIG["monte_carlo_synthesizer"]["core"]["ne_range_cm3"])
    )
    thin_tau_threshold: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8
    epochs: int = 10
    device: str = "cpu"
    data_loss_weight: float = 1.0
    sobolev_weight: float = 0.1
    equality_weight: float = 5.0
    inequality_weight: float = 1.0
    alm_rho: float = 2.0
    planck_weight: float = 0.0
    core_shell_weight: float = 0.0
    geometry_weight: float = 0.0

    @classmethod
    def from_config(
        cls,
        input_dim: Optional[int] = None,
        use_composition_head: bool = False,
        composition_dim: int = 0,
        **overrides: Any,
    ) -> "HierarchicalInversionConfig":
        inferred_dim = input_dim or int(_CONFIG["instrument"]["resolution"])
        return cls(
            input_dim=inferred_dim,
            use_composition_head=use_composition_head,
            composition_dim=composition_dim,
            **overrides,
        )


@dataclass
class GeometryPhaseDataset:
    """Stub for future Phase-2 geometric inversion."""

    message: str = "Phase 2 geometry dataset is not implemented in v1."


if nn is not None:

    class ThermoPhaseDataset(Dataset):
        """Phase-1 dataset for optically-thin thermodynamic inversion."""

        def __init__(
            self,
            spectra: np.ndarray,
            temperatures_K: np.ndarray,
            electron_densities_cm3: np.ndarray,
            config: HierarchicalInversionConfig,
            compositions: Optional[np.ndarray] = None,
            tau_values: Optional[np.ndarray] = None,
            jacobian_targets: Optional[np.ndarray] = None,
        ) -> None:
            _require_torch()
            spectra = np.asarray(spectra, dtype=np.float32)
            temperatures_K = np.asarray(temperatures_K, dtype=np.float32).reshape(-1)
            electron_densities_cm3 = np.asarray(electron_densities_cm3, dtype=np.float32).reshape(-1)

            if spectra.ndim != 2:
                raise ValueError("spectra must be a 2D array [n_samples, n_wavelengths].")
            if spectra.shape[0] != temperatures_K.shape[0] or spectra.shape[0] != electron_densities_cm3.shape[0]:
                raise ValueError("spectra, temperatures_K, and electron_densities_cm3 must share n_samples.")

            self.config = config
            self.spectra = torch.as_tensor(spectra, dtype=torch.float32)
            self.temperatures_K = torch.as_tensor(temperatures_K, dtype=torch.float32)
            self.electron_densities_cm3 = torch.as_tensor(electron_densities_cm3, dtype=torch.float32)
            self.targets = torch.stack(
                [
                    HierarchicalPIInverter.normalize_temperature_static(
                        self.temperatures_K, config.temperature_bounds_K
                    ),
                    HierarchicalPIInverter.normalize_density_static(
                        self.electron_densities_cm3, config.electron_density_bounds_cm3
                    ),
                ],
                dim=1,
            )

            if compositions is not None:
                compositions = np.asarray(compositions, dtype=np.float32)
                if compositions.shape[0] != spectra.shape[0]:
                    raise ValueError("compositions must share n_samples with spectra.")
                self.compositions = torch.as_tensor(compositions, dtype=torch.float32)
            else:
                self.compositions = None

            if tau_values is None:
                tau_values = np.zeros(spectra.shape[0], dtype=np.float32)
            tau_values = np.asarray(tau_values, dtype=np.float32).reshape(-1)
            if tau_values.shape[0] != spectra.shape[0]:
                raise ValueError("tau_values must share n_samples with spectra.")
            self.tau_values = torch.as_tensor(tau_values, dtype=torch.float32)
            self.thin_mask = self.tau_values <= float(config.thin_tau_threshold)

            if jacobian_targets is None:
                jacobian_targets = self._estimate_rte_jacobian_targets(spectra)
            jacobian_targets = np.asarray(jacobian_targets, dtype=np.float32)
            if jacobian_targets.shape != (spectra.shape[0], config.output_dim, spectra.shape[1]):
                raise ValueError(
                    "jacobian_targets must have shape [n_samples, output_dim, n_wavelengths]."
                )
            self.jacobian_targets = torch.as_tensor(jacobian_targets, dtype=torch.float32)

        def _estimate_rte_jacobian_targets(self, spectra: np.ndarray) -> np.ndarray:
            # Phase-1 proxy Jacobians: intensity envelope encodes temperature-like response,
            # first derivative encodes sensitivity to line-shape/optical-thickness changes.
            centered = spectra - spectra.mean(axis=1, keepdims=True)
            scale = np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
            temp_proxy = centered / scale

            slope_proxy = np.gradient(spectra, axis=1)
            slope_scale = np.maximum(np.linalg.norm(slope_proxy, axis=1, keepdims=True), 1e-8)
            density_proxy = slope_proxy / slope_scale

            return np.stack([temp_proxy, density_proxy], axis=1).astype(np.float32)

        def __len__(self) -> int:
            return int(self.spectra.shape[0])

        def __getitem__(self, idx: int) -> Dict[str, Tensor]:
            item = {
                "spectrum": self.spectra[idx],
                "target": self.targets[idx],
                "temperature_K": self.temperatures_K[idx],
                "electron_density_cm3": self.electron_densities_cm3[idx],
                "thin_mask": self.thin_mask[idx],
                "jacobian_target": self.jacobian_targets[idx],
                "tau": self.tau_values[idx],
            }
            if self.compositions is not None:
                item["composition"] = self.compositions[idx]
            return item


    class ThermoInversionNet(nn.Module):
        """Phase-1 thermodynamic inversion network with optional composition head."""

        def __init__(self, config: HierarchicalInversionConfig) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(hidden))
                in_dim = hidden
            self.backbone = nn.Sequential(*layers)
            self.thermo_head = nn.Linear(in_dim, config.output_dim)
            self.use_composition_head = bool(config.use_composition_head and config.composition_dim > 0)
            self.composition_head = (
                nn.Linear(in_dim, config.composition_dim) if self.use_composition_head else None
            )

        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            features = self.backbone(x)
            thermo = torch.tanh(self.thermo_head(features))
            outputs = {"thermo": thermo}
            if self.composition_head is not None:
                outputs["composition_logits"] = self.composition_head(features)
                outputs["composition"] = torch.softmax(outputs["composition_logits"], dim=1)
            return outputs


    class GeometryInversionHead(nn.Module):
        """Stub head for future Phase-2 geometry inversion."""

        def __init__(self, input_dim: int, output_dim: int = 3) -> None:
            super().__init__()
            self.proj = nn.Linear(input_dim, output_dim)

        def forward(self, x: Tensor) -> Tensor:
            raise NotImplementedError("Geometry inversion is planned but not implemented in v1.")


    class SobolevConstraintLoss(nn.Module):
        """Combined data-fit, Sobolev, and ALM-style constraint loss."""

        def __init__(self, config: HierarchicalInversionConfig) -> None:
            super().__init__()
            self.config = config

        def _compute_output_jacobian(self, outputs: Tensor, inputs: Tensor) -> Tensor:
            jacobian_rows = []
            for out_idx in range(outputs.shape[1]):
                grad_outputs = torch.ones_like(outputs[:, out_idx])
                grads = torch.autograd.grad(
                    outputs[:, out_idx],
                    inputs,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                jacobian_rows.append(grads)
            return torch.stack(jacobian_rows, dim=1)

        def forward(
            self,
            predictions: Dict[str, Tensor],
            batch: Dict[str, Tensor],
            inputs: Tensor,
            decoded: Dict[str, Tensor],
        ) -> Tuple[Tensor, Dict[str, float]]:
            thermo_pred = predictions["thermo"]
            target = batch["target"]
            thin_mask = batch.get("thin_mask")
            if thin_mask is not None:
                thin_weights = thin_mask.float().view(-1, 1)
                thin_denom = torch.clamp(thin_weights.sum(), min=1.0)
                data_loss = ((thermo_pred - target) ** 2 * thin_weights).sum() / (
                    thin_denom * thermo_pred.shape[1]
                )
            else:
                data_loss = torch.mean((thermo_pred - target) ** 2)

            jacobian_model = self._compute_output_jacobian(thermo_pred, inputs)
            jacobian_target = batch["jacobian_target"]
            if thin_mask is not None:
                jac_weights = thin_mask.float().view(-1, 1, 1)
                jac_denom = torch.clamp(jac_weights.sum(), min=1.0)
                sobolev_loss = ((jacobian_model - jacobian_target) ** 2 * jac_weights).sum() / (
                    jac_denom * jacobian_model.shape[1] * jacobian_model.shape[2]
                )
            else:
                sobolev_loss = torch.mean((jacobian_model - jacobian_target) ** 2)

            penalties = []
            metrics: Dict[str, float] = {
                "data_loss": float(data_loss.detach().cpu()),
                "sobolev_loss": float(sobolev_loss.detach().cpu()),
            }

            if "composition" in predictions and "composition" in batch:
                composition = predictions["composition"]
                sum_violation = composition.sum(dim=1) - 1.0
                eq_penalty = sum_violation.pow(2).mean()
                nonneg_penalty = torch.relu(-composition).pow(2).mean()
                composition_fit = torch.mean((composition - batch["composition"]) ** 2)
                penalties.append(self.config.equality_weight * eq_penalty)
                penalties.append(self.config.inequality_weight * nonneg_penalty)
                penalties.append(composition_fit)
                metrics["composition_eq_penalty"] = float(eq_penalty.detach().cpu())
                metrics["composition_fit"] = float(composition_fit.detach().cpu())

            temp = decoded["temperature_K"]
            ne = decoded["electron_density_cm3"]
            t_min, t_max = self.config.temperature_bounds_K
            ne_min, ne_max = self.config.electron_density_bounds_cm3
            temperature_bounds_penalty = (
                torch.relu(t_min - temp).pow(2) + torch.relu(temp - t_max).pow(2)
            ).mean()
            density_bounds_penalty = (
                torch.relu(ne_min - ne).pow(2) + torch.relu(ne - ne_max).pow(2)
            ).mean()
            inequality_penalty = temperature_bounds_penalty + density_bounds_penalty
            penalties.append(self.config.inequality_weight * inequality_penalty)
            metrics["inequality_penalty"] = float(inequality_penalty.detach().cpu())

            alm_penalty = self.config.alm_rho * sum(penalties) if penalties else torch.zeros_like(data_loss)
            total = (
                self.config.data_loss_weight * data_loss
                + self.config.sobolev_weight * sobolev_loss
                + alm_penalty
            )
            metrics["total_loss"] = float(total.detach().cpu())
            return total, metrics
else:

    class ThermoPhaseDataset(Dataset):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class ThermoInversionNet:  # pragma: no cover - fallback only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class GeometryInversionHead:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class SobolevConstraintLoss:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


@dataclass
class TrainingHistory:
    losses: List[float] = field(default_factory=list)
    metrics: List[Dict[str, float]] = field(default_factory=list)


class ForwardModelWrapper:
    """Wrap TwoZonePlasma as a synthetic spectrum generator for inversion."""

    def __init__(self, elements: list, fetcher=None, fwhm_nm: float = 0.5):
        self.elements = elements
        self.fetcher = fetcher or DataFetcher()
        self.fwhm_nm = fwhm_nm

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        T_core, T_shell, ne_core, ne_shell = theta
        d_core_m = 1e-6
        d_shell_m = 1e-6

        core = PlasmaZoneParams(
            T_e_K=T_core,
            T_i_K=max(T_core * 0.8, 3000.0),
            n_e_cm3=ne_core,
            thickness_m=d_core_m,
            label="Core",
        )
        shell = PlasmaZoneParams(
            T_e_K=T_shell,
            T_i_K=max(T_shell * 0.8, 3000.0),
            n_e_cm3=ne_shell,
            thickness_m=d_shell_m,
            label="Shell",
        )

        model = TwoZonePlasma(core, shell, self.elements, self.fetcher)
        wl, I_raw, _ = model.run()
        if self.fwhm_nm > 0.0:
            I_sim = PhysicsCalculator.instrumental_broadening(
                I_raw, wl, fwhm_instrument_nm=self.fwhm_nm
            )
        else:
            I_sim = I_raw
        m = float(np.max(I_sim))
        if m > 0.0:
            I_sim = I_sim / m
        return I_sim.astype(np.float32)


class HierarchicalPIInverter:
    """High-level API for Phase-1 thermodynamic inversion and Phase-2 stubs."""

    def __init__(self, config: HierarchicalInversionConfig) -> None:
        _require_torch()
        self.config = config
        self.device = torch.device(config.device)
        self.model = ThermoInversionNet(config).to(self.device)
        self.loss_fn = SobolevConstraintLoss(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.history = TrainingHistory()
        self.selected_indices: Optional[np.ndarray] = None

    @staticmethod
    def normalize_temperature_static(values: Tensor, bounds: Tuple[float, float]) -> Tensor:
        vmin, vmax = bounds
        return 2.0 * ((values.log10() - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))) - 1.0

    @staticmethod
    def denormalize_temperature_static(values: Tensor, bounds: Tuple[float, float]) -> Tensor:
        vmin, vmax = bounds
        scaled = (values + 1.0) * 0.5
        log_val = scaled * (np.log10(vmax) - np.log10(vmin)) + np.log10(vmin)
        return torch.pow(torch.tensor(10.0, device=values.device, dtype=values.dtype), log_val)

    @staticmethod
    def normalize_density_static(values: Tensor, bounds: Tuple[float, float]) -> Tensor:
        vmin, vmax = bounds
        return 2.0 * ((values.log10() - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))) - 1.0

    @staticmethod
    def denormalize_density_static(values: Tensor, bounds: Tuple[float, float]) -> Tensor:
        vmin, vmax = bounds
        scaled = (values + 1.0) * 0.5
        log_val = scaled * (np.log10(vmax) - np.log10(vmin)) + np.log10(vmin)
        return torch.pow(torch.tensor(10.0, device=values.device, dtype=values.dtype), log_val)

    def decode_outputs(self, thermo: Tensor) -> Dict[str, Tensor]:
        return {
            "temperature_K": self.denormalize_temperature_static(
                thermo[:, 0], self.config.temperature_bounds_K
            ),
            "electron_density_cm3": self.denormalize_density_static(
                thermo[:, 1], self.config.electron_density_bounds_cm3
            ),
        }

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_losses: List[float] = []
        last_metrics: Dict[str, float] = {}
        for batch in loader:
            inputs = batch["spectrum"].to(self.device)
            inputs = inputs.clone().detach().requires_grad_(True)
            batch_on_device = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            decoded = self.decode_outputs(predictions["thermo"])
            loss, metrics = self.loss_fn(predictions, batch_on_device, inputs, decoded)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            last_metrics = metrics

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        last_metrics = dict(last_metrics)
        last_metrics["epoch_loss"] = epoch_loss
        self.history.losses.append(epoch_loss)
        self.history.metrics.append(last_metrics)
        return last_metrics

    def fit(self, dataset: ThermoPhaseDataset, epochs: Optional[int] = None) -> TrainingHistory:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        n_epochs = int(epochs or self.config.epochs)
        for epoch_idx in range(n_epochs):
            self.train_epoch(loader)
            print(f"[Train] Epoch {epoch_idx + 1}/{n_epochs} selesai.")
        return self.history

    def predict_thermo(self, spectra: np.ndarray) -> Dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            arr = np.asarray(spectra, dtype=np.float32)
            if self.selected_indices is not None:
                if arr.shape[-1] < np.max(self.selected_indices) + 1:
                    raise ValueError("Input spectrum does not match selected mRMR indices.")
                arr = arr[..., self.selected_indices]
            tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            predictions = self.model(tensor)
            decoded = self.decode_outputs(predictions["thermo"])
            result = {
                "T_e_K": decoded["temperature_K"].detach().cpu().numpy(),
                "n_e_cm3": decoded["electron_density_cm3"].detach().cpu().numpy(),
            }
            if "composition" in predictions:
                result["composition"] = predictions["composition"].detach().cpu().numpy()
            return result

    def predict_geometry(self, spectra: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Phase 2 geometry inversion is planned but not implemented in v1.")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        device: Optional[str] = None,
    ) -> "HierarchicalPIInverter":
        config_kwargs = dict(checkpoint["config"])
        if device is not None:
            config_kwargs["device"] = device
        inverter = cls(HierarchicalInversionConfig(**config_kwargs))
        inverter.model.load_state_dict(checkpoint["state_dict"])
        inverter.model.eval()
        selected = checkpoint.get("selected_indices")
        if selected is not None:
            inverter.selected_indices = np.asarray(selected, dtype=int)
        return inverter


@dataclass
class GeometryFitResult:
    core_thickness_m: float
    shell_thickness_m: float
    tau_shell_max: float
    mse: float
    simulated_spectrum: np.ndarray
    metadata: Dict[str, Any]


class Phase2GeometrySolver:
    """Phase-2 geometry solver using forward-model search with fixed thermodynamics."""

    def __init__(
        self,
        config: HierarchicalInversionConfig,
        elements: Optional[Sequence[Tuple[str, float]]] = None,
        fetcher: Optional[DataFetcher] = None,
    ) -> None:
        self.config = config
        self.fetcher = fetcher or DataFetcher()
        default_elements = list(
            zip(_CONFIG["plasma_target"]["elements"], _CONFIG["plasma_target"]["fractions"])
        )
        self.base_elements = list(elements) if elements is not None else default_elements

    def _expanded_elements(self, core_temp_K: float, core_ne_cm3: float) -> List[Tuple[str, int, float]]:
        expanded = []
        for elem, frac in self.base_elements:
            f_neu, f_ion = PhysicsCalculator.compute_saha_ionization_fractions(
                elem, float(frac), core_temp_K, core_ne_cm3, self.fetcher
            )
            if f_neu > 1e-4:
                expanded.append((elem, 1, f_neu))
            if f_ion > 1e-4:
                expanded.append((elem, 2, f_ion))
        total = sum(frac for _, _, frac in expanded)
        if total <= 0.0:
            raise ValueError("No valid species generated for Phase 2 geometry fitting.")
        return [(sym, sp, frac / total) for sym, sp, frac in expanded]

    def fit(
        self,
        observed_spectrum: np.ndarray,
        core_temp_K: float,
        core_ne_cm3: float,
    ) -> GeometryFitResult:
        observed = np.asarray(observed_spectrum, dtype=np.float64)
        expanded = self._expanded_elements(core_temp_K, core_ne_cm3)

        core_temp_i = max(core_temp_K * self.config.core_ion_temp_factor, 3000.0)
        shell_temp_K = max(core_temp_K * self.config.shell_temp_factor, 3000.0)
        shell_temp_i = max(shell_temp_K * self.config.shell_ion_temp_factor, 3000.0)
        shell_ne_cm3 = max(core_ne_cm3 * self.config.shell_density_factor, 1e12)

        d_core_grid = np.linspace(
            self.config.core_thickness_bounds_m[0],
            self.config.core_thickness_bounds_m[1],
            self.config.geometry_grid_size,
        )
        d_shell_grid = np.linspace(
            self.config.shell_thickness_bounds_m[0],
            self.config.shell_thickness_bounds_m[1],
            self.config.geometry_grid_size,
        )

        best_result: Optional[GeometryFitResult] = None
        for d_core_m in d_core_grid:
            for d_shell_m in d_shell_grid:
                core = PlasmaZoneParams(
                    T_e_K=core_temp_K,
                    T_i_K=core_temp_i,
                    n_e_cm3=core_ne_cm3,
                    thickness_m=float(d_core_m),
                    label="Core",
                )
                shell = PlasmaZoneParams(
                    T_e_K=shell_temp_K,
                    T_i_K=shell_temp_i,
                    n_e_cm3=shell_ne_cm3,
                    thickness_m=float(d_shell_m),
                    label="Shell",
                )
                model = TwoZonePlasma(core, shell, expanded, fetcher=self.fetcher, use_rte=True)
                _, sim_spec, metadata = model.run()
                sim_spec = np.asarray(sim_spec, dtype=np.float64)
                if sim_spec.shape[0] != observed.shape[0]:
                    sim_axis = np.linspace(0.0, 1.0, sim_spec.shape[0], dtype=np.float64)
                    obs_axis = np.linspace(0.0, 1.0, observed.shape[0], dtype=np.float64)
                    sim_spec = np.interp(obs_axis, sim_axis, sim_spec)
                sim_max = float(np.max(sim_spec))
                if sim_max > 0.0:
                    sim_spec = sim_spec / sim_max
                mse = float(np.mean((observed - sim_spec) ** 2))
                if best_result is None or mse < best_result.mse:
                    best_result = GeometryFitResult(
                        core_thickness_m=float(d_core_m),
                        shell_thickness_m=float(d_shell_m),
                        tau_shell_max=float(metadata.get("tau_shell_max", 0.0)),
                        mse=mse,
                        simulated_spectrum=sim_spec.astype(np.float32),
                        metadata=metadata,
                    )

        if best_result is None:
            raise RuntimeError("Phase 2 geometry fitting failed to produce a candidate.")
        return best_result


def run_inversion_demo() -> None:
    _require_torch()
    print("=== Hierarchical PI Inversion Demo (Phase 1) ===")
    n_points = 256
    config = HierarchicalInversionConfig.from_config(input_dim=n_points, epochs=5, batch_size=4)
    grid = np.linspace(0.0, 1.0, n_points, dtype=np.float32)

    temperatures = np.linspace(9000.0, 15000.0, 12, dtype=np.float32)
    densities = np.linspace(1e17, 4e17, 12, dtype=np.float32)
    spectra = []
    for i, (temp, ne) in enumerate(zip(temperatures, densities)):
        center = 0.25 + 0.5 * i / max(1, len(temperatures) - 1)
        width = 0.03 + 0.01 * i
        spectrum = np.exp(-((grid - center) ** 2) / (2.0 * width ** 2))
        spectrum += 0.15 * ((np.log10(ne) - 17.0) / 1.0) * np.sin(8.0 * np.pi * grid)
        spectrum = np.clip(spectrum, 0.0, None)
        spectra.append(spectrum.astype(np.float32))
    spectra = np.asarray(spectra, dtype=np.float32)

    dataset = ThermoPhaseDataset(
        spectra=spectra,
        temperatures_K=temperatures,
        electron_densities_cm3=densities,
        config=config,
    )
    inverter = HierarchicalPIInverter(config)
    history = inverter.fit(dataset, epochs=5)
    prediction = inverter.predict_thermo(spectra[:2])
    print(f"Final training loss: {history.losses[-1]:.4e}")
    print(f"Predicted T_e (first 2): {prediction['T_e_K']}")
    print(f"Predicted n_e (first 2): {prediction['n_e_cm3']}")


if __name__ == "__main__":
    run_inversion_demo()
