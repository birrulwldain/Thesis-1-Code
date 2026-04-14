from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.core.config import ensure_parent_dir
from src.core.contracts import ReportContext
from src.core.metrics import evaluate_thermo_predictions, print_metrics_summary
from src.core.reporting import infer_dataset_code, write_training_report
from src.core.serialization import load_pipeline_artifact, save_pipeline_artifact
from src.data.io import build_dataset_split, load_dataset_bundle

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("PyTorch is required for CNNTransformerModel.")


class _CNNTransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        conv_channels: int,
        kernel_size: int,
        dropout: float,
        composition_dim: int = 0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.mono_encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(conv_channels, encoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.poly_encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(conv_channels, encoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.mono_positional = nn.Parameter(torch.zeros(1, input_dim, encoder_dim))
        self.poly_positional = nn.Parameter(torch.zeros(1, input_dim, encoder_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=max(encoder_dim * 2, 64),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.mono_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.poly_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=max(encoder_dim * 2, 64),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.decoder_norm = nn.LayerNorm(encoder_dim)

        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.thermo_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 2),
        )
        self.composition_head = (
            nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim, composition_dim),
                nn.Softmax(dim=-1),
            )
            if composition_dim > 0
            else None
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, input_dim),
        )

    def _encode_branch(
        self,
        inputs: torch.Tensor,
        stem: nn.Module,
        positional: torch.Tensor,
        transformer: nn.Module,
    ) -> torch.Tensor:
        tokens = stem(inputs.unsqueeze(1)).transpose(1, 2)
        tokens = tokens + positional[:, : tokens.shape[1], :]
        return transformer(tokens)

    def forward(self, mono_inputs: torch.Tensor, poly_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        mono_encoded = self._encode_branch(
            mono_inputs,
            self.mono_encoder,
            self.mono_positional,
            self.mono_transformer,
        )
        poly_encoded = self._encode_branch(
            poly_inputs,
            self.poly_encoder,
            self.poly_positional,
            self.poly_transformer,
        )
        cross_context, attn_weights = self.cross_attention(
            query=poly_encoded,
            key=mono_encoded,
            value=mono_encoded,
            need_weights=True,
            average_attn_weights=False,
        )
        decoded = self.decoder(tgt=poly_encoded + cross_context, memory=mono_encoded)
        decoded = self.decoder_norm(decoded)

        fused = self.fusion(torch.cat([mono_encoded.mean(dim=1), decoded.mean(dim=1)], dim=-1))
        outputs = {
            "thermo": self.thermo_head(fused),
            "mono_reconstruction": self.reconstruction_head(fused),
            "cross_attention_weights": attn_weights,
        }
        if self.composition_head is not None:
            outputs["composition"] = self.composition_head(fused)
        return outputs


class CNNTransformerModel:
    model_name = "cnn_transformer"

    def __init__(self, project_config: dict[str, Any], **kwargs) -> None:
        self.project_config = project_config
        self.model_params = dict(kwargs.get("model_params") or {})
        self.training_params = dict(kwargs.get("training_params") or {})
        self.preprocessing_params = dict(kwargs.get("preprocessing_params") or {})

        self.device = "cpu"
        self.input_dim: int | None = None
        self.composition_dim: int = 0
        self.input_scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.model: _CNNTransformerBackbone | None = None
        self.dataset = None
        self.metrics: dict[str, float] | None = None
        self.prediction_keys = ["T_e_K", "n_e_cm3"]
        self.legacy_keys = ["T_e_core_K", "n_e_core_cm3"]
        self.reconstruction_key = "mono_reconstruction"

    def _coerce_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_hparams(self, input_dim: int) -> dict[str, Any]:
        merged = dict(self.model_params)
        merged.update(self.training_params)
        merged.update(self.preprocessing_params)

        encoder_dim = self._coerce_int(merged.get("encoder_dim", 128), 128)
        num_heads = self._coerce_int(merged.get("num_heads", 4), 4)
        num_encoder_layers = self._coerce_int(merged.get("num_encoder_layers", merged.get("num_layers", 2)), 2)
        num_decoder_layers = self._coerce_int(merged.get("num_decoder_layers", merged.get("num_layers", 2)), 2)
        conv_channels = self._coerce_int(merged.get("conv_channels", max(32, encoder_dim // 2)), max(32, encoder_dim // 2))
        kernel_size = self._coerce_int(merged.get("kernel_size", 7), 7)
        dropout = self._coerce_float(merged.get("dropout", 0.1), 0.1)
        poly_smoothing_kernel = self._coerce_int(merged.get("poly_smoothing_kernel", 5), 5)

        if encoder_dim <= 0:
            encoder_dim = 128
        encoder_dim = min(encoder_dim, max(num_heads, input_dim))
        if num_heads <= 0:
            num_heads = 1
        if encoder_dim % num_heads != 0:
            encoder_dim = max(num_heads, encoder_dim - (encoder_dim % num_heads))
            if encoder_dim % num_heads != 0:
                encoder_dim = num_heads
        kernel_size = max(3, kernel_size | 1)
        poly_smoothing_kernel = max(3, poly_smoothing_kernel | 1)
        return {
            "encoder_dim": encoder_dim,
            "num_heads": num_heads,
            "num_encoder_layers": max(1, num_encoder_layers),
            "num_decoder_layers": max(1, num_decoder_layers),
            "num_layers": max(1, max(num_encoder_layers, num_decoder_layers)),
            "conv_channels": max(8, conv_channels),
            "kernel_size": kernel_size,
            "poly_smoothing_kernel": poly_smoothing_kernel,
            "dropout": min(max(dropout, 0.0), 0.5),
            "epochs": self._coerce_int(merged.get("epochs", 12), 12),
            "batch_size": self._coerce_int(merged.get("batch_size", 8), 8),
            "learning_rate": self._coerce_float(merged.get("learning_rate", 1e-3), 1e-3),
            "weight_decay": self._coerce_float(merged.get("weight_decay", 1e-5), 1e-5),
            "reconstruction_weight": self._coerce_float(merged.get("reconstruction_weight", 0.2), 0.2),
            "composition_weight": self._coerce_float(merged.get("composition_weight", 0.5), 0.5),
        }

    def _build_model(self, input_dim: int, hparams: dict[str, Any], composition_dim: int) -> _CNNTransformerBackbone:
        _require_torch()
        return _CNNTransformerBackbone(
            input_dim=input_dim,
            encoder_dim=hparams["encoder_dim"],
            num_heads=hparams["num_heads"],
            num_encoder_layers=hparams["num_encoder_layers"],
            num_decoder_layers=hparams["num_decoder_layers"],
            conv_channels=hparams["conv_channels"],
            kernel_size=hparams["kernel_size"],
            dropout=hparams["dropout"],
            composition_dim=composition_dim,
        )

    def _poly_view(self, spectra: np.ndarray, kernel_size: int) -> np.ndarray:
        _require_torch()
        kernel_size = max(3, kernel_size | 1)
        tensor = torch.as_tensor(np.asarray(spectra, dtype=np.float32), dtype=torch.float32).unsqueeze(1)
        smoothed = F.avg_pool1d(tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return smoothed.squeeze(1).cpu().numpy().astype(np.float32)

    def _prepare_views(
        self,
        spectra: np.ndarray | tuple[np.ndarray, np.ndarray] | list[np.ndarray],
        *,
        poly_kernel: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(spectra, (tuple, list)) and len(spectra) == 2:
            mono_view = np.asarray(spectra[0], dtype=np.float32)
            poly_view = np.asarray(spectra[1], dtype=np.float32)
        else:
            mono_view = np.asarray(spectra, dtype=np.float32)
            poly_view = self._poly_view(mono_view, poly_kernel)
        if mono_view.ndim != 2 or poly_view.ndim != 2:
            raise ValueError("mono/poly spectra harus berupa matriks 2D [n_samples, n_points].")
        if mono_view.shape != poly_view.shape:
            raise ValueError("mono/poly spectra harus memiliki shape yang sama.")
        return mono_view, poly_view

    def _predict_array(self, spectra: np.ndarray | tuple[np.ndarray, np.ndarray] | list[np.ndarray]) -> dict[str, np.ndarray]:
        _require_torch()
        if self.model is None or self.input_scaler is None or self.target_scaler is None:
            raise RuntimeError("Model belum dimuat atau dilatih.")
        self.model.eval()
        hparams = self._resolve_hparams(self.input_dim or np.asarray(spectra if not isinstance(spectra, (tuple, list)) else spectra[0]).shape[1])
        mono_view, poly_view = self._prepare_views(spectra, poly_kernel=hparams["poly_smoothing_kernel"])
        mono_features = self.input_scaler.transform(mono_view)
        poly_features = self.input_scaler.transform(poly_view)
        mono_tensor = torch.as_tensor(mono_features, dtype=torch.float32, device=self.device)
        poly_tensor = torch.as_tensor(poly_features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.model(mono_tensor, poly_tensor)
        thermo = self.target_scaler.inverse_transform(outputs["thermo"].detach().cpu().numpy())
        reconstruction = self.input_scaler.inverse_transform(outputs[self.reconstruction_key].detach().cpu().numpy())
        predictions = {
            self.prediction_keys[0]: thermo[:, 0],
            self.prediction_keys[1]: thermo[:, 1],
            self.legacy_keys[0]: thermo[:, 0],
            self.legacy_keys[1]: thermo[:, 1],
            self.reconstruction_key: reconstruction,
        }
        if "composition" in outputs:
            predictions["composition"] = outputs["composition"].detach().cpu().numpy()
        return predictions

    def fit(self, *args, **kwargs):
        _require_torch()
        if not args:
            raise TypeError("fit() membutuhkan minimal path dataset.")

        dataset_file = str(args[0])
        output_model = str(args[1]) if len(args) > 1 else str(Path(dataset_file).with_suffix(".pkl"))
        report_file = str(args[2]) if len(args) > 2 and args[2] is not None else None

        dataset = load_dataset_bundle(dataset_file)
        split = build_dataset_split(dataset)
        self.dataset = dataset
        self.input_dim = int(dataset.spectra.shape[1])
        self.composition_dim = int(dataset.compositions.shape[1]) if dataset.compositions is not None else 0

        hparams = self._resolve_hparams(dataset.spectra.shape[1])
        for key in (
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "encoder_dim",
            "num_heads",
            "num_encoder_layers",
            "num_decoder_layers",
            "num_layers",
            "conv_channels",
            "kernel_size",
            "poly_smoothing_kernel",
            "dropout",
            "reconstruction_weight",
            "composition_weight",
        ):
            if key in kwargs and kwargs[key] is not None:
                hparams[key] = kwargs[key]
        if "num_layers" in kwargs and kwargs["num_layers"] is not None:
            hparams["num_encoder_layers"] = int(kwargs["num_layers"])
            hparams["num_decoder_layers"] = int(kwargs["num_layers"])
        device_override = self.model_params.get("device") or self.training_params.get("device")
        self.device = str(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        mono_view = dataset.spectra
        poly_view = self._poly_view(dataset.spectra, hparams["poly_smoothing_kernel"])

        train_mono = self.input_scaler.fit_transform(mono_view[split.train_indices])
        train_poly = self.input_scaler.transform(poly_view[split.train_indices])
        test_mono = self.input_scaler.transform(mono_view[split.test_indices])
        test_poly = self.input_scaler.transform(poly_view[split.test_indices])
        train_targets = np.column_stack(
            [dataset.temperatures_K[split.train_indices], dataset.electron_densities_cm3[split.train_indices]]
        )
        test_targets = np.column_stack(
            [dataset.temperatures_K[split.test_indices], dataset.electron_densities_cm3[split.test_indices]]
        )
        scaled_train_targets = self.target_scaler.fit_transform(train_targets)
        scaled_mono_reconstruction = train_mono.astype(np.float32)
        train_composition = dataset.compositions[split.train_indices].astype(np.float32) if dataset.compositions is not None else None

        self.model = self._build_model(dataset.spectra.shape[1], hparams, self.composition_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
        loss_fn = nn.MSELoss()

        train_mono_tensor = torch.as_tensor(train_mono, dtype=torch.float32)
        train_poly_tensor = torch.as_tensor(train_poly, dtype=torch.float32)
        target_tensor = torch.as_tensor(scaled_train_targets, dtype=torch.float32)
        recon_tensor = torch.as_tensor(scaled_mono_reconstruction, dtype=torch.float32)
        dataset_tensors: list[torch.Tensor] = [train_mono_tensor, train_poly_tensor, target_tensor, recon_tensor]
        if train_composition is not None:
            dataset_tensors.append(torch.as_tensor(train_composition, dtype=torch.float32))
        batch_size = max(1, min(hparams["batch_size"], len(train_mono_tensor)))
        loader = DataLoader(TensorDataset(*dataset_tensors), batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(hparams["epochs"]):
            for batch in loader:
                batch_mono = batch[0].to(self.device)
                batch_poly = batch[1].to(self.device)
                batch_target = batch[2].to(self.device)
                batch_recon = batch[3].to(self.device)
                batch_comp = batch[4].to(self.device) if len(batch) > 4 else None
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_mono, batch_poly)
                thermo_loss = loss_fn(pred["thermo"], batch_target)
                recon_loss = loss_fn(pred[self.reconstruction_key], batch_recon)
                loss = thermo_loss + float(hparams["reconstruction_weight"]) * recon_loss
                if batch_comp is not None and "composition" in pred:
                    composition_loss = loss_fn(pred["composition"], batch_comp)
                    loss = loss + float(hparams["composition_weight"]) * composition_loss
                loss.backward()
                optimizer.step()

        scaled_predictions = self.model(
            torch.as_tensor(test_mono, dtype=torch.float32, device=self.device),
            torch.as_tensor(test_poly, dtype=torch.float32, device=self.device),
        )
        prediction_array = self.target_scaler.inverse_transform(scaled_predictions["thermo"].detach().cpu().numpy())
        predictions = {
            self.prediction_keys[0]: prediction_array[:, 0],
            self.prediction_keys[1]: prediction_array[:, 1],
            self.legacy_keys[0]: prediction_array[:, 0],
            self.legacy_keys[1]: prediction_array[:, 1],
            self.reconstruction_key: self.input_scaler.inverse_transform(
                scaled_predictions[self.reconstruction_key].detach().cpu().numpy()
            ),
        }
        if "composition" in scaled_predictions:
            predictions["composition"] = scaled_predictions["composition"].detach().cpu().numpy()
        metrics = evaluate_thermo_predictions(predictions, dataset, split)
        print_metrics_summary(metrics, has_holdout=split.has_holdout)

        self.metrics = metrics
        self.save(output_model)
        if report_file is None:
            report_file = os.path.splitext(output_model)[0] + "_report.txt"
        self._write_report(report_file, dataset_file, output_model, split, hparams)
        return metrics

    def predict(self, spectra):
        return self._predict_array(spectra)

    def save(self, output_model: str) -> None:
        _require_torch()
        if self.model is None or self.input_scaler is None or self.target_scaler is None:
            raise RuntimeError("Tidak ada model terlatih untuk disimpan.")
        ensure_parent_dir(output_model)
        payload = {
            "model_type": "cnn_transformer_v1",
            "model_family": self.model_name,
            "config": {
                "model_params": self.model_params,
                "training_params": self.training_params,
                "preprocessing_params": self.preprocessing_params,
            },
            "state_dict": self.model.state_dict(),
            "input_scaler": self.input_scaler,
            "target_scaler": self.target_scaler,
            "device": self.device,
            "metrics": self.metrics or {},
            "input_dim": int(self.dataset.spectra.shape[1]) if self.dataset is not None else None,
            "composition_dim": self.composition_dim,
        }
        save_pipeline_artifact(output_model, payload)

    def _write_report(
        self,
        report_file: str,
        dataset_file: str,
        output_model: str,
        split,
        hparams: dict[str, Any],
    ) -> None:
        if self.dataset is None or self.metrics is None:
            raise RuntimeError("Report tidak bisa ditulis sebelum training selesai.")
        context = ReportContext(
            dataset_code=infer_dataset_code(dataset_file),
            dataset_file=dataset_file,
            model_output=output_model,
            model_type="cnn_transformer_v1",
            pipeline="Mono CNN Encoder + Cross Attention + Poly CNN Decoder",
            feature_mode="dual_view",
            sample_count=self.dataset.spectra.shape[0],
            input_dim=self.dataset.spectra.shape[1],
            train_count=len(split.train_indices),
            test_count=len(split.test_indices),
            epochs=hparams["epochs"],
            batch_size=hparams["batch_size"],
            learning_rate=hparams["learning_rate"],
            physics_informed=False,
            extra_model_lines=[
                ("encoder_dim", hparams["encoder_dim"]),
                ("num_heads", hparams["num_heads"]),
                ("encoder layers", hparams["num_encoder_layers"]),
                ("decoder layers", hparams["num_decoder_layers"]),
                ("conv_channels", hparams["conv_channels"]),
                ("kernel_size", hparams["kernel_size"]),
                ("poly smoothing kernel", hparams["poly_smoothing_kernel"]),
                ("reconstruction weight", hparams["reconstruction_weight"]),
                ("composition weight", hparams["composition_weight"]),
                ("device", self.device),
            ],
        )
        write_training_report(report_file, context, self.metrics)

    @classmethod
    def load(cls, path: str):
        _require_torch()
        checkpoint = load_pipeline_artifact(path)
        instance = cls(
            project_config={},
            model_params=dict(checkpoint.get("config", {}).get("model_params", {})),
            training_params=dict(checkpoint.get("config", {}).get("training_params", {})),
            preprocessing_params=dict(checkpoint.get("config", {}).get("preprocessing_params", {})),
        )
        input_dim = int(checkpoint.get("input_dim") or 1)
        instance.input_dim = input_dim
        instance.composition_dim = int(checkpoint.get("composition_dim") or 0)
        hparams = instance._resolve_hparams(input_dim)
        instance.device = str(checkpoint.get("device", "cpu"))
        instance.input_scaler = checkpoint.get("input_scaler")
        instance.target_scaler = checkpoint.get("target_scaler")
        instance.model = instance._build_model(input_dim, hparams, instance.composition_dim)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        instance.metrics = checkpoint.get("metrics", {})
        return instance

