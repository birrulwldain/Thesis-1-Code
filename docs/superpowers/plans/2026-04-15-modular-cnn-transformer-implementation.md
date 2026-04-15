# Modular CNN + Transformer Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor monolithic `cnn_transformer.py` into modular CNN, Transformer, and Hybrid components with clean separation of concerns.

**Architecture:** Three independent modules (cnn, transformer, hybrid) with composable interfaces. CNN extracts local spectral features, Transformer models long-range dependencies, Hybrid combines both.

**Tech Stack:** PyTorch 2.x, Python 3.11+, numpy, sklearn for preprocessing

---

## File Structure

### New Files to Create

| File | Responsibility |
|------|----------------|
| `src/models/cnn/__init__.py` | Export CNNBackbone, CNNClassifier |
| `src/models/cnn/backbone.py` | CNN feature extraction with Conv1d |
| `src/models/cnn/classifier.py` | Classification head for CNN |
| `src/models/transformer/__init__.py` | Export TransformerEncoder, TransformerDecoder |
| `src/models/transformer/encoder.py` | Vanilla Transformer Encoder |
| `src/models/transformer/decoder.py` | Optional Transformer Decoder |
| `src/models/hybrid/__init__.py` | Export CNNTransformerModel, CrossAttentionModule |
| `src/models/hybrid/cnn_transformer.py` | Fusion of CNN + Transformer |
| `src/models/hybrid/cross_attention.py` | Cross-attention mechanism |

### Files to Modify

| File | Changes |
|------|---------|
| `src/models/registry.py` | Add new modular models to registry |
| `src/models/cnn.py` | Deprecate stub, redirect to new module |

### Test Files to Create

| File | Tests |
|------|-------|
| `tests/models/test_cnn_backbone.py` | CNN backbone unit tests |
| `tests/models/test_cnn_classifier.py` | CNN classifier unit tests |
| `tests/models/test_transformer_encoder.py` | Transformer encoder unit tests |
| `tests/models/test_cnn_transformer.py` | Integration tests |

---

## Task 1: Create CNN Module Structure

**Files:**
- Create: `src/models/cnn/__init__.py`
- Create: `src/models/cnn/backbone.py`
- Test: `tests/models/test_cnn_backbone.py`

- [ ] **Step 1: Create CNN module directory**

```bash
mkdir -p src/models/cnn
```

- [ ] **Step 2: Create CNN backbone implementation**

Create `src/models/cnn/backbone.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """
    Vanilla CNN backbone for 1D spectral data.
    
    Architecture:
    - Input: (batch, channels, sequence_length)
    - Multiple Conv1d + BatchNorm1d + GELU blocks
    - AdaptiveAvgPool1d for global feature extraction
    - Output: (batch, features)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        features: int = 512,
        depths: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if depths is None:
            depths = [64, 128, 256, 512]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3, 3]
        
        if len(depths) != len(kernel_sizes):
            raise ValueError(
                f"depths and kernel_sizes must have same length. "
                f"Got {len(depths)} and {len(kernel_sizes)}"
            )
        
        layers = []
        in_dim = in_channels
        
        for i, (out_dim, kernel_size) in enumerate(zip(depths, kernel_sizes)):
            padding = kernel_size // 2
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(depths) - 1 else 0.0),
            ])
            if i < len(depths) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_dim = out_dim
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(depths[-1], features) if depths[-1] != features else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, sequence_length)
        Returns:
            Feature tensor (batch, features)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, channels, seq_len), got {x.dim()}D"
            )
        
        features = self.conv_layers(x)
        pooled = self.global_pool(features).squeeze(-1)
        output = self.project(pooled)
        return output


def create_cnn_backbone(
    in_channels: int = 1,
    features: int = 512,
    dropout: float = 0.1,
) -> CNNBackbone:
    """Factory function to create standard CNN backbone."""
    return CNNBackbone(
        in_channels=in_channels,
        features=features,
        dropout=dropout,
    )
```

- [ ] **Step 3: Create CNN module exports**

Create `src/models/cnn/__init__.py`:

```python
from __future__ import annotations

from src.models.cnn.backbone import CNNBackbone, create_cnn_backbone
from src.models.cnn.classifier import CNNClassifier

__all__ = [
    "CNNBackbone",
    "CNNClassifier",
    "create_cnn_backbone",
]
```

- [ ] **Step 4: Write failing test for CNN backbone**

Create `tests/models/test_cnn_backbone.py`:

```python
import pytest
import torch
from src.models.cnn.backbone import CNNBackbone, create_cnn_backbone


class TestCNNBackbone:
    def test_forward_pass(self):
        """Test basic forward pass with valid input."""
        model = CNNBackbone(in_channels=1, features=512)
        x = torch.randn(4, 1, 2048)
        output = model(x)
        assert output.shape == (4, 512)
    
    def test_custom_depths(self):
        """Test with custom depth configuration."""
        model = CNNBackbone(
            in_channels=1,
            features=256,
            depths=[32, 64, 128],
            kernel_sizes=[5, 3, 3],
        )
        x = torch.randn(2, 1, 1024)
        output = model(x)
        assert output.shape == (2, 256)
    
    def test_invalid_input_dimension(self):
        """Test error handling for invalid input dimensions."""
        model = CNNBackbone()
        x = torch.randn(4, 1, 2048, 3)  # 4D instead of 3D
        with pytest.raises(ValueError, match="Expected 3D input"):
            model(x)
    
    def test_depth_kernel_mismatch(self):
        """Test error when depths and kernel_sizes don't match."""
        with pytest.raises(ValueError, match="same length"):
            CNNBackbone(
                depths=[64, 128],
                kernel_sizes=[7, 5, 3],
            )
    
    def test_factory_function(self):
        """Test the factory function creates valid model."""
        model = create_cnn_backbone(features=256, dropout=0.2)
        assert isinstance(model, CNNBackbone)
        x = torch.randn(2, 1, 2048)
        output = model(x)
        assert output.shape == (2, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 5: Run test to verify it fails**

```bash
cd /Users/birrulwldain/Projects/Thesis-1-Code
python -m pytest tests/models/test_cnn_backbone.py -v
```

Expected: Tests PASS (since we already implemented the code in Step 2)

- [ ] **Step 6: Commit**

```bash
git add src/models/cnn/__init__.py src/models/cnn/backbone.py tests/models/test_cnn_backbone.py
git commit -m "feat: add CNN backbone module with Conv1d feature extraction"
```

---

## Task 2: Create CNN Classifier Module

**Files:**
- Create: `src/models/cnn/classifier.py`
- Modify: `src/models/cnn/__init__.py`
- Test: `tests/models/test_cnn_classifier.py`

- [ ] **Step 1: Write failing test for CNN classifier**

Create `tests/models/test_cnn_classifier.py`:

```python
import pytest
import torch
from src.models.cnn.backbone import CNNBackbone
from src.models.cnn.classifier import CNNClassifier


class TestCNNClassifier:
    def test_thermo_prediction(self):
        """Test prediction of Te and Ne."""
        backbone = CNNBackbone(features=512)
        model = CNNClassifier(backbone, num_outputs=2)
        
        x = torch.randn(4, 1, 2048)
        output = model(x)
        
        assert "thermo" in output
        assert output["thermo"].shape == (4, 2)
    
    def test_composition_prediction(self):
        """Test prediction with composition outputs."""
        backbone = CNNBackbone(features=512)
        model = CNNClassifier(
            backbone, 
            num_outputs=2,
            num_compositions=11,  # Si, Al, Fe, Ca, Mg, Na, K, Ti, Mn, P, Ba
        )
        
        x = torch.randn(2, 1, 2048)
        output = model(x)
        
        assert "thermo" in output
        assert output["thermo"].shape == (2, 2)
        assert "composition" in output
        assert output["composition"].shape == (2, 11)
    
    def test_shared_backbone(self):
        """Test that backbone can be shared across classifiers."""
        backbone = CNNBackbone(features=256)
        
        model1 = CNNClassifier(backbone, num_outputs=2)
        model2 = CNNClassifier(backbone, num_outputs=2)
        
        x = torch.randn(2, 1, 1024)
        out1 = model1(x)
        out2 = model2(x)
        
        assert out1["thermo"].shape == (2, 2)
        assert out2["thermo"].shape == (2, 2)
    
    def test_hidden_dim_configuration(self):
        """Test custom hidden dimension in classifier head."""
        backbone = CNNBackbone(features=512)
        model = CNNClassifier(
            backbone,
            num_outputs=2,
            hidden_dim=128,
        )
        
        x = torch.randn(2, 1, 2048)
        output = model(x)
        assert output["thermo"].shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/models/test_cnn_classifier.py::TestCNNClassifier::test_thermo_prediction -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.models.cnn.classifier'"

- [ ] **Step 3: Implement CNN classifier**

Create `src/models/cnn/classifier.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from src.models.cnn.backbone import CNNBackbone


class CNNClassifier(nn.Module):
    """
    CNN-based classifier for LIBS spectra.
    
    Uses CNNBackbone + MLP head for regression (Te, Ne, compositions).
    """
    
    def __init__(
        self,
        backbone: CNNBackbone,
        num_outputs: int = 2,
        num_compositions: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.num_outputs = num_outputs
        self.num_compositions = num_compositions
        
        # Thermo head (Te, Ne)
        self.thermo_head = nn.Sequential(
            nn.Linear(backbone.features if hasattr(backbone, 'features') else 512, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )
        
        # Composition head (optional)
        self.composition_head = None
        if num_compositions > 0:
            self.composition_head = nn.Sequential(
                nn.Linear(backbone.features if hasattr(backbone, 'features') else 512, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_compositions),
                nn.Softmax(dim=-1),
            )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input spectra (batch, channels, sequence_length)
        Returns:
            Dictionary with:
            - 'thermo': (batch, num_outputs) - Te, Ne predictions
            - 'composition': (batch, num_compositions) - if enabled
        """
        features = self.backbone(x)
        
        output = {
            "thermo": self.thermo_head(features),
        }
        
        if self.composition_head is not None:
            output["composition"] = self.composition_head(features)
        
        return output
```

- [ ] **Step 4: Update CNN module exports**

Modify `src/models/cnn/__init__.py`:

```python
from __future__ import annotations

from src.models.cnn.backbone import CNNBackbone, create_cnn_backbone
from src.models.cnn.classifier import CNNClassifier

__all__ = [
    "CNNBackbone",
    "CNNClassifier",
    "create_cnn_backbone",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/models/test_cnn_classifier.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/cnn/classifier.py src/models/cnn/__init__.py tests/models/test_cnn_classifier.py
git commit -m "feat: add CNN classifier module with thermo and composition heads"
```

---

## Task 3: Create Transformer Encoder Module

**Files:**
- Create: `src/models/transformer/__init__.py`
- Create: `src/models/transformer/encoder.py`
- Test: `tests/models/test_transformer_encoder.py`

- [ ] **Step 1: Create Transformer module directory**

```bash
mkdir -p src/models/transformer
```

- [ ] **Step 2: Write failing test for Transformer encoder**

Create `tests/models/test_transformer_encoder.py`:

```python
import pytest
import torch
from src.models.transformer.encoder import TransformerEncoder


class TestTransformerEncoder:
    def test_forward_pass(self):
        """Test basic forward pass with valid input."""
        model = TransformerEncoder(input_dim=256, d_model=256)
        x = torch.randn(4, 100, 256)  # (batch, seq_len, features)
        output = model(x)
        assert output.shape == (4, 256)
    
    def test_custom_configuration(self):
        """Test with custom transformer configuration."""
        model = TransformerEncoder(
            input_dim=512,
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=2048,
        )
        x = torch.randn(2, 50, 512)
        output = model(x)
        assert output.shape == (2, 512)
    
    def test_positional_encoding(self):
        """Test that positional encoding is applied."""
        model = TransformerEncoder(
            input_dim=128,
            d_model=128,
            max_len=1000,
        )
        
        # Same input at different positions should give different outputs
        x1 = torch.zeros(1, 10, 128)
        x2 = torch.zeros(1, 10, 128)
        
        # Add different patterns
        x1[:, ::2, :] = 1.0
        x2[:, 1::2, :] = 1.0
        
        out1 = model(x1)
        out2 = model(x2)
        
        # Outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_dropout_applied(self):
        """Test that dropout is applied in training mode."""
        model = TransformerEncoder(
            input_dim=256,
            d_model=256,
            dropout=0.3,
        )
        model.train()
        
        x = torch.randn(4, 50, 256)
        output = model(x)
        assert output.shape == (4, 256)
    
    def test_max_sequence_length(self):
        """Test handling of sequence length within max_len."""
        model = TransformerEncoder(
            input_dim=256,
            d_model=256,
            max_len=500,
        )
        
        # Should work within max_len
        x = torch.randn(2, 400, 256)
        output = model(x)
        assert output.shape == (2, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/models/test_transformer_encoder.py::TestTransformerEncoder::test_forward_pass -v
```

Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 4: Implement Transformer encoder**

Create `src/models/transformer/encoder.py`:

```python
from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (seq_len, batch, d_model)
        """
        x = x + self.dropout(self.pe[:x.size(0)])
        return x


class TransformerEncoder(nn.Module):
    """
    Vanilla Transformer Encoder for spectral sequence modeling.
    
    Uses PyTorch's nn.TransformerEncoder with standard positional encoding.
    
    Architecture:
    - Input: (batch, sequence_length, features)
    - Positional encoding (sinusoidal)
    - Multiple TransformerEncoderLayer blocks
    - Global average pooling
    - Output: (batch, d_model)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # We use (seq_len, batch, features) for transformer
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, sequence_length, input_dim)
        Returns:
            Encoded tensor (batch, d_model)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, input_dim), got {x.dim()}D"
            )
        
        batch_size, seq_len, _ = x.shape
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Back to (batch, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)
        
        return x


def create_transformer_encoder(
    input_dim: int,
    d_model: int = 256,
    num_layers: int = 6,
    dropout: float = 0.1,
) -> TransformerEncoder:
    """Factory function to create standard transformer encoder."""
    return TransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
    )
```

- [ ] **Step 5: Create Transformer module exports**

Create `src/models/transformer/__init__.py`:

```python
from __future__ import annotations

from src.models.transformer.encoder import TransformerEncoder, create_transformer_encoder
from src.models.transformer.decoder import TransformerDecoder

__all__ = [
    "TransformerEncoder",
    "TransformerDecoder",
    "create_transformer_encoder",
]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/models/test_transformer_encoder.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/models/transformer/__init__.py src/models/transformer/encoder.py tests/models/test_transformer_encoder.py
git commit -m "feat: add Transformer encoder module with positional encoding"
```

---

## Task 4: Create Transformer Decoder Module

**Files:**
- Create: `src/models/transformer/decoder.py`
- Modify: `src/models/transformer/__init__.py`
- Test: `tests/models/test_transformer_decoder.py`

- [ ] **Step 1: Write failing test for Transformer decoder**

Create `tests/models/test_transformer_decoder.py`:

```python
import pytest
import torch
from src.models.transformer.decoder import TransformerDecoder


class TestTransformerDecoder:
    def test_forward_pass(self):
        """Test basic forward pass with valid input."""
        model = TransformerDecoder(d_model=256, num_layers=4)
        
        tgt = torch.randn(4, 50, 256)  # (batch, tgt_len, d_model)
        memory = torch.randn(4, 100, 256)  # (batch, src_len, d_model)
        
        output = model(tgt, memory)
        assert output.shape == (4, 50, 256)
    
    def test_with_masks(self):
        """Test forward pass with attention masks."""
        model = TransformerDecoder(d_model=256, num_layers=2)
        
        tgt = torch.randn(2, 30, 256)
        memory = torch.randn(2, 60, 256)
        
        tgt_mask = torch.ones(30, 30).bool()
        memory_mask = torch.ones(60, 60).bool()
        
        output = model(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        assert output.shape == (2, 30, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/models/test_transformer_decoder.py::TestTransformerDecoder::test_forward_pass -v
```

Expected: FAIL

- [ ] **Step 3: Implement Transformer decoder**

Create `src/models/transformer/decoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class TransformerDecoder(nn.Module):
    """
    Optional Transformer Decoder for seq2seq tasks.
    
    Used when decoder needs to attend to encoder outputs.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
    
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target tensor (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Target attention mask (tgt_len, tgt_len)
            memory_mask: Memory attention mask (src_len, src_len)
        Returns:
            Decoded tensor (batch, tgt_len, d_model)
        """
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return output
```

- [ ] **Step 4: Update Transformer module exports**

Modify `src/models/transformer/__init__.py`:

```python
from __future__ import annotations

from src.models.transformer.encoder import TransformerEncoder, create_transformer_encoder
from src.models.transformer.decoder import TransformerDecoder

__all__ = [
    "TransformerEncoder",
    "TransformerDecoder",
    "create_transformer_encoder",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/models/test_transformer_decoder.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/transformer/decoder.py src/models/transformer/__init__.py tests/models/test_transformer_decoder.py
git commit -m "feat: add Transformer decoder module for seq2seq tasks"
```

---

## Task 5: Create Cross-Attention Module

**Files:**
- Create: `src/models/hybrid/__init__.py`
- Create: `src/models/hybrid/cross_attention.py`
- Test: `tests/models/test_cross_attention.py`

- [ ] **Step 1: Create Hybrid module directory**

```bash
mkdir -p src/models/hybrid
```

- [ ] **Step 2: Write failing test for cross-attention**

Create `tests/models/test_cross_attention.py`:

```python
import pytest
import torch
from src.models.hybrid.cross_attention import CrossAttentionModule


class TestCrossAttentionModule:
    def test_forward_pass(self):
        """Test basic forward pass with valid input."""
        model = CrossAttentionModule(
            cnn_dim=512,
            transformer_dim=256,
            embed_dim=256,
        )
        
        cnn_features = torch.randn(4, 100, 512)
        transformer_features = torch.randn(4, 100, 256)
        
        cnn_out, tf_out = model(cnn_features, transformer_features)
        
        assert cnn_out.shape == (4, 100, 256)
        assert tf_out.shape == (4, 100, 256)
    
    def test_dimension_mismatch(self):
        """Test error when sequence lengths don't match."""
        model = CrossAttentionModule(
            cnn_dim=512,
            transformer_dim=256,
            embed_dim=256,
        )
        
        cnn_features = torch.randn(2, 50, 512)
        transformer_features = torch.randn(2, 60, 256)  # Different seq_len
        
        with pytest.raises(RuntimeError):
            model(cnn_features, transformer_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/models/test_cross_attention.py::TestCrossAttentionModule::test_forward_pass -v
```

Expected: FAIL

- [ ] **Step 4: Implement cross-attention module**

Create `src/models/hybrid/cross_attention.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class CrossAttentionModule(nn.Module):
    """
    Cross-attention between CNN and Transformer features.
    
    Used in advanced hybrid architectures where CNN and Transformer
    features attend to each other before fusion.
    """
    
    def __init__(
        self,
        cnn_dim: int,
        transformer_dim: int,
        embed_dim: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cnn_dim = cnn_dim
        self.transformer_dim = transformer_dim
        self.embed_dim = embed_dim
        
        # Projection layers
        self.cnn_project = nn.Linear(cnn_dim, embed_dim)
        self.transformer_project = nn.Linear(transformer_dim, embed_dim)
        
        # Cross-attention: CNN attends to Transformer
        self.cnn_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: Transformer attends to CNN
        self.transformer_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projections
        self.cnn_out = nn.Linear(embed_dim, embed_dim)
        self.transformer_out = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        cnn_features: torch.Tensor, 
        transformer_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_features: (batch, seq_len, cnn_dim)
            transformer_features: (batch, seq_len, transformer_dim)
        Returns:
            Attended features for both modalities:
            - cnn_attended: (batch, seq_len, embed_dim)
            - transformer_attended: (batch, seq_len, embed_dim)
        """
        # Project to common dimension
        cnn_proj = self.cnn_project(cnn_features)
        tf_proj = self.transformer_project(transformer_features)
        
        # CNN attends to Transformer
        cnn_attended, _ = self.cnn_attention(
            query=cnn_proj,
            key=tf_proj,
            value=tf_proj,
        )
        cnn_attended = self.dropout(cnn_attended)
        cnn_attended = self.norm(cnn_attended + cnn_proj)
        
        # Transformer attends to CNN
        tf_attended, _ = self.transformer_attention(
            query=tf_proj,
            key=cnn_proj,
            value=cnn_proj,
        )
        tf_attended = self.dropout(tf_attended)
        tf_attended = self.norm(tf_attended + tf_proj)
        
        # Final output projection
        cnn_out = self.cnn_out(cnn_attended)
        tf_out = self.transformer_out(tf_attended)
        
        return cnn_out, tf_out
```

- [ ] **Step 5: Create Hybrid module exports**

Create `src/models/hybrid/__init__.py`:

```python
from __future__ import annotations

from src.models.hybrid.cross_attention import CrossAttentionModule
from src.models.hybrid.cnn_transformer import CNNTransformerModel

__all__ = [
    "CNNTransformerModel",
    "CrossAttentionModule",
]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/models/test_cross_attention.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/models/hybrid/__init__.py src/models/hybrid/cross_attention.py tests/models/test_cross_attention.py
git commit -m "feat: add cross-attention module for multi-modal feature fusion"
```

---

## Task 6: Create CNN-Transformer Hybrid Model

**Files:**
- Create: `src/models/hybrid/cnn_transformer.py`
- Modify: `src/models/hybrid/__init__.py`
- Test: `tests/models/test_cnn_transformer.py`

- [ ] **Step 1: Write failing test for hybrid model**

Create `tests/models/test_cnn_transformer.py`:

```python
import pytest
import torch
from src.models.cnn.backbone import CNNBackbone
from src.models.transformer.encoder import TransformerEncoder
from src.models.hybrid.cnn_transformer import CNNTransformerModel


class TestCNNTransformerModel:
    def test_forward_pass(self):
        """Test basic forward pass with valid input."""
        cnn_backbone = CNNBackbone(in_channels=1, features=256)
        transformer_encoder = TransformerEncoder(input_dim=256, d_model=256)
        
        model = CNNTransformerModel(
            cnn_backbone=cnn_backbone,
            transformer_encoder=transformer_encoder,
            num_outputs=2,
        )
        
        x = torch.randn(4, 1, 2048)
        output = model(x)
        
        assert "thermo" in output
        assert output["thermo"].shape == (4, 2)
    
    def test_with_composition_head(self):
        """Test model with composition prediction."""
        cnn_backbone = CNNBackbone(features=256)
        transformer_encoder = TransformerEncoder(input_dim=256, d_model=256)
        
        model = CNNTransformerModel(
            cnn_backbone=cnn_backbone,
            transformer_encoder=transformer_encoder,
            num_outputs=2,
            num_compositions=11,
        )
        
        x = torch.randn(2, 1, 2048)
        output = model(x)
        
        assert "thermo" in output
        assert output["thermo"].shape == (2, 2)
        assert "composition" in output
        assert output["composition"].shape == (2, 11)
    
    def test_feature_extraction(self):
        """Test that intermediate features are returned."""
        cnn_backbone = CNNBackbone(features=128)
        transformer_encoder = TransformerEncoder(input_dim=128, d_model=128)
        
        model = CNNTransformerModel(
            cnn_backbone=cnn_backbone,
            transformer_encoder=transformer_encoder,
            num_outputs=2,
        )
        
        x = torch.randn(2, 1, 1024)
        output = model(x, return_features=True)
        
        assert "thermo" in output
        assert "features" in output
        assert output["features"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/models/test_cnn_transformer.py::TestCNNTransformerModel::test_forward_pass -v
```

Expected: FAIL

- [ ] **Step 3: Implement CNN-Transformer hybrid model**

Create `src/models/hybrid/cnn_transformer.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from src.models.cnn.backbone import CNNBackbone
from src.models.transformer.encoder import TransformerEncoder


class CNNTransformerModel(nn.Module):
    """
    Hybrid model combining CNN feature extraction with Transformer attention.
    
    Architecture:
    1. CNN backbone extracts local spectral features
    2. Project CNN features to Transformer dimension
    3. Transformer Encoder models long-range dependencies
    4. Classification head predicts Te, Ne, compositions
    
    This is the main model for LIBS inversion task.
    """
    
    def __init__(
        self,
        cnn_backbone: CNNBackbone,
        transformer_encoder: TransformerEncoder,
        num_outputs: int = 2,
        num_compositions: int = 0,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cnn_backbone = cnn_backbone
        self.transformer_encoder = transformer_encoder
        self.num_outputs = num_outputs
        self.num_compositions = num_compositions
        self.projection_dim = projection_dim
        
        # Feature projection (CNN features -> Transformer input)
        cnn_features = cnn_backbone.features if hasattr(cnn_backbone, 'features') else 512
        self.feature_projection = nn.Sequential(
            nn.Linear(cnn_features, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )
        
        # Thermo head
        self.thermo_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_outputs),
        )
        
        # Composition head (optional)
        self.composition_head = None
        if num_compositions > 0:
            self.composition_head = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, num_compositions),
                nn.Softmax(dim=-1),
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input spectra (batch, channels, sequence_length)
            return_features: Whether to return intermediate features
        Returns:
            Dictionary with:
            - 'thermo': (batch, num_outputs) - Te, Ne predictions
            - 'composition': (batch, num_compositions) - if enabled
            - 'features': (batch, projection_dim) - if return_features=True
        """
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Project to Transformer dimension
        projected = self.feature_projection(cnn_features)
        
        # Reshape for transformer: (batch, 1, projection_dim)
        projected = projected.unsqueeze(1)
        
        # Transformer encoding
        encoded = self.transformer_encoder(projected)
        
        # Classification
        output = {
            "thermo": self.thermo_head(encoded),
        }
        
        if self.composition_head is not None:
            output["composition"] = self.composition_head(encoded)
        
        if return_features:
            output["features"] = encoded
        
        return output


def create_cnn_transformer(
    in_channels: int = 1,
    cnn_features: int = 512,
    transformer_dim: int = 256,
    num_outputs: int = 2,
    num_layers: int = 6,
    dropout: float = 0.1,
) -> CNNTransformerModel:
    """Factory function to create standard CNN-Transformer model."""
    cnn_backbone = CNNBackbone(
        in_channels=in_channels,
        features=cnn_features,
        dropout=dropout,
    )
    transformer_encoder = TransformerEncoder(
        input_dim=cnn_features,
        d_model=transformer_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    return CNNTransformerModel(
        cnn_backbone=cnn_backbone,
        transformer_encoder=transformer_encoder,
        num_outputs=num_outputs,
        dropout=dropout,
    )
```

- [ ] **Step 4: Update Hybrid module exports**

Modify `src/models/hybrid/__init__.py`:

```python
from __future__ import annotations

from src.models.hybrid.cross_attention import CrossAttentionModule
from src.models.hybrid.cnn_transformer import CNNTransformerModel, create_cnn_transformer

__all__ = [
    "CNNTransformerModel",
    "CrossAttentionModule",
    "create_cnn_transformer",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/models/test_cnn_transformer.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/hybrid/cnn_transformer.py src/models/hybrid/__init__.py tests/models/test_cnn_transformer.py
git commit -m "feat: add CNN-Transformer hybrid model for LIBS inversion"
```

---

## Task 7: Update Model Registry

**Files:**
- Modify: `src/models/registry.py`
- Modify: `src/models/cnn.py`
- Test: `tests/models/test_registry.py`

- [ ] **Step 1: Write failing test for registry**

Create `tests/models/test_registry.py`:

```python
import pytest
from src.models.registry import available_models, create_model


class TestModelRegistry:
    def test_available_models(self):
        """Test that all expected models are available."""
        models = available_models()
        
        # Check new modular models
        assert "cnn" in models
        assert "transformer" in models
        assert "cnn_transformer" in models
        
        # Check existing models still available
        assert "pi" in models
        assert "svr" in models
    
    def test_create_cnn_model(self):
        """Test creating CNN model via registry."""
        model = create_model("cnn", project_config={})
        assert model is not None
    
    def test_create_transformer_model(self):
        """Test creating Transformer model via registry."""
        model = create_model("transformer", project_config={})
        assert model is not None
    
    def test_create_cnn_transformer_model(self):
        """Test creating hybrid model via registry."""
        model = create_model("cnn_transformer", project_config={})
        assert model is not None
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model name."""
        with pytest.raises(ValueError, match="tidak dikenal"):
            create_model("invalid_model", project_config={})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/models/test_registry.py -v
```

Expected: FAIL (registry not updated yet)

- [ ] **Step 3: Update model registry**

Modify `src/models/registry.py`:

```python
from __future__ import annotations

from typing import Any

from src.models.cnn.classifier import CNNClassifier
from src.models.cnn.backbone import CNNBackbone
from src.models.transformer.encoder import TransformerEncoder
from src.models.hybrid.cnn_transformer import CNNTransformerModel
from src.models.cnn_transformer import CNNTransformerModel as LegacyCNNTransformerModel
from src.models.pi import PIModel
from src.models.svr import SVRModel


MODEL_REGISTRY = {
    # Existing models
    "pi": PIModel,
    "svr": SVRModel,
    "cnn_transformer": CNNTransformerModel,
    
    # New modular models
    "cnn": CNNClassifier,
    "transformer": TransformerEncoder,
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def create_model(
    model_name: str,
    *,
    project_config: dict[str, Any],
    model_params: dict[str, Any] | None = None,
    training_params: dict[str, Any] | None = None,
    preprocessing_params: dict[str, Any] | None = None,
):
    normalized = str(model_name).strip().lower()
    try:
        model_cls = MODEL_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. Pilihan: {', '.join(available_models())}"
        ) from exc
    
    # Handle different model signatures
    if normalized == "cnn":
        return model_cls(
            backbone=CNNBackbone(),
            **(model_params or {}),
        )
    elif normalized == "transformer":
        return model_cls(
            input_dim=project_config.get("input_dim", 2048),
            **(model_params or {}),
        )
    else:
        return model_cls(
            project_config,
            model_params=model_params,
            training_params=training_params,
            preprocessing_params=preprocessing_params,
        )
```

- [ ] **Step 4: Deprecate old CNN stub**

Modify `src/models/cnn.py`:

```python
from __future__ import annotations

import warnings
from typing import Any

from src.models.cnn.classifier import CNNClassifier
from src.models.cnn.backbone import CNNBackbone


class CNNModel:
    """
    Deprecated: Use src.models.cnn.CNNClassifier instead.
    
    This stub is kept for backward compatibility only.
    """
    
    model_name = "cnn"
    
    def __init__(self, project_config: dict[str, Any], **kwargs) -> None:
        warnings.warn(
            "CNNModel is deprecated. Use src.models.cnn.CNNClassifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.project_config = project_config
        self.kwargs = kwargs
        
        # Create new modular model
        self._model = CNNClassifier(
            backbone=CNNBackbone(),
            **kwargs,
        )
    
    def fit(self, *args, **kwargs):
        warnings.warn(
            "CNNModel is deprecated. Use src.models.cnn.CNNClassifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._model.fit(*args, **kwargs)
    
    def predict(self, spectra):
        warnings.warn(
            "CNNModel is deprecated. Use src.models.cnn.CNNClassifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._model.predict(spectra)
    
    def save(self, output_model: str) -> None:
        warnings.warn(
            "CNNModel is deprecated. Use src.models.cnn.CNNClassifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._model.save(output_model)
    
    @classmethod
    def load(cls, path: str):
        warnings.warn(
            "CNNModel is deprecated. Use src.models.cnn.CNNClassifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls._model.load(path)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/models/test_registry.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/registry.py src/models/cnn.py tests/models/test_registry.py
git commit -m "feat: update model registry with new modular CNN/Transformer models"
```

---

## Task 8: Integration Testing and Validation

**Files:**
- Modify: `tests/test_libs_inversion.py` (if exists)
- Create: `tests/integration/test_model_training.py`

- [ ] **Step 1: Create integration test for model training**

Create `tests/integration/test_model_training.py`:

```python
import pytest
import torch
import numpy as np
from src.models.registry import create_model
from src.data.io import build_dataset_split


class TestModelTrainingIntegration:
    """Integration tests for model training pipeline."""
    
    def test_cnn_training(self):
        """Test complete training loop for CNN model."""
        model = create_model("cnn", project_config={})
        
        # Create dummy dataset
        spectra = torch.randn(10, 1, 2048)
        temperatures = torch.randn(10, 1) * 5000 + 10000
        electron_densities = torch.randn(10, 1) * 1e17 + 5e17
        
        # Forward pass
        output = model(spectra)
        assert output["thermo"].shape == (10, 2)
    
    def test_transformer_training(self):
        """Test complete training loop for Transformer model."""
        model = create_model(
            "transformer", 
            project_config={"input_dim": 2048},
        )
        
        # Create dummy dataset (batch, seq_len, features)
        x = torch.randn(4, 2048, 256)
        
        # Forward pass
        output = model(x)
        assert output.shape == (4, 256)
    
    def test_cnn_transformer_training(self):
        """Test complete training loop for CNN-Transformer hybrid."""
        model = create_model("cnn_transformer", project_config={})
        
        # Create dummy dataset
        spectra = torch.randn(8, 1, 2048)
        
        # Forward pass
        output = model(spectra)
        assert output["thermo"].shape == (8, 2)
    
    def test_all_models_in_registry(self):
        """Test that all registered models can be instantiated."""
        from src.models.registry import available_models
        
        for model_name in available_models():
            try:
                model = create_model(model_name, project_config={})
                assert model is not None, f"Model {model_name} returned None"
            except Exception as e:
                pytest.fail(f"Failed to create model {model_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run integration tests**

```bash
python -m pytest tests/integration/test_model_training.py -v
```

Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS (existing tests should still work)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_model_training.py tests/models/test_registry.py
git commit -m "test: add integration tests for modular model training pipeline"
```

---

## Task 9: Documentation and Cleanup

**Files:**
- Modify: `README.md`
- Modify: `src/models/README.md` (create if needed)

- [ ] **Step 1: Create models module documentation**

Create `src/models/README.md`:

```markdown
# Models Module

This directory contains modular model architectures for LIBS spectral inversion.

## Architecture

```
models/
├── cnn/              # Convolutional Neural Network modules
│   ├── backbone.py   # CNN feature extraction
│   └── classifier.py # Classification/regression head
├── transformer/      # Transformer modules
│   ├── encoder.py   # Transformer encoder
│   └── decoder.py   # Transformer decoder (optional)
├── hybrid/           # Hybrid models
│   ├── cnn_transformer.py  # CNN + Transformer fusion
│   └── cross_attention.py  # Cross-attention mechanism
└── registry.py       # Model registry
```

## Usage

### CNN Model

```python
from src.models.cnn import CNNClassifier, CNNBackbone

backbone = CNNBackbone(features=512)
model = CNNClassifier(backbone, num_outputs=2)

x = torch.randn(4, 1, 2048)
output = model(x)
```

### Transformer Model

```python
from src.models.transformer import TransformerEncoder

model = TransformerEncoder(input_dim=256, d_model=256, num_layers=6)

x = torch.randn(4, 100, 256)
output = model(x)
```

### CNN-Transformer Hybrid

```python
from src.models.hybrid import CNNTransformerModel, create_cnn_transformer

# Option 1: Factory function
model = create_cnn_transformer(
    cnn_features=512,
    transformer_dim=256,
    num_layers=6,
)

# Option 2: Manual construction
from src.models.cnn import CNNBackbone
from src.models.transformer import TransformerEncoder

cnn_backbone = CNNBackbone(features=512)
transformer_encoder = TransformerEncoder(input_dim=512, d_model=256)

model = CNNTransformerModel(
    cnn_backbone=cnn_backbone,
    transformer_encoder=transformer_encoder,
)

x = torch.randn(4, 1, 2048)
output = model(x)
```

### Via Registry

```python
from src.models.registry import create_model

model = create_model("cnn_transformer", project_config={})
```

## Testing

```bash
# Run all model tests
pytest tests/models/ -v

# Run specific module tests
pytest tests/models/test_cnn_backbone.py -v
pytest tests/models/test_transformer_encoder.py -v
pytest tests/models/test_cnn_transformer.py -v
```
```

- [ ] **Step 2: Update main README**

Add to `README.md` in Usage section:

```markdown
### Model Architectures

This repository supports multiple model architectures:

- **SVR** (`svr`): Support Vector Regression with PCA
- **PI** (`pi`): Physics-Informed neural network
- **CNN** (`cnn`): Pure Convolutional Neural Network
- **Transformer** (`transformer`): Pure Transformer Encoder
- **CNN-Transformer** (`cnn_transformer`): Hybrid fusion model (recommended)

To train a specific model:

```bash
# CNN-Transformer (recommended)
python scripts/train_model.py --model cnn_transformer --dataset dataset.h5

# Pure CNN
python scripts/train_model.py --model cnn --dataset dataset.h5

# Pure Transformer
python scripts/train_model.py --model transformer --dataset dataset.h5
```
```

- [ ] **Step 3: Commit**

```bash
git add src/models/README.md README.md
git commit -m "docs: add modular model architecture documentation"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** All components from spec document have corresponding tasks
- [ ] **No placeholders:** Every step contains actual code, no TBD/TODO
- [ ] **Type consistency:** Method signatures match across all tasks
- [ ] **Test coverage:** Each component has unit tests
- [ ] **Integration tests:** End-to-end training pipeline tested
- [ ] **Documentation:** Usage examples provided

---

## Success Criteria

- [ ] All 9 tasks completed with passing tests
- [ ] No breaking changes to existing API
- [ ] New modular models work with existing training scripts
- [ ] Documentation complete and accurate
- [ ] Code follows project style guidelines (black, type hints)

---

**Plan complete.** Two execution options:

1. **Subagent-Driven (recommended)** - Fresh subagent per task with review checkpoints
2. **Inline Execution** - Execute in this session with checkpoints

Which approach?
