# Modular CNN + Transformer Architecture Design

**Date:** 2026-04-15  
**Status:** Approved  
**Author:** Birrul Walidain  

---

## 1. Overview

This document describes the modular architecture for separating CNN and Transformer components into independent, reusable modules. The design enables:

- **Independent usage**: CNN and Transformer can be used separately
- **Composable fusion**: Hybrid models combine both architectures cleanly
- **Vanilla PyTorch primitives**: Minimal custom complexity, maximum compatibility
- **Easy experimentation**: Swap components without rewriting entire models

---

## 2. Architecture

### 2.1 Directory Structure

```
src/models/
├── cnn/
│   ├── __init__.py
│   ├── backbone.py          # CNN backbone (Conv1d + pooling)
│   └── classifier.py        # Classification/regression head
├── transformer/
│   ├── __init__.py
│   ├── encoder.py           # Transformer Encoder (vanilla nn.TransformerEncoder)
│   └── decoder.py           # Transformer Decoder (optional, for seq2seq)
├── hybrid/
│   ├── __init__.py
│   ├── cnn_transformer.py   # Fusion: CNN encoder + Transformer encoder
│   └── cross_attention.py   # Cross-attention between CNN/Transformer features
└── registry.py              # Model registry (existing, updated)
```

### 2.2 Component Responsibilities

| Component | Purpose | Dependencies |
|-----------|---------|--------------|
| `cnn/backbone.py` | Extract spectral features via Conv1d layers | `torch.nn` only |
| `cnn/classifier.py` | Map CNN features to predictions | `cnn/backbone.py` |
| `transformer/encoder.py` | Sequence modeling with self-attention | `torch.nn.TransformerEncoder` |
| `transformer/decoder.py` | Optional decoder for seq2seq tasks | `torch.nn.TransformerDecoder` |
| `hybrid/cnn_transformer.py` | Combine CNN features + Transformer attention | `cnn/*`, `transformer/*` |
| `hybrid/cross_attention.py` | Cross-attention mechanism between modalities | `torch.nn.MultiheadAttention` |

---

## 3. Component Specifications

### 3.1 CNN Module

**File:** `src/models/cnn/backbone.py`

```python
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
        depths: list[int] = [64, 128, 256, 512],
        kernel_sizes: list[int] = [7, 5, 3, 3],
        dropout: float = 0.1,
    ):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, sequence_length)
        Returns:
            Feature tensor (batch, features)
        """
        pass
```

**File:** `src/models/cnn/classifier.py`

```python
class CNNClassifier(nn.Module):
    """
    CNN-based classifier for LIBS spectra.
    
    Uses CNNBackbone + MLP head for regression (Te, Ne, compositions).
    """
    
    def __init__(
        self,
        backbone: CNNBackbone,
        num_outputs: int = 2,  # Te, Ne
        hidden_dim: int = 256,
    ):
        pass
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with 'thermo' key (batch, num_outputs)
        """
        pass
```

### 3.2 Transformer Module

**File:** `src/models/transformer/encoder.py`

```python
class TransformerEncoder(nn.Module):
    """
    Vanilla Transformer Encoder for spectral sequence modeling.
    
    Uses PyTorch's nn.TransformerEncoder with standard positional encoding.
    
    Architecture:
    - Input: (batch, sequence_length, features)
    - Positional encoding (learnable or sinusoidal)
    - Multiple TransformerEncoderLayer blocks
    - Global average pooling
    - Output: (batch, features)
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
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, sequence_length, input_dim)
        Returns:
            Encoded tensor (batch, d_model)
        """
        pass
```

**File:** `src/models/transformer/decoder.py`

```python
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
        pass
    
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass
```

### 3.3 Hybrid Module

**File:** `src/models/hybrid/cnn_transformer.py`

```python
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
        projection_dim: int = 256,
    ):
        pass
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input spectra (batch, 1, sequence_length)
        Returns:
            Dictionary with:
            - 'thermo': (batch, 2) - Te, Ne predictions
            - 'features': (batch, projection_dim) - intermediate features
        """
        pass
```

**File:** `src/models/hybrid/cross_attention.py`

```python
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
        pass
    
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
            Attended features for both modalities
        """
        pass
```

---

## 4. Integration with Existing Code

### 4.1 Model Registry Update

Update `src/models/registry.py` to support new modular models:

```python
MODEL_REGISTRY = {
    # Existing models
    "pi": PIModel,
    "svr": SVRModel,
    
    # New modular models
    "cnn": CNNClassifier,  # Pure CNN
    "transformer": TransformerEncoder,  # Pure Transformer
    "cnn_transformer": CNNTransformerModel,  # Hybrid
}
```

### 4.2 Backward Compatibility

The existing `cnn_transformer.py` (524 lines) will be:
1. **Refactored** into the new modular structure
2. **Kept as wrapper** for backward compatibility during transition
3. **Deprecated** with warning after migration complete

### 4.3 Training Script Compatibility

No changes needed to `scripts/train_model.py` - model selection happens via registry.

---

## 5. Data Flow

```
Input Spectra (batch, 1, seq_len)
        │
        ▼
┌───────────────────┐
│  CNN Backbone     │  → Extract local spectral features
│  (Conv1d blocks)  │
└───────────────────┘
        │
        ▼
Features (batch, cnn_features)
        │
        ▼
┌───────────────────┐
│  Projection       │  → Map to Transformer dimension
│  (Linear + LayerNorm) │
└───────────────────┘
        │
        ▼
Projected (batch, seq_len, d_model)
        │
        ▼
┌───────────────────┐
│  Transformer      │  → Model long-range dependencies
│  Encoder          │
│  (Self-Attention) │
└───────────────────┘
        │
        ▼
Encoded (batch, d_model)
        │
        ▼
┌───────────────────┐
│  Classification   │  → Predict Te, Ne
│  Head (MLP)       │
└───────────────────┘
        │
        ▼
Output: {'thermo': (batch, 2)}
```

---

## 6. Error Handling

| Error | Handling |
|-------|----------|
| Invalid input shape | Raise `ValueError` with expected shape |
| Missing PyTorch | Import error with helpful message |
| Dimension mismatch | Assert in `forward()` with clear message |
| NaN/Inf in output | Optional check in training loop |

---

## 7. Testing Strategy

### 7.1 Unit Tests

- `tests/test_cnn_backbone.py` - Test CNN feature extraction
- `tests/test_transformer_encoder.py` - Test Transformer encoding
- `tests/test_cross_attention.py` - Test cross-attention mechanism

### 7.2 Integration Tests

- `tests/test_cnn_transformer.py` - Test full hybrid model
- `tests/test_model_registry.py` - Test model creation via registry

### 7.3 Smoke Tests

```python
# Test CNN
from src.models.cnn import CNNClassifier, CNNBackbone
cnn = CNNClassifier(CNNBackbone(), num_outputs=2)
x = torch.randn(4, 1, 2048)
out = cnn(x)
assert out['thermo'].shape == (4, 2)

# Test Transformer
from src.models.transformer import TransformerEncoder
tf = TransformerEncoder(input_dim=2048, d_model=256)
x = torch.randn(4, 2048, 256)
out = tf(x)
assert out.shape == (4, 256)

# Test Hybrid
from src.models.hybrid import CNNTransformerModel
model = CNNTransformerModel(CNNBackbone(), TransformerEncoder(...))
x = torch.randn(4, 1, 2048)
out = model(x)
assert out['thermo'].shape == (4, 2)
```

---

## 8. Migration Plan

### Phase 1: Create New Modules
1. Create `src/models/cnn/` directory
2. Create `src/models/transformer/` directory
3. Create `src/models/hybrid/` directory
4. Implement backbone, encoder, classifier components

### Phase 2: Implement Hybrid Model
1. Implement `CNNTransformerModel` using new components
2. Implement `CrossAttentionModule` (optional advanced feature)
3. Verify output matches existing implementation

### Phase 3: Update Registry
1. Update `src/models/registry.py` with new models
2. Add backward compatibility wrapper for old `cnn_transformer.py`
3. Update imports in training scripts

### Phase 4: Testing
1. Write unit tests for each component
2. Write integration tests for hybrid model
3. Run existing test suite to ensure no regressions

### Phase 5: Documentation
1. Add docstrings to all new modules
2. Update README with new architecture
3. Add usage examples

---

## 9. Success Criteria

- [ ] All components pass unit tests
- [ ] Hybrid model matches existing implementation output (within tolerance)
- [ ] Training script works with new modular models
- [ ] Model registry correctly instantiates all variants
- [ ] No breaking changes to existing API

---

## 10. References

- **PyTorch Transformer API**: https://pytorch.org/docs/stable/nn.html#transformer
- **lucidrains/vit-pytorch**: https://github.com/lucidrains/vit-pytorch
- **huggingface/pytorch-image-models**: https://github.com/huggingface/pytorch-image-models
- **Original CNN-Transformer implementation**: `src/models/cnn_transformer.py` (524 lines)
