
import torch
import torch.nn as nn
import numpy as np
import inspect
from typing import Optional, Tuple

from temporal.registry.core import register_module


# -----------------------------
# Base Embedding Interface
# -----------------------------
class BaseEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each embedding must implement its own forward method.")


# -----------------------------
# Value Embedding
# -----------------------------
@register_module("embedding", "value")
class TimeSeriesValueEmbedding(BaseEmbedding):
    def __init__(self, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_projection(x)


# -----------------------------
# Positional Embedding (Original - Commented Out)
# -----------------------------
# @register_module("embedding", "positional")
# class PositionalEmbedding(BaseEmbedding):
#     ...

# -----------------------------
# Sinusoidal Positional Embedding (Explicit Signature)
# -----------------------------
@register_module("embedding", "sinusoidal")
class SinusoidalPositionalEmbedding(BaseEmbedding):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__(d_model=dim)
        self.dim = dim
        self.max_seq_len = max_seq_len
        weights = self._init_weights()
        self.register_buffer('weight', weights)

    def _init_weights(self) -> torch.Tensor:
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / self.dim) for j in range(self.dim)]
                for pos in range(self.max_seq_len)
            ]
        )
        out = torch.zeros(self.max_seq_len, self.dim)
        sentinel = self.dim // 2 if self.dim % 2 == 0 else (self.dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        return out

    @torch.no_grad()
    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Generates positional embeddings based on explicit integer batch size, 
        sequence length, and past sequence length.
        """
        # We now expect integers directly from the caller (Decoder/Encoder)
        _bsz = batch_size
        _seq_len = seq_len
        _start = past_key_values_length

        # Early-exit on empty sequence
        if _seq_len <= 0:
            return torch.empty((_bsz, 0, self.dim), device=self.weight.device, dtype=self.weight.dtype)

        # Build position indices
        _end = _start + _seq_len
        if _end > self.max_seq_len:
            max_req_pos = _end - 1
            raise IndexError(
                f"Requested position index {max_req_pos} is out of bounds for "
                f"SinusoidalPositionalEmbedding with max_seq_len {self.max_seq_len}."
            )

        positions = torch.arange(_start, _end, dtype=torch.long, device=self.weight.device)

        # Check for empty positions defensively (should not happen if _seq_len > 0)
        if positions.numel() == 0:
            print(f"Warning: Position tensor empty after arange(start={_start}, end={_end}). Should not happen.")
            return torch.empty((_bsz, 0, self.dim), device=self.weight.device, dtype=self.weight.dtype)

        # Gather + unsqueeze for batch broadcast
        return self.weight[positions].unsqueeze(0)  # [1, seq_len, dim]


# -----------------------------
# Patch Embedding
# -----------------------------
@register_module("embedding", "patch")
class TimeSeriesPatchEmbedding(BaseEmbedding):
    def __init__(self, patch_size: int, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.feature_size = feature_size
        self.patch_projection = nn.Linear(patch_size * feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        if F != self.feature_size:
            raise ValueError(f"Input feature size ({F}) doesn't match model feature size ({self.feature_size}).")
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length ({L}) is not divisible by patch size ({self.patch_size}).")

        num_patches = L // self.patch_size
        x_patched = x.view(B, num_patches, self.patch_size, F)
        x_flattened = x_patched.view(B, num_patches, -1)
        embedded_patches = self.patch_projection(x_flattened)
        return embedded_patches


# -----------------------------
# Global Embedding
# -----------------------------
@register_module("embedding", "global")
class TimeSeriesGlobalEmbedding(BaseEmbedding):
    def __init__(self, seq_len: int, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.global_projection = nn.Linear(seq_len * feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        if L != self.seq_len:
            raise ValueError(f"Input sequence length ({L}) doesn't match model sequence length ({self.seq_len}).")
        if F != self.feature_size:
            raise ValueError(f"Input feature size ({F}) doesn't match model feature size ({self.feature_size}).")

        x_flattened = x.view(B, -1)
        embedded_global = self.global_projection(x_flattened)
        return embedded_global.unsqueeze(1)

