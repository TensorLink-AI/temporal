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

    def forward(self, x):
        raise NotImplementedError("Each embedding must implement its own forward method.")


# -----------------------------
# Value Embedding
# -----------------------------
@register_module("embedding", "value")
class TimeSeriesValueEmbedding(BaseEmbedding):
    def __init__(self, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)

    def forward(self, x):
        return self.value_projection(x)


# -----------------------------
# Sinusoidal Positional Embedding
# -----------------------------
@register_module("embedding", "positional_sinusoidal")
class TimeSeriesSinusoidalPositionalEmbedding(BaseEmbedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(embedding_dim)
        self.weight = self._init_weight(torch.empty(num_positions, embedding_dim))

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        bsz, seq_len = input_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return self.weight[positions]


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
            raise ValueError(f"Expected feature_size={self.feature_size}, got {F}.")
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length {L} not divisible by patch_size {self.patch_size}.")
