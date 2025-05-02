
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
# Positional Embedding
# -----------------------------
@register_module("embedding", "positional")
class PositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

# NOTE: The following SinusoidalPositionalEmbedding seems designed for specific model
# architectures (like transformers expecting specific input shapes and handling)
# and might need adjustments based on how it's integrated.
@register_module("embedding", "sinusoidal")
class SinusoidalPositionalEmbedding(BaseEmbedding):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__(d_model=dim) # Assuming d_model is equivalent to dim here
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.weight = nn.Parameter(self._init_weights())

    def _init_weights(self) -> torch.Tensor:
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / self.dim) for j in range(self.dim)]
                for pos in range(self.max_seq_len)
            ]
        )
        out = torch.zeros(self.max_seq_len, self.dim)
        # Apply sin to even indices and cos to odd indices
        sentinel = self.dim // 2 if self.dim % 2 == 0 else (self.dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2])) # Corrected indexing for cosine part
        # out.detach_() # .detach_() is in-place and returns the same tensor detached.
                       # We want to return the tensor itself.
        return out

    # Note: This forward method seems designed for HF Transformers style usage,
    # taking input_shape and past_key_values_length.
    # It might need adaptation depending on your specific model structure.
    @torch.no_grad()
    def forward(self, input_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        bsz, seq_len = input_shape[:2]

        # make sure both are Python ints for torch.arange
        seq_len = int(seq_len)                             # new
        start   = int(past_key_values_length)              # keep explicit
        end     = start + seq_len

        positions = torch.arange(
            start,
            end,
            dtype=torch.long,
            device=self.weight.device,
        )

        # Ensure positions do not exceed max_seq_len
        if positions.max() >= self.max_seq_len:
             raise IndexError(
                 f"Requested position index {positions.max()} is out of bounds for " +
                 f"SinusoidalPositionalEmbedding with max_seq_len {self.max_seq_len}."
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
        # Project the flattened patch to the embedding dimension
        self.patch_projection = nn.Linear(patch_size * feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F)
                              B = batch size, L = sequence length, F = feature size
        Returns:
            torch.Tensor: Embedded tensor of shape (B, num_patches, d_model)
        """
        B, L, F = x.shape
        if F != self.feature_size:
            raise ValueError(f"Input feature size ({F}) doesn't match model feature size ({self.feature_size}).")
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length ({L}) is not divisible by patch size ({self.patch_size}).")

        num_patches = L // self.patch_size
        # Reshape: (B, L, F) -> (B, num_patches, patch_size, F)
        x_patched = x.view(B, num_patches, self.patch_size, F)
        # Flatten patches: (B, num_patches, patch_size, F) -> (B, num_patches, patch_size * F)
        x_flattened = x_patched.view(B, num_patches, -1)
        # Project patches: (B, num_patches, patch_size * F) -> (B, num_patches, d_model)
        embedded_patches = self.patch_projection(x_flattened)
        return embedded_patches


# -----------------------------
# Global Embedding
# -----------------------------
@register_module("embedding", "global")
class TimeSeriesGlobalEmbedding(BaseEmbedding):
    def __init__(self, seq_len: int, feature_size: int, d_model: int):
        """
        Embeds the entire time series into a single vector.
        This example implements a simple flattening and projection.
        """
        super().__init__(d_model)
        self.seq_len = seq_len
        self.feature_size = feature_size
        # Project the flattened sequence * features to the embedding dimension
        self.global_projection = nn.Linear(seq_len * feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F)
                              B = batch size, L = sequence length, F = feature size
        Returns:
            torch.Tensor: Embedded tensor of shape (B, 1, d_model)
                          (The sequence dimension is reduced to 1 for the global embedding)
        """
        B, L, F = x.shape
        if L != self.seq_len:
            raise ValueError(f"Input sequence length ({L}) doesn't match model sequence length ({self.seq_len}).")
        if F != self.feature_size:
            raise ValueError(f"Input feature size ({F}) doesn't match model feature size ({self.feature_size}).")

        # Flatten the sequence and feature dimensions: (B, L, F) -> (B, L * F)
        x_flattened = x.view(B, -1)
        # Project to embedding dimension: (B, L * F) -> (B, d_model)
        embedded_global = self.global_projection(x_flattened)
        # Add a sequence dimension for consistency: (B, d_model) -> (B, 1, d_model)
        return embedded_global.unsqueeze(1)

