
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
# Sinusoidal Positional Embedding (Flexible Signature)
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
    def forward(self, *args, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Accepts either:
          * forward(batch_size, seq_len, past_key_values_length=0) [New Style]
          * forward(input_shape, past_key_values_length=0)      [Old Style]
            where input_shape is a tuple/Tensor like (batch, seq_len, ...)
        """
        # --------------------------------------------
        # 1) Unpack arguments & Handle potential tensor values robustly
        batch_size: int
        seq_len: int

        if len(args) == 1:
            # Old style: forward(input_shape, past_key_values_length=...)
            input_shape = args[0]
            try:
                bsz_arg = input_shape[0]
                seq_len_arg = input_shape[1]

                # --- Robust conversion to int --- 
                if torch.is_tensor(bsz_arg):
                    if bsz_arg.numel() == 1:
                        batch_size = int(bsz_arg.item())
                    else:
                        print(f"Warning (forward/old): bsz_arg was tensor {bsz_arg.shape}. Using first element.")
                        batch_size = int(bsz_arg[0].item())
                else:
                    batch_size = int(bsz_arg)

                if torch.is_tensor(seq_len_arg):
                    if seq_len_arg.numel() == 1:
                        seq_len = int(seq_len_arg.item())
                    else:
                        print(f"Warning (forward/old): seq_len_arg was tensor {seq_len_arg.shape}. Using first element.")
                        seq_len = int(seq_len_arg[0].item())
                else:
                    seq_len = int(seq_len_arg)
                # --- End robust conversion --- 

            except (TypeError, IndexError) as e:
                 raise TypeError(
                    f"SinusoidalPositionalEmbedding.forward: Could not unpack batch_size and seq_len "
                    f"from single argument input_shape={input_shape}. Error: {e}"
                )

        elif len(args) == 2:
            # New style: forward(batch_size, seq_len, past_key_values_length=...)
            # Assume these are already reasonably convertible to int
            try:
                 batch_size = int(args[0])
                 seq_len    = int(args[1])
            except (TypeError, ValueError) as e:
                 raise TypeError(
                     f"SinusoidalPositionalEmbedding.forward: Could not convert batch_size={args[0]} "
                     f"and seq_len={args[1]} to integers. Error: {e}"
                 )
        else:
            raise TypeError(
                f"SinusoidalPositionalEmbedding.forward expected 1 (input_shape) or 2 (batch_size, seq_len) "
                f"positional arguments, got {len(args)} args={args}"
            )

        # Handle past_key_values_length (robustly, just in case)
        if torch.is_tensor(past_key_values_length):
             if past_key_values_length.numel() == 1:
                 past_len = int(past_key_values_length.item())
             else:
                 print(f"Warning (forward): past_key_values_length was tensor {past_key_values_length.shape}. Using first element.")
                 past_len = int(past_key_values_length[0].item())
        else:
             past_len = int(past_key_values_length)
        # --------------------------------------------

        # 2) Early-exit on empty sequence
        if seq_len <= 0:
            # Use the derived integer batch_size here
            return torch.empty((batch_size, 0, self.dim),
                               device=self.weight.device,
                               dtype=self.weight.dtype)

        # 3) Build position indices
        start = past_len
        end   = past_len + seq_len
        if end > self.max_seq_len:
             # Check against the maximum position index (end - 1)
             max_req_pos = end - 1
             raise IndexError(
                 f"Requested position index {max_req_pos} is out of bounds for "
                 f"SinusoidalPositionalEmbedding with max_seq_len {self.max_seq_len}."
             )

        positions = torch.arange(start, end, dtype=torch.long,
                                 device=self.weight.device)

        # Check bounds ONLY if positions is not empty (redundant given seq_len > 0 check, but safe)
        if positions.numel() > 0:
             # We already checked end > max_seq_len, this check is slightly redundant
             # max_pos = positions.max()
             # if max_pos >= self.max_seq_len:
             #     raise IndexError(...) # Should not happen if end check is correct
             pass
        else:
             # This should not happen if seq_len > 0
             print(f"Warning: Position tensor empty after arange(start={start}, end={end}). Should not happen if seq_len > 0.")
             return torch.empty((batch_size, 0, self.dim), device=self.weight.device, dtype=self.weight.dtype)

        # 4) Gather + unsqueeze for batch broadcast
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

