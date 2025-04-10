import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


EMBEDDING_REGISTRY = {
    "value": TimeSeriesValueEmbedding,
    "positional_sinusoidal": TimeSeriesSinusoidalPositionalEmbedding,
    "patch" : TimeSeriesPatchEmbedding ,
    "point"TimeSeriesPointEmbedding
    # Add others like "learned_positional", "categorical", etc.
}


class BaseEmbedding(nn.Module):
    """Base class for embedding layers to ensure generalization across different types."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        raise NotImplementedError("Each embedding class must implement the forward method.")

class TimeSeriesSinusoidalPositionalEmbedding(BaseEmbedding):
    """Sinusoidal positional encoding, generalized for any time series length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(embedding_dim)
        self.weight = self._init_weight(torch.empty(num_positions, embedding_dim))

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid error in PyTorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """input_shape is expected to be [batch_size, seq_len]."""
        bsz, seq_len = input_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return self.weight[positions]

class TimeSeriesValueEmbedding(BaseEmbedding):
    """Feature embedding layer for time series values."""

    def __init__(self, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)

    def forward(self, x):
        print("x.shape:", x.shape)

        return self.value_projection(x)

class TimeSeriesPatchEmbedding(BaseEmbedding):
    """
    Splits the time dimension into patches of size `patch_size`,
    then projects each patch into `d_model`.
    Assumes input shape (B, L, feature_size).
    """
    def __init__(self, patch_size: int, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.feature_size = feature_size
        
        # Each patch has (patch_size * feature_size) values
        self.patch_projection = nn.Linear(patch_size * feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, feature_size)
        Returns: (B, num_patches, d_model) where num_patches = L // patch_size
        """
        # Check that L is multiple of patch_size
        B, L, F = x.shape
        if F != self.feature_size:
            raise ValueError(f"Expected feature_size={self.feature_size}, got input's last dim = {F}.")

        if (L % self.patch_size) != 0:
            raise ValueError(f"Sequence length {L} not divisible by patch_size {self.patch_size}.")

        num_patches = L // self.patch_size
        # Reshape: each patch is contiguous chunk of length patch_size in the time dimension
        # shape => (B, num_patches, patch_size, feature_size)
        x = x.reshape(B, num_patches, self.patch_size, self.feature_size)

        # Flatten patch dimension => (patch_size * feature_size)
        # shape => (B, num_patches, patch_size * feature_size)
        x = x.view(B, num_patches, self.patch_size * self.feature_size)

        # Project each patch to d_model
        # => (B, num_patches, d_model)
        x = self.patch_projection(x)

        return x

# -------------------------------------------------------------------
# 2. Point Embedding
# -------------------------------------------------------------------
class TimeSeriesPointEmbedding(BaseEmbedding):
    """
    Treat the entire sequence (or sub-chunk) as one "point" to project to d_model.
    If your input has shape (B, L, feature_size) and you want a single embedding
    per sample, you can flatten L*feature_size => input_len, then project to d_model.
    """
    def __init__(self, input_len: int, d_model: int):
        super().__init__(d_model)
        self.input_len = input_len
        self.point_proj = nn.Linear(input_len, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, feature_size), flatten to (B, L*feature_size) => input_len
        => project => (B, 1, d_model).
        If x is exactly (B, input_len), that's also valid (then we treat the last dimension as 1).
        """
        if x.dim() == 2:
            # (B, L) => L must match input_len
            B, L = x.shape
            if L != self.input_len:
                raise ValueError(f"Expected input_len={self.input_len}, got L={L}")
            # project => (B, d_model), then unsqueeze => (B,1,d_model)
            out = self.point_proj(x).unsqueeze(1)
            return out

        elif x.dim() == 3:
            B, L, F = x.shape
            total_len = L * F
            if total_len != self.input_len:
                raise ValueError(
                    f"Expected L*F={total_len} to match input_len={self.input_len}, got mismatch."
                )
            # Flatten => (B, L*F)
            x = x.view(B, total_len)
            out = self.point_proj(x).unsqueeze(1)  # => (B,1,d_model)
            return out

        else:
            raise ValueError("TimeSeriesPointEmbedding expects 2D or 3D input (B,L) or (B,L,F).")

EMBEDDING_REGISTRY = {
    "value": TimeSeriesValueEmbedding,
    "positional_sinusoidal": TimeSeriesSinusoidalPositionalEmbedding,
    "patch": TimeSeriesPatchEmbedding,   # NEW
    "point": TimeSeriesPointEmbedding,   # NEW
    # You can add other custom embeddings as needed
}

# -------------------------------------------------------------------
# 5. AutoTimeSeriesEmbedding remains the same, but can now create "patch" or "point"
# -------------------------------------------------------------------
class AutoTimeSeriesEmbedding:
    @staticmethod
    def from_config(
        config,  # -> must define .hidden_size, .context_length, .prediction_length, .feature_size, etc.
        embedding_type: str = "value",
        **kwargs
    ):
        embedding_cls = EMBEDDING_REGISTRY.get(embedding_type)
        if embedding_cls is None:
            raise ValueError(
                f"Unknown embedding type '{embedding_type}'. "
                f"Available: {list(EMBEDDING_REGISTRY.keys())}"
            )

        # Introspect the constructor to only pass relevant args
        import inspect
        sig = inspect.signature(embedding_cls.__init__)
        accepted_keys = set(sig.parameters.keys()) - {"self"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}

        # Add common values from config if not explicitly passed
        if "d_model" in accepted_keys and "d_model" not in filtered_kwargs:
            filtered_kwargs["d_model"] = config.hidden_size
        if "embedding_dim" in accepted_keys and "embedding_dim" not in filtered_kwargs:
            filtered_kwargs["embedding_dim"] = config.hidden_size
        if "feature_size" in accepted_keys and "feature_size" not in filtered_kwargs:
            filtered_kwargs["feature_size"] = getattr(config, "feature_size", None)
        if "num_positions" in accepted_keys and "num_positions" not in filtered_kwargs:
            filtered_kwargs["num_positions"] = config.context_length + config.prediction_length
        if "patch_size" in accepted_keys and "patch_size" not in filtered_kwargs:
            filtered_kwargs["patch_size"] = getattr(config, "patch_size", 1)
        if "input_len" in accepted_keys and "input_len" not in filtered_kwargs:
            filtered_kwargs["input_len"] = getattr(config, "input_len", None)

        return embedding_cls(**filtered_kwargs)

