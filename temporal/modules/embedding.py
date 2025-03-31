import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


EMBEDDING_REGISTRY = {
    "value": TimeSeriesValueEmbedding,
    "positional_sinusoidal": TimeSeriesSinusoidalPositionalEmbedding,
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
        return self.value_projection(x)

    

class AutoTimeSeriesEmbedding:
    @staticmethod
    def from_config(
        config: BaseTimeSeriesConfig,
        embedding_type: str = "value",
        **kwargs
    ):
        embedding_cls = EMBEDDING_REGISTRY.get(embedding_type)
        if embedding_cls is None:
            raise ValueError(f"Unknown embedding type '{embedding_type}'. Available: {list(EMBEDDING_REGISTRY.keys())}")

        # Filter kwargs by class signature
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

        return embedding_cls(**filtered_kwargs)
