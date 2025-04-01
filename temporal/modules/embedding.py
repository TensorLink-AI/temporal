import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

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
        #print("x.shape:", x.shape)

        return self.value_projection(x)

    

    # FeatureEmbedder tht can be call in encoders, decoders etc
    # TimeSeriesEmbedding that can be call in that same vein
