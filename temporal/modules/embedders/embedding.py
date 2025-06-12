
import torch
import torch.nn as nn
import numpy as np
import inspect
import math
from typing import Optional, Tuple, List, Dict, Union, Sequence, Any, Callable

from temporal.registry.core import register_module, resolve
from temporal.models.module_builder_helper import ModuleBuilder

# --- RoPE Helper Functions ---

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    This is a core operation for applying Rotary Positional Embeddings (RoPE).
    It splits the last dimension of the tensor in half, negates the second half,
    and then concatenates them in a swapped order.

    Args:
        x (torch.Tensor): The input tensor, e.g., a query or key tensor of shape
            `[..., seq_len, head_dim]`.

    Returns:
        torch.Tensor: The tensor with the second half of its last dimension rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embeddings (RoPE) to query and key tensors.

    Args:
        q (torch.Tensor): The query tensor of shape `[B, H, T, D_head]`.
        k (torch.Tensor): The key tensor of shape `[B, H, T, D_head]`.
        cos (torch.Tensor): The cosine part of the embeddings, shape `[T, D_head]`.
        sin (torch.Tensor): The sine part of the embeddings, shape `[T, D_head]`.
        position_ids (Optional[torch.Tensor]): Optional position indices to use
            for gathering the embeddings, useful for KV caching.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The transformed query and key tensors.
    """
    if cos.dim() == 2:
        if position_ids is None:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- Base Embedding Interface ---

class BaseEmbedding(nn.Module):
    """Abstract base class for all embedding modules.

    This class provides a common interface for different types of embeddings
    used in the model, ensuring they all have a consistent `d_model` attribute.

    Attributes:
        d_model (int): The dimensionality of the output embedding.
    """
    def __init__(self, d_model: int):
        """Initializes the BaseEmbedding.

        Args:
            d_model (int): The dimensionality of the embedding space.
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        """The forward pass for the embedding. Must be implemented by subclasses."""
        raise NotImplementedError("Each embedding must implement its own forward method.")

# --- Value Embeddings ---

@register_module("embedding", "value")
class TimeSeriesValueEmbedding(BaseEmbedding):
    """Embeds time series features into a dense vector representation.

    This is a standard value embedding that uses a linear projection to map
    input features to the model's hidden dimension. It can optionally apply
    Layer Normalization to the output.

    Attributes:
        value_projection (nn.Linear): The linear layer for projection.
        value_norm (Optional[nn.LayerNorm]): An optional layer normalization.
    """
    def __init__(self, feature_size: int, d_model: int, use_value_norm: bool = False):
        """Initializes the TimeSeriesValueEmbedding.

        Args:
            feature_size (int): The number of input features per time step.
            d_model (int): The target embedding dimension.
            use_value_norm (bool): If True, applies LayerNorm after the projection.
        """
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)
        self.value_norm = nn.LayerNorm(d_model) if use_value_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects the input features into the embedding space.

        Args:
            x (torch.Tensor): The input tensor of shape `[B, T, feature_size]`.

        Returns:
            torch.Tensor: The embedded tensor of shape `[B, T, d_model]`.
        """
        projected_value = self.value_projection(x)
        if self.value_norm is not None:
            return self.value_norm(projected_value)
        return projected_value

@register_module("embedding", "flexible_value")
class FlexibleValueEmbedding(BaseEmbedding):
    """An advanced value embedding for handling multiple feature groups.

    This module can handle inputs that are logically divided into multiple
    blocks of features (e.g., static, known, and observed covariates).
    It applies a separate projection to each block and then sums the results.

    Attributes:
        projections (Union[nn.Module, nn.ModuleList]): The projection module(s).
        input_dims (List[int]): A list of the input dimensions for each block.
        layer_norm (Optional[nn.LayerNorm]): Optional layer normalization.
    """
    def __init__(
        self,
        *,
        d_model: int,
        input_dims: Union[int, Sequence[int]],
        proj_builder: Optional[Callable[[int, int, Dict[str, Any]], nn.Module]] = None,
        proj_kwargs: Optional[Dict[str, Any]] = None,
        use_layer_norm: bool = False
    ):
        """Initializes the FlexibleValueEmbedding.

        Args:
            d_model (int): The output embedding size.
            input_dims (Union[int, Sequence[int]]): The size of the input features.
            proj_builder (Optional[Callable]): A factory function to create the
                projection layer(s). Defaults to `nn.Linear`.
            proj_kwargs (Optional[Dict[str, Any]]): Extra kwargs for the `proj_builder`.
            use_layer_norm (bool): If True, applies LayerNorm to the final embedding.
        """
        super().__init__(d_model)
        proj_kwargs = proj_kwargs or {}
        proj_builder = proj_builder or (lambda in_dim, out_dim, kw: nn.Linear(in_dim, out_dim, bias=False, **kw))

        if isinstance(input_dims, (list, tuple)):
            self.projections = nn.ModuleList(
                [proj_builder(in_dim, d_model, proj_kwargs) for in_dim in input_dims]
            )
            self.input_dims = list(input_dims)
        else:
            self.projections = proj_builder(input_dims, d_model, proj_kwargs)
            self.input_dims = [input_dims]
        
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects one or more feature blocks and sums the results.

        Args:
            x (torch.Tensor): The input tensor. Its last dimension should be the
                sum of `input_dims`.

        Returns:
            torch.Tensor: The final embedded tensor of shape `[B, T, d_model]`.
        """
        if isinstance(self.projections, nn.ModuleList):
            feature_blocks = torch.split(x, self.input_dims, dim=-1)
            embedding_sum = sum(proj(block) for proj, block in zip(self.projections, feature_blocks))
        else:
            embedding_sum = self.projections(x)
        
        if self.layer_norm is not None:
            return self.layer_norm(embedding_sum)
        return embedding_sum

# --- Positional Embeddings ---

@register_module("embedding", "sinusoidal")
class SinusoidalPositionalEmbedding(BaseEmbedding):
    """Creates fixed sinusoidal positional embeddings.

    This embedding uses a pre-computed table of sine and cosine waves of
    different frequencies to represent position. It is not learned.

    Attributes:
        max_seq_len (int): The maximum sequence length for which embeddings
            are pre-computed.
        weight (torch.Tensor): The buffer holding the pre-computed embedding table.
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        """Initializes the SinusoidalPositionalEmbedding."""
        super().__init__(d_model=d_model)
        self.max_seq_len = max_seq_len
        weights = self._init_weights()
        self.register_buffer('weight', weights, persistent=False)

    def _init_weights(self) -> torch.Tensor:
        """Initializes the sinusoidal embedding table."""
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @torch.no_grad()
    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        """Retrieves the positional embeddings for a given sequence range."""
        start_position = past_key_values_length
        end_position = start_position + seq_len
        if end_position > self.max_seq_len:
            raise IndexError(
                f"Requested position index {end_position - 1} is out of bounds for "
                f"SinusoidalPositionalEmbedding with max_seq_len {self.max_seq_len}."
            )
        positions = torch.arange(start_position, end_position, dtype=torch.long, device=self.weight.device)
        return self.weight[positions].unsqueeze(0).expand(batch_size, -1, -1)

@register_module("embedding", "learned_abs")
class LearnedAbsolutePositionalEmbedding(BaseEmbedding):
    """A learnable absolute positional embedding."""
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__(d_model=d_model)
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

@register_module("embedding", "rotary")
class RotaryPositionalEmbedding(BaseEmbedding):
    """Generates Rotary Positional Embedding frequencies (cosine and sine)."""
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        if d_model % 2 != 0:
            raise ValueError(f"Rotary embedding 'd_model' (head dimension) must be even, but got {d_model}")
        super().__init__(d_model=d_model)
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the cached cosine and sine embeddings sliced to the requested length."""
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self._build_cache(seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

# --- Other Embedding Types ---

@register_module("embedding", "patch")
class TimeSeriesPatchEmbedding(BaseEmbedding):
    """Embeds a time series by patching it and projecting the patches."""
    def __init__(self, patch_size: int, feature_size: int, d_model: int, stride: Optional[int] = None, pad_value: float = 0.0):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.projection = nn.Linear(patch_size * feature_size, d_model, bias=False)
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        if L < self.patch_size:
            pad_len = self.patch_size - L
        else:
            pad_len = (self.stride - (L - self.patch_size) % self.stride) % self.stride
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=self.pad_value)
        
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        flattened_patches = patches.contiguous().view(B, -1, patches.shape[-2] * patches.shape[-1])
        return self.projection(flattened_patches)

@register_module("embedding", "stacked_embedding")
class StackedPositionalEmbedding(BaseEmbedding):
    """A container that sums the outputs of multiple positional embedding types."""
    def __init__(self, d_model: int, embedding_configs: List[Dict], builder: ModuleBuilder):
        super().__init__(d_model)
        self.embeddings = nn.ModuleList()
        for config in embedding_configs:
            module = builder._build(
                kind="embedding",
                name=config["type"],
                base_kwargs={"d_model": d_model},
                user_kwargs=config.get("kwargs", {})
            )
            self.embeddings.append(module)
        if not self.embeddings:
             print("Warning: No embeddings configured for StackedPositionalEmbedding.")

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = next(self.parameters(), torch.tensor([])).device
        combined_embedding = torch.zeros((batch_size, seq_len, self.d_model), device=device)
        for embedding_module in self.embeddings:
            # Prepare args for each module individually to handle different signatures
            sig = inspect.signature(embedding_module.forward)
            module_kwargs = {"batch_size": batch_size, "seq_len": seq_len}
            for param_name in sig.parameters:
                if param_name in kwargs:
                    module_kwargs[param_name] = kwargs[param_name]
            
            pos_embed = embedding_module(**module_kwargs)
            combined_embedding += pos_embed
        return combined_embedding
