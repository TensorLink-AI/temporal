"""
Module: temporal_embeddings.py

This module provides a variety of time-series embedding classes and helper functions,
all registered via the `temporal.registry.core` system. Embeddings include value,
positional (sinusoidal, rotary, learned absolute, Shaw relative, Fourier, Time2Vec,
ALiBi, bucketed relative, convolutional, time-delta), patch, global, and flexible
templates, as well as a stacked wrapper. Google-style docstrings are applied
throughout for clarity.
"""
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Corrected imports for registry and builder
from temporal.registry.core import register_module, resolve
from temporal.models.module_builder_helper import ModuleBuilder


# === RoPE Helper Functions ===

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dimensions of the input tensor.

    Splits the last dimension in half and concatenates the second half negated
    before the first half, effectively rotating the vector.

    Args:
        x: Input tensor of shape [..., dim].

    Returns:
        Tensor with rotated halves along the last dimension.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim].
        k: Key tensor with same shape as `q`.
        cos: Cosine embeddings, shape [seq_len, dim] or [batch, 1, seq_len, dim].
        sin: Sine embeddings, matching `cos` shape.
        position_ids: Optional position indices [batch, seq_len].

    Returns:
        Tuple of rotated (query, key) tensors.
    """
    if cos.dim() == 2:
        if position_ids is None:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        else:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------
# Base Embedding Interface
# -----------------------------
class BaseEmbedding(nn.Module):
    """
    Abstract base class for all embedding modules. Subclasses must implement `forward`.
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: Dimensionality of the embedding.
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the embedding. Must be overridden in subclasses.
        """
        raise NotImplementedError("Each embedding must implement its own forward method.")


# -----------------------------
# Value Embeddings
# -----------------------------
@register_module("embedding", "value")
class TimeSeriesValueEmbedding(BaseEmbedding):
    """
    Embed raw time-series feature vectors via linear projection and optional LayerNorm.
    """

    def __init__(
        self,
        feature_size: int,
        d_model: int,
        use_value_norm: bool = False,
    ):
        """
        Args:
            feature_size: Number of input features per time step.
            d_model: Output embedding dimension.
            use_value_norm: Whether to apply LayerNorm after projection.
        """
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)
        self.value_norm = nn.LayerNorm(d_model) if use_value_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, feature_size].

        Returns:
            Embedded tensor of shape [batch, seq_len, d_model].
        """
        proj = self.value_projection(x)
        return self.value_norm(proj) if self.value_norm else proj


@register_module("embedding", "flexible_value")
class FlexibleValueEmbedding(BaseEmbedding):
    """
    Flexible embedding with support for multiple input blocks and optional LayerNorm.
    """

    def __init__(
        self,
        *,
        d_model: int,
        input_dims: Union[int, Sequence[int]],
        proj_builder: Optional[Callable[[int, int, Dict[str, Any]], nn.Module]] = None,
        proj_kwargs: Optional[Dict[str, Any]] = None,
        use_layer_norm: bool = False,
    ):
        """
        Args:
            d_model: Output embedding dimension.
            input_dims: Single int or sequence of ints for each block.
            proj_builder: Factory for projection modules.
            proj_kwargs: Extra kwargs passed to proj_builder.
            use_layer_norm: Apply LayerNorm after summation if True.
        """
        super().__init__(d_model)
        proj_kwargs = proj_kwargs or {}
        if proj_builder is None:
            proj_builder = lambda in_d, out_d, kw: nn.Linear(in_d, out_d, bias=False, **kw)
        if isinstance(input_dims, (list, tuple)):
            self.input_dims = list(input_dims)
            self.projections = nn.ModuleList(
                [proj_builder(in_d, d_model, proj_kwargs) for in_d in self.input_dims]
            )
        else:
            self.input_dims = [input_dims]
            self.projections = proj_builder(input_dims, d_model, proj_kwargs)
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, sum(input_dims)] or [batch, seq_len, input_dims].

        Returns:
            Embedded tensor of shape [batch, seq_len, d_model].
        """
        if isinstance(self.projections, nn.ModuleList):
            blocks = torch.split(x, self.input_dims, dim=-1)
            emb = sum(proj(b) for proj, b in zip(self.projections, blocks))
        else:
            emb = self.projections(x)
        return self.layer_norm(emb) if self.layer_norm else emb


# -----------------------------
# Positional Embeddings
# -----------------------------
@register_module("embedding", "sinusoidal")
class SinusoidalPositionalEmbedding(BaseEmbedding):
    """
    Precomputed sinusoidal positional embeddings without learned parameters.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        """
        Args:
            d_model: Embedding dimension.
            max_seq_len: Maximum supported sequence length.
        """
        super().__init__(d_model)
        self.max_seq_len = max_seq_len
        weights = self._init_weights()
        self.register_buffer("weight", weights)

    def _init_weights(self) -> torch.Tensor:
        """
        Create the sinusoidal encoding table.

        Returns:
            Tensor of shape [max_seq_len, d_model].
        """
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / self.d_model) for j in range(self.d_model)]
            for pos in range(self.max_seq_len)
        ])
        out = torch.zeros(self.max_seq_len, self.d_model)
        half = self.d_model // 2
        out[:, :half] = torch.FloatTensor(np.sin(pos_enc[:, 0::2]))
        out[:, half:] = torch.FloatTensor(np.cos(pos_enc[:, 1::2]))
        return out

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        seq_len: int,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Retrieve embeddings for positions [past_key_values_length:past_key_values_length+seq_len].

        Args:
            batch_size: Batch size.
            seq_len: Number of positions to embed.
            past_key_values_length: Offset into the weight table.

        Returns:
            Tensor of shape [1, seq_len, d_model].

        Raises:
            IndexError: If requested positions exceed max_seq_len.
        """
        start = past_key_values_length
        if seq_len <= 0:
            return torch.empty((batch_size, 0, self.d_model), device=self.weight.device)
        end = start + seq_len
        if end > self.max_seq_len:
            raise IndexError(
                f"Pos {end-1} out of range for max_seq_len {self.max_seq_len}."
            )
        pos = torch.arange(start, end, device=self.weight.device)
        return self.weight[pos].unsqueeze(0)


@register_module("embedding", "rotary")
class RotaryPositionalEmbedding(BaseEmbedding):
    """
    Generate Rotary Positional Embedding cos/sin caches for RoPE.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        """
        Args:
            d_model: Must be even.
            max_seq_len: Cache size.
            base: Base frequency for embedding.
        """
        assert d_model % 2 == 0, "d_model must be even"
        super().__init__(d_model)
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """
        Build cos/sin buffers for up to `seq_len` positions.
        """
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos/sin slices for the requested seq_len.

        Args:
            x: Dummy tensor to infer device/dtype.
            seq_len: Desired length.

        Returns:
            (cos, sin) each shape [seq_len, d_model].
        """
        if seq_len > self.max_seq_len_cached or x.device != self.cos_cached.device:
            self._build_cache(max(seq_len, self.max_seq_len_cached))
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


@register_module("embedding", "learned_abs")
class LearnedAbsolutePositionalEmbedding(BaseEmbedding):
    """
    Learned absolute positional embedding via nn.Embedding.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        """
        Args:
            d_model: Embedding dimension.
            max_seq_len: Maximum positions.
        """
        super().__init__(d_model)
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            seq_len: Number of positions.
            past_key_values_length: Offset.

        Returns:
            Tensor [1, seq_len, d_model].

        Raises:
            IndexError: If positions exceed max_seq_len.
        """
        start = past_key_values_length
        if seq_len <= 0:
            return torch.empty((batch_size, 0, self.d_model), device=self.embedding.weight.device)
        end = start + seq_len
        if end > self.max_seq_len:
            raise IndexError(f"Pos {end-1} out of range for max_seq_len {self.max_seq_len}.")
        pos = torch.arange(start, end, device=self.embedding.weight.device)
        return self.embedding(pos).unsqueeze(0)


@register_module("embedding", "relative_shaw")
class ShawRelativePositionalBias(BaseEmbedding):
    """
    Shaw-relative positional bias for self-attention.
    """

    def __init__(self, num_heads: int, max_distance: int = 128):
        """
        Args:
            num_heads: Number of attention heads.
            max_distance: Clipping distance.
        """
        super().__init__(num_heads)
        self.max_dist = max_distance
        self.bias_table = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Compute relative bias tensor [1, heads, seq_len, seq_len].
        """
        device = self.bias_table.weight.device
        idx = torch.arange(seq_len, device=device)
        dist = idx[None, :] - idx[:, None]
        dist_clamped = dist.clamp(-self.max_dist, self.max_dist) + self.max_dist
        bias = self.bias_table(dist_clamped)
        return bias.permute(2, 0, 1).unsqueeze(0)


@register_module("embedding", "fourier")
class FourierFeatureEmbedding(BaseEmbedding):
    """
    Random Fourier features for positional encoding.
    """

    def __init__(self, d_model: int, num_features: int = 16):
        """
        Args:
            d_model: Output dimension.
            num_features: Number of random frequencies.
        """
        super().__init__(d_model)
        self.num_features = num_features
        self.proj = nn.Linear(2 * num_features, d_model)
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000), num_features)))

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor [1, seq_len, d_model].
        """
        device = self.proj.weight.device
        pos = torch.arange(seq_len, device=device).unsqueeze(-1)
        args = pos * self.freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(feats).unsqueeze(0)


@register_module("embedding", "time2vec")
class Time2VecEmbedding(BaseEmbedding):
    """
    Time2Vec positional embedding.
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: d_model = 1 + periodic dimensions.
        """
        super().__init__(d_model)
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor [1, seq_len, d_model].
        """
        device = next(self.parameters()).device
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        lin = self.linear(t)
        per = torch.sin(self.periodic(t))
        return torch.cat([lin, per], dim=-1).unsqueeze(0)


@register_module("embedding", "alibi")
class ALiBiPositionalBias(BaseEmbedding):
    """
    ALiBi positional bias for attention.
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        """
        Args:
            num_heads: Number of heads.
            max_seq_len: Max sequence length.
        """
        super().__init__(num_heads)
        self.max_seq_len = max_seq_len
        slopes = torch.Tensor(self._get_alibi_slopes(num_heads))
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_alibi_slopes(n: int) -> List[float]:
        """
        Compute ALiBi slopes for each head.
        """
        def get_slopes_pow2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_pow2(n)
        m = 2 ** math.floor(math.log2(n))
        return get_slopes_pow2(m) + get_slopes_pow2(2*m)[::2][: n-m]

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Compute ALiBi bias tensor [1, heads, seq_len, seq_len].
        """
        device = self.slopes.device
        slopes = self.slopes.view(1, -1, 1, 1)
        pos = torch.arange(seq_len, device=device)
        diff = (pos[None, :] - pos[:, None]).abs().view(1, 1, seq_len, seq_len)
        return -diff * slopes


@register_module("embedding", "bucketed")
class BucketedRelativeBias(BaseEmbedding):
    """
    Bucketed relative positional bias.
    """

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        """
        Args:
            num_heads: Number of heads.
            num_buckets: Number of buckets.
            max_distance: Clipping distance.
        """
        super().__init__(num_heads)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_buckets = nn.Embedding(num_buckets, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Compute bucketed bias [1, heads, seq_len, seq_len].
        """
        device = self.relative_buckets.weight.device
        pos = torch.arange(seq_len, device=device)
        diff = pos[None, :] - pos[:, None]
        diff_clamped = diff.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bucket_size = (2*self.max_distance+1) / self.num_buckets
        buckets = (diff_clamped / bucket_size).floor().long().clamp(0, self.num_buckets-1)
        bias = self.relative_buckets(buckets)
        return bias.permute(2,0,1).unsqueeze(0)


@register_module("embedding", "conv_pos")
class ConvolutionalPositionalEmbedding(BaseEmbedding):
    """
    Learnable conv-based refinement of sinusoidal embeddings.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, max_seq_len: int = 2048):
        """
        Args:
            d_model: Embedding dimension.
            kernel_size: Conv1d kernel size.
            max_seq_len: Max seq len.
        """
        super().__init__(d_model)
        base_cls = resolve("embedding", "sinusoidal")
        self.base = base_cls(d_model=d_model, max_seq_len=max_seq_len)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            seq_len: Seq length.
            past_key_values_length: Offset.

        Returns:
            Tensor [1, seq_len, d_model].
        """
        emb = self.base(batch_size=batch_size, seq_len=seq_len, past_key_values_length=past_key_values_length)
        x = emb.permute(0,2,1)
        x = self.conv(x)
        return x.permute(0,2,1)


@register_module("embedding", "timedelta")
class TimeDeltaEmbedding(BaseEmbedding):
    """
    Embedding based on time delta (position) via MLP.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        """
        Args:
            d_model: Output dim.
            hidden_dim: MLP hidden size.
        """
        super().__init__(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor [1, seq_len, d_model].
        """
        device = next(self.mlp.parameters(), torch.zeros(1)).device
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        return self.mlp(t).unsqueeze(0)


@register_module("embedding", "stacked_embedding")
class StackedPositionalEmbedding(BaseEmbedding):
    """
    Combine multiple embedding modules by summation.
    """

    def __init__(self, d_model: int, embedding_configs: List[Dict[str, Any]], builder: ModuleBuilder):
        """
        Args:
            d_model: Embedding dim.
            embedding_configs: List of dicts with keys `type` and optional `args`.
            builder: ModuleBuilder instance for constructing embeddings.
        """
        super().__init__(d_model)
        self.embeddings = nn.ModuleList()
        for cfg in embedding_configs:
            kind = cfg.get("type")
            args = cfg.get("args", {}).copy()
            if not kind:
                raise ValueError("Each embedding config must have a 'type'.")
            module = builder._build(kind="embedding", name=kind, base_kwargs={"hidden_size": d_model}, user_kwargs=args)
            self.embeddings.append(module)
        if not self.embeddings:
            print("Warning: No embeddings configured for StackedPositionalEmbedding.")

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Sum the outputs of all configured embeddings.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            **kwargs: Passed to each embedding.

        Returns:
            Tensor [batch_size, seq_len, d_model].
        """
        if not self.embeddings:
            device = kwargs.get('device', 'cpu')
            return torch.zeros((batch_size, seq_len, self.d_model), device=device)
        # Determine a device
        device = next(self.embeddings[0].parameters(), next(self.embeddings[0].buffers())).device
        combined = torch.zeros((batch_size, seq_len, self.d_model), device=device)
        for mod in self.embeddings:
            emb = mod(batch_size=batch_size, seq_len=seq_len, **kwargs)
            if emb.shape == (batch_size, seq_len, self.d_model) or emb.shape[0] == 1:
                combined = combined + emb
            else:
                print(f"Warning: Skipped embedding {type(mod).__name__} shape {emb.shape}.")
        return combined
