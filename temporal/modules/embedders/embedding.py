import torch
import torch.nn as nn
import numpy as np
import inspect
import math  # Import math for log calculation in BucketedRelativeBias or ALiBi
from typing import Optional, Tuple, List, Dict, Union, Sequence, Any, Callable

# === Corrected Imports ===
from temporal.registry.core import register_module, resolve  # resolve is in core
from temporal.models.module_builder_helper import ModuleBuilder
from torch.fft import rfft, irfft
import pywt
"""
Module providing a variety of time-series embedding classes and helper functions.
All embeddings are registered via the temporal.registry.core system.
Embeddings include value, flexible value, positional (sinusoidal, rotary, learned absolute,
Shaw relative, Fourier, Time2Vec, ALiBi, bucketed relative, convolutional, time-delta),
patch, global, and a stacked wrapper.
"""

import torch
import torch.nn as nn
import numpy as np
import inspect
import math
from typing import Optional, Tuple, List, Dict, Union, Sequence, Any, Callable

from temporal.registry.core import register_module, resolve
from torch.fft import rfft, irfft
import pywt
"""
Module providing a variety of time-series embedding classes and helper functions.
"""

# --- RoPE Helper Functions ---
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        if position_ids is None:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        else:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Base Embedding
class BaseEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each embedding must implement its own forward method.")

# Value Embedding
@register_module("embedding", "value")
class TimeSeriesValueEmbedding(BaseEmbedding):
    def __init__(self, feature_size: int, d_model: int, use_value_norm: bool = False):
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)
        self.value_norm = nn.LayerNorm(d_model) if use_value_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.value_projection(x)
        if self.value_norm is not None:
            return self.value_norm(proj)
        return proj

# Flexible Value Embedding (remains the same)
@register_module("embedding", "flexible_value")
class FlexibleValueEmbedding(BaseEmbedding):
    def __init__(self, *, d_model: int, input_dims: Union[int, Sequence[int]], proj_builder: Callable[[int, int, Dict[str, Any]], nn.Module] = None, proj_kwargs: Dict[str, Any] = None, use_layer_norm: bool = False):
        super().__init__(d_model)
        proj_kwargs = proj_kwargs or {}
        if proj_builder is None:
            proj_builder = lambda in_dim, out_dim, kw: nn.Linear(in_dim, out_dim, bias=False, **kw)
        if isinstance(input_dims, (list, tuple)):
            self.projections = nn.ModuleList([proj_builder(in_dim, d_model, proj_kwargs) for in_dim in input_dims])
            self.input_dims = list(input_dims)
        else:
            self.projections = proj_builder(input_dims, d_model, proj_kwargs)
            self.input_dims = [input_dims]
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.projections, nn.ModuleList):
            blocks = torch.split(x, self.input_dims, dim=-1)
            emb = sum(proj(b) for proj, b in zip(self.projections, blocks))
        else:
            emb = self.projections(x)
        if self.layer_norm is not None:
            return self.layer_norm(emb)
        return emb

# Sinusoidal Positional Embedding
@register_module("embedding", "sinusoidal")
class SinusoidalPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__(d_model)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        Args:
            x: Input tensor of shape [B, L, D].
        Returns:
            Tensor with positional encodings added.
        """
        # The positional encoding is sliced up to the sequence length of the input
        # and added to the input tensor `x`.
        x = x + self.pe[:x.size(1), :]
        return x

# Patch Embedding
@register_module("embedding", "patch")
class TimeSeriesPatchEmbedding(BaseEmbedding):
    def __init__(self, patch_size: int, feature_size: int, d_model: int, stride: Optional[int] = None, pad_value: float = 0.0, use_mlp: bool = False, mlp_hidden_size: Optional[int] = None, output_patch_size: Optional[int] = None):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.pad_value = pad_value
        self.flat_size = patch_size * feature_size
        self.output_patch_size = output_patch_size if output_patch_size is not None else patch_size
        
        if use_mlp:
            H = mlp_hidden_size or d_model
            self.proj = nn.Sequential(
                nn.Linear(self.flat_size, H),
                nn.ReLU(),
                nn.Linear(H, d_model),
            )
        else:
            self.proj = nn.Linear(self.flat_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correctly pads, unfolds, and projects the input tensor into patches.
        Args:
            x: Input tensor of shape [B, L, F].
        Returns:
            Tensor of shape [B, num_patches, d_model].
        """
        B, L, F = x.shape
        
        # 1. Pad the sequence length if necessary
        pad_len = (self.stride - (L - self.patch_size) % self.stride) % self.stride
        if pad_len > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_len), "constant", self.pad_value)
        
        # 2. Unfold to create patches: [B, F, num_patches, patch_size]
        patches = x.permute(0, 2, 1).unfold(dimension=2, size=self.patch_size, step=self.stride)
        
        # 3. Flatten patch and feature dimensions: [B, num_patches, F * patch_size]
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, -1, self.flat_size)
        
        # 4. Project to d_model
        return self.proj(patches)

# Rotary Positional Embedding
@register_module("embedding", "rotary")
class RotaryPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        assert d_model % 2 == 0, "Rotary embedding dim must be even"
        super().__init__(d_model)
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Infers seq_len from input and returns the cos/sin caches.
        Args:
            x: A dummy tensor of shape [B, L, D] to infer seq_len and device.
        Returns:
            Tuple of (cos, sin), each of shape [seq_len, d_model].
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._build_cache(max(seq_len, self.max_seq_len_cached))
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        return (x * cos) + (rotate_half(x) * sin)
        
# --- Other Embedding Implementations (Unchanged) ---
# Global Embedding
@register_module("embedding", "global")
class TimeSeriesGlobalEmbedding(BaseEmbedding):
    def __init__(self, seq_len: int, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.global_projection = nn.Linear(seq_len * feature_size, d_model, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        if L != self.seq_len or F != self.feature_size:
            raise ValueError(f"Input shape mismatch: expected (L={self.seq_len}, F={self.feature_size}), got (L={L}, F={F})")
        flat = x.view(B, -1)
        out = self.global_projection(flat)
        return out.unsqueeze(1)

# Learned Absolute Positional Embedding
@register_module("embedding", "learned_abs")
class LearnedAbsolutePositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__(d_model)
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)
    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        start = past_key_values_length
        end = start + seq_len
        if end > self.max_seq_len:
            raise IndexError(f"Position index {end-1} out of bounds for max_seq_len {self.max_seq_len}.")
        positions = torch.arange(start, end, device=self.embedding.weight.device)
        embeds = self.embedding(positions)
        return embeds.unsqueeze(0).expand(batch_size, -1, -1)
# ... [rest of the file remains the same] ...

# -----------------------------
# Shaw Relative Positional Bias
# -----------------------------
@register_module("embedding", "relative_shaw")
class ShawRelativePositionalBias(BaseEmbedding):
    """
    Learnable relative positional biases as in Shaw et al.
    """
    def __init__(self, num_heads: int, max_distance: int = 128):
        """
        Args:
            num_heads: Number of attention heads.
            max_distance: Maximum relative distance.
        """
        super().__init__(d_model=num_heads)
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Compute relative bias tensor for attention scores.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor of shape [1, num_heads, seq_len, seq_len].
        """
        device = self.relative_bias.weight.device
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None]
        clipped = torch.clamp(diff, -self.max_distance, self.max_distance) + self.max_distance
        biases = self.relative_bias(clipped)
        return biases.permute(2, 0, 1).unsqueeze(0)

# -----------------------------
# Fourier Feature Embedding
# -----------------------------
@register_module("embedding", "fourier")
class FourierFeatureEmbedding(BaseEmbedding):
    """
    Embed positions using random Fourier features.
    """
    def __init__(self, d_model: int, num_features: int = 16):
        """
        Args:
            d_model: Output embedding dimension.
            num_features: Number of Fourier features.
        """
        super().__init__(d_model)
        self.num_features = num_features
        self.proj = nn.Linear(2 * num_features, d_model)
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0, np.log(1000), num_features))
        )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Embed positions into Fourier feature space.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor of shape [B, seq_len, d_model].
        """
        device = self.proj.weight.device
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        args = positions * self.freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        embeds = self.proj(feats)
        return embeds.unsqueeze(0).expand(batch_size, -1, -1)

# -----------------------------
# Time2Vec Embedding
# -----------------------------

@register_module("embedding", "time2vec")
class Time2VecEmbedding(BaseEmbedding):
    """
    Time2Vec positional embedding: 1 linear + sinusoidal components.
    If use_cos=True, uses (d_model - 1) // 2 sin/cos pairs (RoPE-compatible).
    If use_cos=False, uses (d_model - 1) sin components (original Time2Vec).
    """
    def __init__(self, d_model: int, use_cos: bool = True):
        """
        Args:
            d_model: Output embedding dimension (must be >= 2).
            use_cos: If True, use sin+cos pairs (RoPE-aligned). If False, sin only.
        """
        super().__init__()
        assert d_model >= 2, "d_model must be >= 2"

        self.use_cos = use_cos
        self.linear = nn.Linear(1, 1)

        if use_cos:
            self.num_freqs = (d_model - 1) // 2
            out_dim = 2 * self.num_freqs
        else:
            self.num_freqs = d_model - 1
            out_dim = self.num_freqs

        self.periodic = nn.Linear(1, self.num_freqs)
        self.d_model = 1 + out_dim  # 1 linear + sin/cos or sin only

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = next(self.parameters()).device
        t = torch.arange(seq_len, device=device, dtype=torch.float32).view(-1, 1)  # [T, 1]

        lin_part = self.linear(t)  # [T, 1]
        freq_proj = self.periodic(t)  # [T, num_freqs]

        if self.use_cos:
            sin_part = torch.sin(freq_proj)
            cos_part = torch.cos(freq_proj)
            per_part = torch.cat([sin_part, cos_part], dim=-1)  # [T, 2 * num_freqs]
        else:
            per_part = torch.sin(freq_proj)  # [T, d_model - 1]

        embeddings = torch.cat([lin_part, per_part], dim=-1)  # [T, d_model]
        embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, d_model]
        return embeddings



# -----------------------------
# ALiBi Positional Bias
# -----------------------------
@register_module("embedding", "alibi")
class ALiBiPositionalBias(BaseEmbedding):
    """
    Attention with linear biases (ALiBi) instead of positional embeddings.
    """
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 2048
    ):
        """
        Args:
            num_heads: Number of attention heads.
            max_seq_len: Max sequence length for slope calculation.
        """
        super().__init__(d_model=num_heads)
        slopes = torch.Tensor(self._get_alibi_slopes(num_heads))
        self.register_buffer("slopes", slopes)
        self.max_seq_len = max_seq_len

    @staticmethod
    def _get_alibi_slopes(n: int) -> List[float]:
        """
        Compute ALiBi slopes for heads.

        Args:
            n: Number of heads.

        Returns:
            List of slopes of length n.
        """
        def p2(v):
            start = 2 ** (-(2 ** -(math.log2(v) - 3)))
            ratio = start
            return [start * ratio**i for i in range(v)]

        if n & (n - 1) == 0:
            return p2(n)
        m = 2 ** math.floor(math.log2(n))
        return p2(m) + p2(2 * m)[0::2][: n - m]

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Generate ALiBi bias tensor for causal attention.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor [1, num_heads, seq_len, seq_len].
        """
        device = self.slopes.device
        slopes = self.slopes[None, :, None, None]
        pos = torch.arange(seq_len, device=device)
        diff = pos[None, :] - pos[:, None]
        abs_diff = diff.abs().unsqueeze(0).unsqueeze(0)
        bias = -abs_diff * slopes
        return bias

# -----------------------------
# Bucketed Relative Bias
# -----------------------------
@register_module("embedding", "bucketed")
class BucketedRelativeBias(BaseEmbedding):
    """
    Bucketed relative positional biases.
    """
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128
    ):
        """
        Args:
            num_heads: Number of attention heads.
            num_buckets: Number of buckets.
            max_distance: Max distance to represent.
        """
        super().__init__(d_model=num_heads)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_buckets = nn.Embedding(num_buckets, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Compute bucketed relative bias.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor [1, num_heads, seq_len, seq_len].
        """
        device = self.relative_buckets.weight.device
        pos = torch.arange(seq_len, device=device)
        diff = pos[None, :] - pos[:, None]
        clipped = torch.clamp(diff, -self.max_distance, self.max_distance) + self.max_distance
        bucket_size = (2 * self.max_distance + 1) / self.num_buckets
        buckets = (clipped.float() / bucket_size).floor().long().clamp(0, self.num_buckets - 1)
        biases = self.relative_buckets(buckets)
        return biases.permute(2, 0, 1).unsqueeze(0)

# -----------------------------
# Convolutional Positional Embedding
# -----------------------------
@register_module("embedding", "conv_pos")
class ConvolutionalPositionalEmbedding(BaseEmbedding):
    """
    Convolutional enhancement of sinusoidal embeddings.
    """
    def __init__(self, d_model: int, kernel_size: int = 3, max_seq_len: int = 2048):
        """
        Args:
            d_model: Embedding dimension.
            kernel_size: Conv1d kernel size.
            max_seq_len: Max sequence length for base emb.
        """
        super().__init__(d_model)
        base_cls = resolve("embedding", "sinusoidal")
        self.base = base_cls(d_model=d_model, max_seq_len=max_seq_len)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Apply conv to base positional embeddings.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            past_key_values_length: Offset index.

        Returns:
            Tensor [B, seq_len, d_model].
        """
        emb = self.base(batch_size, seq_len, past_key_values_length)
        x = emb.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

# -----------------------------
# TimeDelta Embedding
# -----------------------------
@register_module("embedding", "timedelta")
class TimeDeltaEmbedding(BaseEmbedding):
    """
    Embedding for time delta features via an MLP.
    """
    def __init__(self, d_model: int, hidden_dim: int = 64):
        """
        Args:
            d_model: Output dimension.
            hidden_dim: Hidden layer size.
        """
        super().__init__(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Embed relative time deltas.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor [B, seq_len, d_model].
        """
        device = next(self.mlp.parameters()).device if list(self.mlp.parameters()) else 'cpu'
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        feats = self.mlp(t)
        return feats.unsqueeze(0).expand(batch_size, -1, -1)

# -----------------------------
# Stacked Positional Embedding
# -----------------------------

# -----------------------------
# Stacked Positional Embedding
# -----------------------------
@register_module("embedding", "stacked_embedding")
class StackedPositionalEmbedding(BaseEmbedding):
    """
    Wrapper to stack multiple positional embeddings sequentially by summing them.
    This wrapper intelligently passes arguments to its sub-modules, preventing
    errors when sub-modules have different forward signatures.
    """
    def __init__(
        self,
        d_model: int,
        embedding_configs: List[Dict[str, Any]],
        builder: ModuleBuilder
    ):
        """
        Args:
            d_model: Embedding dimension.
            embedding_configs: List of configs for sub-embeddings.
            builder: ModuleBuilder to instantiate embeddings.
        """
        super().__init__(d_model)
        self.embeddings = nn.ModuleList()
        self._forward_param_names = []  # Store parameter names for each forward method

        for config in embedding_configs:
            embed_type = config.get("type")
            args = config.get("args", {}).copy()
            if not embed_type:
                raise ValueError("Each embedding config must have a 'type'.")

            # Ensure d_model is passed correctly to sub-embeddings if they need it
            if 'd_model' not in args:
                 args['d_model'] = d_model
            
            # Build the module using the provided builder
            module = builder._build(
                kind="embedding",
                name=embed_type,
                user_kwargs=args
            )
            self.embeddings.append(module)
            
            # Inspect and store the parameter names of the module's forward method
            self._forward_param_names.append(
                inspect.signature(module.forward).parameters.keys()
            )

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Sum outputs of configured embeddings, intelligently passing only supported arguments.

        Args:
            **kwargs: A dictionary of arguments that might be needed by any of the
                      sub-embeddings. This can include `batch_size`, `seq_len`,
                      `past_key_values_length`, `x` (for RoPE), `device`, etc.

        Returns:
            Tensor of shape [B, seq_len, d_model] representing the summed embeddings.
        """
        batch_size = kwargs.get("batch_size")
        seq_len = kwargs.get("seq_len")
        if batch_size is None or seq_len is None:
            raise ValueError("The `forward` method of StackedPositionalEmbedding requires "
                             "`batch_size` and `seq_len` to be passed as keyword arguments.")
            
        # Determine the device from parameters or kwargs
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = kwargs.get('device', 'cpu')

        # Initialize the combined tensor
        combined_embedding = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Iterate through each sub-embedding
        for i, module in enumerate(self.embeddings):
            # Get the supported parameter names for the current module
            supported_params = self._forward_param_names[i]
            
            # Filter the provided kwargs to only include parameters supported by the module
            sub_kwargs = {
                key: value for key, value in kwargs.items() if key in supported_params
            }
            
            # Call the sub-embedding with only the arguments it can accept
            output = module(**sub_kwargs)
            
            # Add the output to the combined embedding if its shape is correct.
            # This handles both standard embeddings and ignores attention biases.
            if isinstance(output, torch.Tensor) and output.shape[-2:] == (seq_len, self.d_model):
                combined_embedding += output

        return combined_embedding


@register_module("embedding", "none")
class NoneEmbedding(BaseEmbedding):
    """
    A placeholder embedding that returns a zero tensor on the correct device.
    This effectively disables the positional embedding when used in a model
    configuration.
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__(d_model)
        # Register a dummy buffer to make the module device-aware.
        self.register_buffer("dummy_buffer", torch.zeros(1), persistent=True)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Returns a zero tensor of the correct shape and on the correct device.

        Args:
            batch_size: The batch size of the input.
            seq_len: The sequence length of the input.
            **kwargs: Additional arguments (ignored).

        Returns:
            A zero tensor of shape [batch_size, seq_len, d_model] on the
            correct device.
        """
        # Use the device of the dummy buffer to create the tensor.
        return torch.zeros(
            batch_size,
            seq_len,
            self.d_model,
            device=self.dummy_buffer.device
        )





@register_module("embedding", "s4")
class S4PositionalEmbedding(BaseEmbedding):
    """
    S4-inspired positional embedding using a learned long convolution kernel.
    Projects identity input over time (like a learned filter bank) and returns
    a [B, T, d_model] positional signal.
    """

    def __init__(self, d_model: int, kernel_size: int = 512, max_seq_len: int = 4096):
        """
        Args:
            d_model: Output embedding dimension.
            kernel_size: Length of the learned filter.
            max_seq_len: Maximum supported sequence length.
        """
        super().__init__(d_model)
        self.kernel_size = kernel_size
        self.max_seq_len = max_seq_len

        self.filter = nn.Parameter(torch.randn(d_model, kernel_size) * 0.01)
        self.skip = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate positional embeddings.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tensor of shape [B, T, d_model].
        """
        device = self.filter.device
        T = seq_len

        # Identity impulse sequence for convolution: [d_model, T]
        x = torch.eye(T, device=device).unsqueeze(0).repeat(self.d_model, 1, 1)

        # FFT convolution with learned kernel
        x_fft = rfft(x, n=2 * T)                              # [d_model, T, Freq]
        k_fft = rfft(self.filter, n=2 * T)                   # [d_model, Freq]
        y_fft = x_fft * k_fft.unsqueeze(1)                   # broadcast multiply
        y = irfft(y_fft, n=2 * T)[..., :T]                   # [d_model, T, T]

        # Extract diagonals → position encodings [T, d_model]
        pos = torch.diagonal(y, dim1=1, dim2=2).transpose(0, 1)  # [T, d_model]

        # Add learnable skip bias
        pos = pos + self.skip[0, :T]

        return pos.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, d_model]



@register_module("embedding", "wavelet")
class WaveletPositionalEmbedding(BaseEmbedding):
    """
    Wavelet-based positional embedding that projects identity-position impulses
    through a learned multi-resolution filter bank (DWT basis).

    This is a fixed-time-index embedding (like sinusoidal) but with multi-scale
    locality and better modeling of transients.

    Args:
        d_model (int): Output embedding dimension.
        wavelet (str): Wavelet type (e.g., 'db1', 'haar', 'coif1').
        level (int): Decomposition level (e.g., 3).
        max_seq_len (int): Maximum supported sequence length.
    """
    def __init__(
        self,
        d_model: int,
        wavelet: str = "db4",
        level: int = 3,
        max_seq_len: int = 2048,
        **kwargs
    ):
        super().__init__(d_model)
        self.wavelet = wavelet
        self.level = level
        self.max_seq_len = max_seq_len

        # Precompute embedding matrix
        basis = torch.eye(max_seq_len)
        coeffs = []
        for i in range(max_seq_len):
            signal = basis[i].numpy()
            c = pywt.wavedec(signal, wavelet=self.wavelet, level=self.level, mode="periodization")
            coeff = np.concatenate(c)
            coeffs.append(coeff)
        coeffs = np.stack(coeffs)  # [T, total_wavelet_dim]
        self.input_dim = coeffs.shape[1]

        # Projection to d_model
        self.proj = nn.Linear(self.input_dim, d_model)
        self.register_buffer("basis_embed", torch.tensor(coeffs, dtype=torch.float32), persistent=False)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            seq_len: Current sequence length.

        Returns:
            Tensor of shape [B, T, d_model]
        """
        pos_embed = self.basis_embed[:seq_len]  # [T, input_dim]
        embed = self.proj(pos_embed)  # [T, d_model]
        return embed.unsqueeze(0).expand(batch_size, -1, -1)