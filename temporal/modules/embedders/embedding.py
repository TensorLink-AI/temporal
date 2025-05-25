
import torch
import torch.nn as nn
import numpy as np
import inspect
import math # Import math for log calculation in BucketedRelativeBias if needed, or ALiBi later
from typing import Optional, Tuple, List, Dict, Union, Sequence, Any, Callable

# === Corrected Imports ===
from temporal.registry.core import register_module, resolve # resolve is in core
# Need ModuleBuilder for type hint and its _build method
from temporal.models.module_builder_helper import ModuleBuilder


# --- RoPE Helper Functions ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Positional Embedding to query and key tensors."""
    # cos, sin: [seq_len, dim] or [bsz, 1, seq_len, dim]
    # q, k: [bsz, num_heads, seq_len, head_dim]

    if cos.dim() == 2: # [seq_len, dim] -> need to gather based on position_ids if provided
        if position_ids is None:
            # Assuming standard range if position_ids not given
             cos = cos[None, None, :, :] # -> [1, 1, seq_len, dim]
             sin = sin[None, None, :, :] # -> [1, 1, seq_len, dim]
        else:
            # Gather based on position IDs: [bsz, seq_len] -> [bsz, seq_len, dim]
            # Ensure indices are within the cached length
            cos = cos[position_ids].unsqueeze(1) # -> [bsz, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1) # -> [bsz, 1, seq_len, dim]
    # else: assume cos/sin already have correct shape e.g. [bsz, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
# --- End RoPE Helper Functions ---


# -----------------------------\
# Base Embedding Interface
# -----------------------------\
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
    def __init__(self, feature_size: int, d_model: int, use_value_norm: bool = False): # Added use_value_norm
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)
        # Conditionally create LayerNorm
        self.value_norm = nn.LayerNorm(d_model) if use_value_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, T, feature_size]
        proj = self.value_projection(x)    # → [B, T, d_model]
        # Apply LayerNorm if it exists
        if self.value_norm is not None:
            return self.value_norm(proj)   # → [B, T, d_model]
        return proj # Return projected value if no norm


@register_module("embedding", "flexible_value")
class FlexibleValueEmbedding(BaseEmbedding):
    def __init__(
        self,
        *,
        d_model: int,
        input_dims: Union[int, Sequence[int]],
        proj_builder: Callable[[int,int,Dict[str,Any]], nn.Module] = None,
        proj_kwargs: Dict[str,Any] = None,
        use_layer_norm: bool = False # Added layer norm option
    ):
        """
        d_model     – output embedding size
        input_dims  – either one int, or a list/tuple of ints for multiple feature-blocks
        proj_builder– factory fn (in_dim, out_dim, extra_kwargs) → nn.Module
                       if None, defaults to a simple linear
        proj_kwargs – extra kwargs passed to proj_builder
        use_layer_norm - If True, applies LayerNorm after projections. Defaults to False.
        """
        super().__init__(d_model)
        proj_kwargs = proj_kwargs or {}

        # default projection: a bias-free linear
        if proj_builder is None:
            proj_builder = lambda in_dim, out_dim, kw: nn.Linear(in_dim, out_dim, bias=False, **kw) # Pass kw to Linear

        # if multiple feature blocks, make one module per block
        if isinstance(input_dims, (list, tuple)):
            self.projections = nn.ModuleList([
                proj_builder(in_dim, d_model, proj_kwargs)
                for in_dim in input_dims
            ])
            self.input_dims = input_dims # Store for splitting in forward
        else:
            self.projections = proj_builder(input_dims, d_model, proj_kwargs)
            self.input_dims = [input_dims] # Store as list for consistency
        
        # Conditionally create LayerNorm
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, T, sum(input_dims)] if multiple blocks,
           or [B, T, input_dims] if single.
        """
        if isinstance(self.projections, nn.ModuleList):
            # split the last dim to match each proj
            # Ensure self.input_dims was stored correctly
            blocks = torch.split(x, self.input_dims, dim=-1)
            # embed each block and sum
            emb = sum(proj(b) for proj, b in zip(self.projections, blocks))
        else:
            emb = self.projections(x)
        
        # Apply LayerNorm if it exists
        if self.layer_norm is not None:
            return self.layer_norm(emb)
        return emb # Return embedded value if no norm


# -----------------------------\
# Sinusoidal Positional Embedding (Explicit Signature)
# -----------------------------\
@register_module("embedding", "sinusoidal")
class SinusoidalPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__(d_model=d_model)
        self.max_seq_len = max_seq_len
        weights = self._init_weights()
        self.register_buffer('weight', weights)

    def _init_weights(self) -> torch.Tensor:
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / self.d_model) for j in range(self.d_model)]
                for pos in range(self.max_seq_len)
            ]
        )
        out = torch.zeros(self.max_seq_len, self.d_model)
        sentinel = self.d_model // 2 if self.d_model % 2 == 0 else (self.d_model // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        return out

    @torch.no_grad()
    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        _bsz = batch_size
        _seq_len = seq_len
        _start = past_key_values_length
        if _seq_len <= 0:
            return torch.empty((_bsz, 0, self.d_model), device=self.weight.device, dtype=self.weight.dtype)
        _end = _start + _seq_len
        if _end > self.max_seq_len:
            max_req_pos = _end - 1
            raise IndexError(
                f"Requested position index {max_req_pos} is out of bounds for "
                f"SinusoidalPositionalEmbedding with max_seq_len {self.max_seq_len}."
            )
        positions = torch.arange(_start, _end, dtype=torch.long, device=self.weight.device)
        if positions.numel() == 0:
            print(f"Warning: Position tensor empty after arange(start={_start}, end={_end}). Should not happen.")
            return torch.empty((_bsz, 0, self.d_model), device=self.weight.device, dtype=self.weight.dtype)
        return self.weight[positions].unsqueeze(0)

# ... (Keep other existing embedding classes: Patch, Global, LearnedAbsolute, ShawRelative, Fourier, Time2Vec, ALiBi, Bucketed, ConvPos, TimeDelta) ...

# -----------------------------\
# Patch Embedding
# -----------------------------\
@register_module("embedding", "patch")
class TimeSeriesPatchEmbedding(BaseEmbedding):
    def __init__(
        self,
        patch_size: int,
        feature_size: int,
        d_model: int,
        stride: int = None,
        pad_value: float = 0.0,
    ):
        """
        Args:
          patch_size:   length of each patch
          feature_size: number of input channels F
          d_model:      output embedding dim D
          stride:       step between patch starts.  If None, uses patch_size (non-overlap).
          pad_value:    what value to pad with if L % stride != 0
        """
        super().__init__(d_model)
        self.patch_size   = patch_size
        self.feature_size = feature_size
        self.stride       = stride or patch_size
        self.pad_value    = pad_value
        self.proj         = nn.Linear(patch_size * feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        returns: [B, num_patches, D]
        """
        B, L, F = x.shape
        if F != self.feature_size:
            raise ValueError(f"Expected F={self.feature_size}, got {F}")

        # --- 1) Pad at end if needed so ((L - patch_size) % stride) == 0 ---\
        if L < self.patch_size:
            pad_len = self.patch_size - L
        else:
            rem = (L - self.patch_size) % self.stride
            pad_len = self.stride - rem if rem != 0 else 0

        if pad_len > 0:
            pad_tensor = torch.full(
                (B, pad_len, F), self.pad_value, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad_tensor], dim=1)
            L = L + pad_len

        # --- 2) Unfold into patches: [B, num_patches, patch_size, F] ---\
        x_patches = x.unfold(
            dimension=1,
            size=self.patch_size,
            step=self.stride
        )

        # --- 3) Flatten and project ---\
        B, num_patches, _, _ = x_patches.shape
        x_flat = x_patches.contiguous().view(B, num_patches, -1)
        return self.proj(x_flat)


# -----------------------------\
# Global Embedding
# -----------------------------\
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



@register_module("embedding", "rotary")
class RotaryPositionalEmbedding(BaseEmbedding):
    """
    Generates Rotary Positional Embedding frequencies (cos/sin).
    This version returns separate cos and sin tensors compatible with apply_rotary_pos_emb.
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        assert d_model % 2 == 0, "Rotary embedding dim (d_model) must be even"
        super().__init__(d_model=d_model)
        self.max_seq_len = max_seq_len
        self.base = base

        # Calculate inverse frequencies
        # Shape: [d_model / 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # freqs shape: [seq_len, d_model / 2]
        freqs = torch.outer(t, self.inv_freq)
        # emb shape: [seq_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Register cos and sin caches
        # Shape: [seq_len, d_model]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin caches sliced to the requested sequence length.

        Args:
            x (torch.Tensor): A dummy tensor to get the target device and dtype.
            seq_len (int): The required sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cos and sin caches, each shape [seq_len, d_model].
        """
        # Ensure cache is large enough and on the correct device/dtype
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            # Rebuild cache if needed (e.g., for longer sequences or device/dtype change)
            # Consider if rebuilding is desired or an error should be raised for seq_len > max_seq_len
            print(f"Warning: Rebuilding RoPE cache for seq_len={seq_len}, device={x.device}, dtype={x.dtype}")
            self._build_cache(seq_len=max(seq_len, self.max_seq_len_cached)) # Build larger if needed

        # Return sliced caches
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


@register_module("embedding", "learned_abs")
class LearnedAbsolutePositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__(d_model=d_model)
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        _bsz = batch_size
        _seq_len = seq_len
        _start = past_key_values_length
        if _seq_len <= 0:
            return torch.empty((_bsz, 0, self.d_model), device=self.embedding.weight.device)
        positions = torch.arange(_start, _start + _seq_len, dtype=torch.long, device=self.embedding.weight.device)
        if positions.numel() > 0 and positions.max() >= self.max_seq_len:
             raise IndexError(f"Position index {positions.max()} out of bounds for LearnedAbsolutePositionalEmbedding with max_seq_len {self.max_seq_len}")
        elif positions.numel() == 0 and _seq_len > 0:
             print(f"Warning: LearnedAbsolutePositionalEmbedding got seq_len={_seq_len} but position tensor is empty.")
             return torch.empty((_bsz, 0, self.d_model), device=self.embedding.weight.device)
        elif _seq_len == 0:
             return torch.empty((_bsz, 0, self.d_model), device=self.embedding.weight.device)

        embeds = self.embedding(positions)
        return embeds.unsqueeze(0)


@register_module("embedding", "relative_shaw")
class ShawRelativePositionalBias(BaseEmbedding):
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__(d_model=num_heads)
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = self.relative_bias.weight.device
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_distance, self.max_distance) + self.max_distance
        biases = self.relative_bias(distance_mat_clipped)
        biases = biases.permute(2, 0, 1).unsqueeze(0)
        return biases


@register_module("embedding", "fourier")
class FourierFeatureEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, num_features: int = 16):
        super().__init__(d_model=d_model)
        self.num_features = num_features
        self.proj = nn.Linear(2 * num_features, d_model)
        self.register_buffer("freqs", torch.exp(torch.linspace(0, np.log(1000), num_features)))

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = self.proj.weight.device
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        args = positions * self.freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        embeds = self.proj(feats)
        return embeds.unsqueeze(0)


@register_module("embedding", "time2vec")
class Time2VecEmbedding(BaseEmbedding):
    def __init__(self, d_model: int):
        super().__init__(d_model=d_model)
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = next(self.parameters()).device
        t = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(-1)
        lin = self.linear(t)
        per = torch.sin(self.periodic(t))
        embeds = torch.cat([lin, per], dim=-1)
        return embeds.unsqueeze(0)


@register_module("embedding", "alibi")
class ALiBiPositionalBias(BaseEmbedding):
    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__(d_model=num_heads)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        slopes = torch.Tensor(self._get_alibi_slopes(num_heads))
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_alibi_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if n == 0: return [] # Handle edge case
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        # Note: This forward signature (batch_size, seq_len) might differ from how it's called
        # in BaseMultiHeadAttention which expects seq_len_q, seq_len_k for bias calculation.
        # Adapting here to generate the causal mask bias based on seq_len.
        # The calling code might need to slice this bias if needed.

        # Generate bias for a square attention matrix T x T
        device = self.slopes.device
        slopes = self.slopes[None, :, None, None] # [1, H, 1, 1]
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None] # [T, T]
        abs_diff = diff.abs().unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        bias = -abs_diff * slopes # [1, H, T, T]
        return bias


@register_module("embedding", "bucketed")
class BucketedRelativeBias(BaseEmbedding):
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__(d_model=num_heads)
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_buckets = nn.Embedding(num_buckets, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = self.relative_buckets.weight.device
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None]
        diff_clipped = torch.clamp(diff, -self.max_distance, self.max_distance) + self.max_distance
        bucket_size = (2 * self.max_distance + 1) / self.num_buckets
        bucket_indices = torch.floor(diff_clipped / bucket_size).long().clamp(0, self.num_buckets - 1)
        biases = self.relative_buckets(bucket_indices)
        biases = biases.permute(2, 0, 1).unsqueeze(0)
        return biases


@register_module("embedding", "conv_pos")
class ConvolutionalPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, kernel_size: int = 3, max_seq_len: int = 2048):
        super().__init__(d_model=d_model)
        base_cls = resolve("embedding", "sinusoidal")
        self.base = base_cls(d_model=d_model, max_seq_len=max_seq_len)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        emb = self.base(batch_size, seq_len, past_key_values_length)
        x = emb.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)


@register_module("embedding", "timedelta")
class TimeDeltaEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__(d_model=d_model)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        if not list(self.mlp.parameters()):
             device = 'cpu'
        else:
             device = next(self.mlp.parameters()).device
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        feats = self.mlp(t)
        return feats.unsqueeze(0)


# --- Updated Stacked Embedding Wrapper ---\
@register_module("embedding", "stacked_embedding")
class StackedPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, embedding_configs: List[Dict], builder: ModuleBuilder):
        super().__init__(d_model=d_model)
        self.embeddings = nn.ModuleList()

        for config in embedding_configs:
            embed_type = config.get("type")
            embed_args = config.get("args", {}).copy()
            if not embed_type:
                raise ValueError("Each embedding config must have a 'type'.")
            try:
                module = builder._build(
                    kind="embedding",
                    name=embed_type,
                    base_kwargs={"hidden_size": d_model},
                    user_kwargs=embed_args
                )
                self.embeddings.append(module)
            except Exception as e:
                print(f"Error building embedding type '{embed_type}' with args {embed_args}: {e}")
                raise
        if not self.embeddings:
             print("Warning: No embeddings configured for StackedPositionalEmbedding.")

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        if not self.embeddings:
             device = kwargs.get('device', 'cpu') # Try to get device from kwargs
             print(f"Warning: StackedPositionalEmbedding has no modules, returning zeros on device {device}.")
             return torch.zeros((batch_size, seq_len, self.d_model), device=device)

        device = 'cpu'
        for module in self.embeddings:
             try: device = next(module.parameters()).device; break
             except StopIteration: pass
             try: device = next(module.buffers()).device; break
             except StopIteration: pass

        combined_embedding = torch.zeros((batch_size, seq_len, self.d_model), device=device)
        processed_any = False

        for embedding_module in self.embeddings:
            try:
                pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len, **kwargs)
            except TypeError as e:
                 sig = inspect.signature(embedding_module.forward)
                 valid_params = set(sig.parameters.keys())
                 has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                 unexpected_kwargs = {k: v for k, v in kwargs.items() if k not in valid_params}
                 if not has_var_kwargs and unexpected_kwargs:
                      print(f"Warning: Retrying {type(embedding_module).__name__}.forward without extra kwargs: {list(unexpected_kwargs.keys())}.")
                      valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                      try: pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len, **valid_kwargs)
                      except Exception as inner_e: raise e # Re-raise original error if retry fails for other reasons
                 else: raise e # Re-raise if it has VAR_KEYWORD or no unexpected_kwargs
            except Exception as e:
                 print(f"Error during forward pass of {type(embedding_module).__name__}: {e}"); raise

            is_sequence_embedding = (pos_embed.dim() == 3 and
                                     pos_embed.shape[-1] == self.d_model and
                                     pos_embed.shape[-2] == seq_len and
                                     pos_embed.shape[0] in [1, batch_size])
            if is_sequence_embedding:
                 combined_embedding = combined_embedding + pos_embed
                 processed_any = True
            else:
                 print(f"Warning: Skipping embedding {type(embedding_module).__name__} due to incompatible shape {pos_embed.shape} for sequence summation.")

        if not processed_any and self.embeddings:
            print("Warning: All configured embeddings were skipped.")

        return combined_embedding
