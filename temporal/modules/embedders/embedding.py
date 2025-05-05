
import torch
import torch.nn as nn
import numpy as np
import inspect
import math # Import math for log calculation in BucketedRelativeBias if needed, or ALiBi later
from typing import Optional, Tuple, List, Dict

# === Corrected Imports ===
from temporal.registry.core import register_module, resolve # resolve is in core
# Need ModuleBuilder for type hint and its _build method
from temporal.models.module_builder_helper import ModuleBuilder


# --- NEW: RoPE Helper Functions ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Positional Embedding to query and key tensors."""
    # cos, sin: [seq_len, dim] or [bsz, 1, seq_len, dim]
    # q, k: [bsz, num_heads, seq_len, head_dim]

    # Ensure cos/sin are broadcastable over heads dimension if needed
    # Simple case assumes cos/sin are [seq_len, head_dim] or similar that can be indexed by position_ids
    # and then applied element-wise after reshaping/repeating.

    if cos.dim() == 2: # [seq_len, dim] -> need to gather based on position_ids if provided
        if position_ids is None:
            # Assuming standard range if position_ids not given
             cos = cos[None, None, :, :] # -> [1, 1, seq_len, dim]
             sin = sin[None, None, :, :] # -> [1, 1, seq_len, dim]
        else:
            # Gather based on position IDs: [bsz, seq_len] -> [bsz, seq_len, dim]
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


# -----------------------------\
# Value Embedding
# -----------------------------\
@register_module("embedding", "value")
class TimeSeriesValueEmbedding(BaseEmbedding):
    def __init__(self, feature_size: int, d_model: int):
        super().__init__(d_model)
        self.value_projection = nn.Linear(feature_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_projection(x)


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

# ... (Keep other existing embedding classes: Patch, Global, Rotary, LearnedAbsolute, ShawRelative, Fourier, Time2Vec, ALiBi, Bucketed, ConvPos, TimeDelta) ...

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
            # too short: pad up to at least one patch
            pad_len = self.patch_size - L
        else:
            rem = (L - self.patch_size) % self.stride
            pad_len = self.stride - rem if rem != 0 else 0

        if pad_len > 0:
            # pad last time‐steps with pad_value
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
        )  # shape: [B, num_patches, patch_size, F]

        # --- 3) Flatten and project ---\
        B, num_patches, _, _ = x_patches.shape
        x_flat = x_patches.contiguous().view(B, num_patches, -1)  # [B, num_patches, patch_size*F]
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
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        assert d_model % 2 == 0, "Rotary embedding dim must be even"
        super().__init__(d_model=d_model)
        self.half = d_model // 2
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.half, 1) / self.half))
        t = np.arange(max_seq_len)
        freqs = np.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", torch.FloatTensor(np.cos(freqs)))
        self.register_buffer("sin", torch.FloatTensor(np.sin(freqs)))

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = self.cos.device
        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)
        cos_interleaved = torch.stack([cos, cos], dim=-1).view(1, seq_len, self.d_model)
        sin_interleaved = torch.stack([sin, sin], dim=-1).view(1, seq_len, self.d_model)
        return torch.cat([cos_interleaved, sin_interleaved], dim=0)

    # Removed static rotate_half from here, moved to top-level helper


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
        if positions.max() >= self.max_seq_len:
             raise IndexError(f"Position index {positions.max()} out of bounds for LearnedAbsolutePositionalEmbedding with max_seq_len {self.max_seq_len}")
        embeds = self.embedding(positions)
        return embeds.unsqueeze(0)


@register_module("embedding", "relative_shaw")
class ShawRelativePositionalBias(BaseEmbedding):
    def __init__(self, num_heads: int, max_distance: int = 128):
        # Note: BaseEmbedding d_model doesn't align well here. num_heads is more relevant.
        super().__init__(d_model=num_heads) # Pass num_heads as d_model for base compatibility
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        # This generates a bias tensor for attention scores, not a standard sequence embedding
        device = self.relative_bias.weight.device
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_distance, self.max_distance) + self.max_distance
        biases = self.relative_bias(distance_mat_clipped) # [seq_len, seq_len, num_heads]
        biases = biases.permute(2, 0, 1).unsqueeze(0) # [1, num_heads, seq_len, seq_len]
        return biases


@register_module("embedding", "fourier")
class FourierFeatureEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, num_features: int = 16):
        super().__init__(d_model=d_model)
        self.num_features = num_features
        self.proj = nn.Linear(2 * num_features, d_model)
        # Use nn.Parameter for frequencies if they should be learnable, or register_buffer if fixed
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
        # BaseEmbedding d_model is not directly applicable here. Use num_heads.
        super().__init__(d_model=num_heads)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # Calculate slopes for ALiBi
        slopes = torch.Tensor(self._get_alibi_slopes(num_heads))
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_alibi_slopes(n):
        # Copied from HF Transformers BLOOM implementation
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        # Generates the ALiBi bias tensor for attention scores
        if seq_len > self.max_seq_len:
            # Note: ALiBi doesn't strictly need max_seq_len, it computes dynamically.
            # Keeping it for potential consistency checks or future use.
             print(f"Warning: seq_len={seq_len} > max_seq_len={self.max_seq_len} for ALiBi. Computing dynamically.")
            # raise IndexError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device = self.slopes.device
        # slopes: [H] -> [1, H, 1, 1]
        slopes = self.slopes[None, :, None, None]
        # positions: [T]
        positions = torch.arange(seq_len, device=device)
        # diff: [T, T] matrix of distances (query_pos - key_pos)
        diff = positions[None, :] - positions[:, None]
        # abs_diff: [1, 1, T, T]
        abs_diff = diff.abs().unsqueeze(0).unsqueeze(0)
        # bias: [1, H, T, T]
        # Negative sign because bias is added, and we want penalties for distance
        bias = -abs_diff * slopes
        # Expand for batch dimension if needed (usually handled by broadcasting)
        # bias = bias.expand(batch_size, -1, -1, -1) # Not needed due to broadcasting
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
        # Generates bias tensor for attention scores
        device = self.relative_buckets.weight.device
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None] # [T, T]
        # Bucketing logic needs careful implementation (assuming causal for now)
        diff = torch.clamp(diff, max=0) # Consider only past relative positions (<= 0)
        diff = diff.abs() # Use absolute distance for bucketing

        # Simplified bucketing (needs refinement based on T5 paper if exact match needed)
        bucket_size = self.max_distance / (self.num_buckets / 2) # Example logic
        is_small = diff < (self.max_distance // 2)
        bucket_if_small = diff // (bucket_size // 2) # Example finer buckets near 0
        bucket_if_large = (self.num_buckets // 2) + \
                          torch.minimum(torch.floor(torch.log(diff / (self.max_distance // 2)) / math.log(2) * (self.num_buckets / 2)),
                                        torch.tensor(self.num_buckets / 2 -1, device=device)) # Example log buckets

        bucket_indices = torch.where(is_small, bucket_if_small, bucket_if_large).long()
        bucket_indices = torch.clamp(bucket_indices, 0, self.num_buckets - 1)


        # # Original attempt - needs review based on T5 paper for exact logic
        # diff_clipped = torch.clamp(diff, -self.max_distance, self.max_distance) + self.max_distance # [0, 2*max_dist]
        # bucket_size = (2 * self.max_distance + 1) / self.num_buckets
        # bucket_indices = torch.floor(diff_clipped / bucket_size).long().clamp(0, self.num_buckets - 1)

        biases = self.relative_buckets(bucket_indices) # [T, T, H]
        biases = biases.permute(2, 0, 1).unsqueeze(0) # [1, H, T, T]
        return biases


@register_module("embedding", "conv_pos")
class ConvolutionalPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, kernel_size: int = 3, max_seq_len: int = 2048):
        super().__init__(d_model=d_model)
        # Use Sinusoidal as the base before convolution
        base_cls = resolve("embedding", "sinusoidal")
        self.base = base_cls(d_model=d_model, max_seq_len=max_seq_len)
        # 1D convolution over the time dimension
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model) # Depthwise conv

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        # Get base sinusoidal embedding [B, T, D]
        emb = self.base(batch_size, seq_len, past_key_values_length)
        # Permute for Conv1D [B, D, T]
        x = emb.permute(0, 2, 1)
        # Apply convolution
        x = self.conv(x)
        # Permute back [B, T, D]
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
        # Ensure mlp parameters are on some device before accessing .device
        if not list(self.mlp.parameters()):
             device = 'cpu' # Fallback if MLP somehow has no parameters
        else:
             device = next(self.mlp.parameters()).device
        # Create time steps [T, 1]
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        # Pass through MLP [T, D]
        feats = self.mlp(t)
        # Expand for batch [B, T, D]
        return feats.unsqueeze(0)


# --- Updated Stacked Embedding Wrapper ---\
@register_module("embedding", "stacked_embedding")
class StackedPositionalEmbedding(BaseEmbedding):
    # === Added builder parameter ===
    def __init__(self, d_model: int, embedding_configs: List[Dict], builder: ModuleBuilder):
        """
        Initializes a wrapper to stack multiple positional embeddings by summing them.

        Args:
            d_model (int): The embedding dimension, must match the model.
            embedding_configs (list[dict]): List of configs for embeddings to stack.
                                           Each dict needs 'type' and 'args'.
            builder (ModuleBuilder): The builder instance used to construct sub-modules.
        """
        super().__init__(d_model=d_model)
        self.embeddings = nn.ModuleList()

        for config in embedding_configs:
            embed_type = config.get("type")
            embed_args = config.get("args", {}).copy()

            if not embed_type:
                raise ValueError("Each embedding config in 'embedding_configs' must have a 'type'.")

            # Use the passed builder's _build method
            try:
                # _build handles argument preparation (including d_model/hidden_size)
                # and builder injection if the sub-module needs it (though unlikely here)
                # We pass embed_args as user_kwargs. base_kwargs are handled by _build if needed.
                module = builder._build(
                    kind="embedding",
                    name=embed_type,
                    # Pass d_model explicitly in base_kwargs for clarity within stacker
                    base_kwargs={"hidden_size": d_model}, # _build maps hidden_size -> d_model
                    user_kwargs=embed_args
                )
                self.embeddings.append(module)
            except Exception as e:
                print(f"Error building embedding type '{embed_type}' with args {embed_args} using builder: {e}")
                raise

        if not self.embeddings:
             print("Warning: No embeddings were configured for StackedPositionalEmbedding.")

    # === forward method remains the same ===
    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Computes and combines embeddings from all stacked modules by summation.
        Handles incompatible shapes (like relative biases).
        """
        if not self.embeddings:
             try:
                 fallback_device = next(self.parameters()).device
             except StopIteration:
                 fallback_device = 'cpu'
             print(f"Warning: StackedPositionalEmbedding has no modules, returning zeros on device {fallback_device}.")
             return torch.zeros((batch_size, seq_len, self.d_model), device=fallback_device)

        # Determine device from the first module that has parameters or buffers
        device = 'cpu' # Default
        for module in self.embeddings:
             try:
                 device = next(module.parameters()).device
                 break
             except StopIteration:
                 try:
                     device = next(module.buffers()).device
                     break
                 except StopIteration:
                     continue
        # If still 'cpu', maybe the ModuleList itself has a device? Unlikely reliable.

        combined_embedding = torch.zeros((batch_size, seq_len, self.d_model), device=device)
        processed_any = False

        for embedding_module in self.embeddings:
            try:
                # Attempt to pass all kwargs, handle TypeError if module doesn't accept them
                pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len, **kwargs)
            except TypeError as e:
                 sig = inspect.signature(embedding_module.forward)
                 valid_params = set(sig.parameters.keys())
                 has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

                 # Identify unexpected kwargs
                 unexpected_kwargs = {k: v for k, v in kwargs.items() if k not in valid_params}

                 if not has_var_kwargs and unexpected_kwargs:
                      print(f"Warning: TypeError calling {type(embedding_module).__name__}.forward. Retrying without extra kwargs: {list(unexpected_kwargs.keys())}. Error: {e}")
                      # Create kwargs dict with only valid parameters
                      valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                      try:
                          pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len, **valid_kwargs)
                      except Exception as inner_e:
                           print(f"Retry failed for {type(embedding_module).__name__}.forward: {inner_e}")
                           raise e # Re-raise original TypeError if retry fails
                 else:
                      # TypeError wasn't due to unexpected kwargs, or module accepts **kwargs
                      print(f"TypeError calling {type(embedding_module).__name__}.forward: {e}. Check signature and required args.")
                      raise e
            except Exception as e:
                 print(f"Error during forward pass of {type(embedding_module).__name__}: {e}")
                 raise

            # --- Shape Check and Summation ---
            # Allow for shapes like [B, T, D], [1, T, D]
            # Also allow for bias shapes like [1, H, T, T] or [B, H, T, T] - these should not be added here.
            is_sequence_embedding = (pos_embed.dim() == 3 and
                                     pos_embed.shape[-1] == self.d_model and
                                     pos_embed.shape[-2] == seq_len and
                                     pos_embed.shape[0] in [1, batch_size])

            if is_sequence_embedding:
                 combined_embedding = combined_embedding + pos_embed
                 processed_any = True
            else:
                 # Handle potential bias tensors returned by relative position modules
                 # They shouldn't be summed with sequence embeddings. StackedEmbedding might
                 # need modification if it's intended to return multiple embedding types (value + bias).
                 # For now, just warn and skip non-sequence embeddings.
                 print(f"Warning: Skipping embedding {type(embedding_module).__name__} due to incompatible shape {pos_embed.shape} for sequence summation. Expected shape like (*, {seq_len}, {self.d_model}).")
                 continue


        if not processed_any and self.embeddings:
            print("Warning: All configured embeddings were skipped due to incompatible shapes or errors.")

        return combined_embedding
