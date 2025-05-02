
import torch
import torch.nn as nn
import numpy as np
import inspect
from typing import Optional, Tuple, List, Dict

# === Corrected Imports ===
from temporal.registry.core import register_module, resolve # resolve is in core
# Need ModuleBuilder for type hint and its _build method
from temporal.models.module_builder_helper import ModuleBuilder


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
# Sinusoidal Positional Embedding (Explicit Signature)
# -----------------------------
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

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)


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
        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, np.log(1000), num_features)), requires_grad=False)

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
        slopes = 1.0 / (2.0 ** torch.arange(num_heads))
        self.register_buffer("slopes", slopes)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise IndexError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        device = self.slopes.device
        slopes = self.slopes.view(1, self.num_heads, 1, 1)
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None]
        diff = diff.abs().view(1, 1, seq_len, seq_len)
        bias = -diff * slopes
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
        diff = torch.clamp(diff, -self.max_distance, self.max_distance) + self.max_distance
        bucket_size = (2 * self.max_distance + 1) / self.num_buckets
        bucket = torch.floor(diff / bucket_size).long().clamp(0, self.num_buckets - 1)
        biases = self.relative_buckets(bucket)
        biases = biases.permute(2, 0, 1).unsqueeze(0)
        return biases


@register_module("embedding", "conv_pos")
class ConvolutionalPositionalEmbedding(BaseEmbedding):
    def __init__(self, d_model: int, kernel_size: int = 3, max_seq_len: int = 2048):
        super().__init__(d_model=d_model)
        base_cls = resolve("embedding", "sinusoidal")
        self.base = base_cls(d_model=d_model, max_seq_len=max_seq_len)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)

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
        # Ensure mlp parameters are on some device before accessing .device
        if not list(self.mlp.parameters()):
             device = 'cpu' # Fallback if MLP somehow has no parameters
        else:
             device = next(self.mlp.parameters()).device
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        feats = self.mlp(t)
        return feats.unsqueeze(0)


# --- Updated Stacked Embedding Wrapper ---
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

        first_embed_module = self.embeddings[0]
        try:
            device = next(first_embed_module.parameters()).device
        except StopIteration:
            try:
                device = next(first_embed_module.buffers()).device
            except StopIteration:
                 # Check device of the nn.ModuleList parameter itself? Unreliable.
                 # Fallback is necessary. 
                 print(f"Warning: Could not determine device for {type(first_embed_module).__name__} or parent StackedPositionalEmbedding. Assuming CPU.")
                 device = 'cpu'

        combined_embedding = torch.zeros((batch_size, seq_len, self.d_model), device=device)
        processed_any = False

        for embedding_module in self.embeddings:
            try:
                pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len, **kwargs)
            except TypeError as e:
                 sig = inspect.signature(embedding_module.forward)
                 valid_params = set(sig.parameters.keys())
                 has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                 if not has_var_kwargs and any(k not in valid_params for k in kwargs):
                      print(f"Warning: TypeError calling {type(embedding_module).__name__}.forward. Retrying without extra kwargs. Error: {e}")
                      try:
                          pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len)
                      except Exception as inner_e:
                           print(f"Retry failed for {type(embedding_module).__name__}.forward: {inner_e}")
                           raise e
                 else:
                      print(f"TypeError calling {type(embedding_module).__name__}.forward: {e}. Check signature.")
                      raise e
            except Exception as e:
                 print(f"Error during forward pass of {type(embedding_module).__name__}: {e}")
                 raise

            expected_shape_prefix = (seq_len, self.d_model)
            if not (pos_embed.shape[-len(expected_shape_prefix):] == expected_shape_prefix and
                    pos_embed.shape[0] in [1, batch_size]):
                 print(f"Warning: Skipping embedding {type(embedding_module).__name__} due to incompatible shape {pos_embed.shape} for summation. Expected shape like (*, {seq_len}, {self.d_model}).")
                 continue

            if pos_embed.shape[-1] != self.d_model:
                raise ValueError(f"Embedding {type(embedding_module).__name__} produced wrong dimension {pos_embed.shape[-1]}, expected {self.d_model}")

            combined_embedding = combined_embedding + pos_embed
            processed_any = True

        if not processed_any and self.embeddings:
            print("Warning: All configured embeddings were skipped due to incompatible shapes or errors.")

        return combined_embedding

