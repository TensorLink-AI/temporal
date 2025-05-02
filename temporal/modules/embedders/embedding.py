
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
# Sinusoidal Positional Embedding (Explicit Signature)
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
    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Generates positional embeddings based on explicit integer batch size, 
        sequence length, and past sequence length.
        """
        # We now expect integers directly from the caller (Decoder/Encoder)
        _bsz = batch_size
        _seq_len = seq_len
        _start = past_key_values_length

        # Early-exit on empty sequence
        if _seq_len <= 0:
            return torch.empty((_bsz, 0, self.dim), device=self.weight.device, dtype=self.weight.dtype)

        # Build position indices
        _end = _start + _seq_len
        if _end > self.max_seq_len:
            max_req_pos = _end - 1
            raise IndexError(
                f"Requested position index {max_req_pos} is out of bounds for "
                f"SinusoidalPositionalEmbedding with max_seq_len {self.max_seq_len}."
            )

        positions = torch.arange(_start, _end, dtype=torch.long, device=self.weight.device)

        # Check for empty positions defensively (should not happen if _seq_len > 0)
        if positions.numel() == 0:
            print(f"Warning: Position tensor empty after arange(start={_start}, end={_end}). Should not happen.")
            return torch.empty((_bsz, 0, self.dim), device=self.weight.device, dtype=self.weight.dtype)

        # Gather + unsqueeze for batch broadcast
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



@register_module("embedding", "rotary")
class RotaryPositionalEmbedding(BaseEmbedding):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        """
        RoPE for dim: must be even.
        Precomputes cos & sin tables of shape [max_seq_len, dim//2].
        """
        assert dim % 2 == 0, "Rotary embedding dim must be even"
        super().__init__(d_model=dim)
        self.dim = dim
        self.half = dim // 2
        self.max_seq_len = max_seq_len

        # angles: [max_seq_len, half]
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.half, 1) / self.half))
        t = np.arange(max_seq_len)
        freqs = np.einsum("i,j->ij", t, inv_freq)
        # register as buffers so they move to the right device automatically
        self.register_buffer("cos", torch.FloatTensor(np.cos(freqs)))
        self.register_buffer("sin", torch.FloatTensor(np.sin(freqs)))

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Returns a [1, seq_len, dim] tensor of cos/sin pairs to multiply into Q/K.
        Usage in your attention:
           q_rot = (q * cos) + (rotate_half(q) * sin)
        """
        device = self.cos.device
        # slice the first seq_len positions
        cos = self.cos[:seq_len].unsqueeze(0)  # [1, seq_len, half]
        sin = self.sin[:seq_len].unsqueeze(0)  # [1, seq_len, half]
        # interleave to match dim: [1, seq_len, dim]
        cos = torch.stack([cos, cos], dim=-1).view(1, seq_len, self.dim)
        sin = torch.stack([sin, sin], dim=-1).view(1, seq_len, self.dim)
        return torch.cat([cos, sin], dim=0)    # you can return both or pack as needed

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Assumes last dim even: split in half and swap [−x2, x1]
        """
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)




@register_module("embedding", "learned_abs")
class LearnedAbsolutePositionalEmbedding(BaseEmbedding):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__(d_model=dim)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        _bsz = batch_size
        _seq_len = seq_len
        _start = past_key_values_length
        if _seq_len <= 0:
            return torch.empty((_bsz, 0, self.dim), device=self.embedding.weight.device)
        positions = torch.arange(_start, _start + _seq_len, dtype=torch.long, device=self.embedding.weight.device)
        embeds = self.embedding(positions)  # [seq_len, dim]
        return embeds.unsqueeze(0)


@register_module("embedding", "relative_shaw")
class ShawRelativePositionalBias(BaseEmbedding):
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__(d_model=num_heads)
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        # Computes bias for each head: [seq_len, seq_len] -> [1, num_heads, seq_len, seq_len]
        device = self.relative_bias.weight.device
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_distance, self.max_distance) + self.max_distance
        biases = self.relative_bias(distance_mat_clipped)  # [seq_len, seq_len, num_heads]
        biases = biases.permute(2, 0, 1).unsqueeze(0)       # [1, num_heads, seq_len, seq_len]
        return biases


@register_module("embedding", "fourier")
class FourierFeatureEmbedding(BaseEmbedding):
    def __init__(self, dim: int, num_features: int = 16):
        super().__init__(d_model=dim)
        self.dim = dim
        self.num_features = num_features
        self.proj = nn.Linear(2 * num_features, dim)
        # fixed frequencies
        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, np.log(1000), num_features)), requires_grad=False)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = self.proj.weight.device
        positions = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len,1]
        args = positions * self.freqs.unsqueeze(0)                    # [seq_len, num_features]
        feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # [seq_len, 2*num_features]
        embeds = self.proj(feats)                                      # [seq_len, dim]
        return embeds.unsqueeze(0)


@register_module("embedding", "time2vec")
class Time2VecEmbedding(BaseEmbedding):
    def __init__(self, dim: int):
        super().__init__(d_model=dim)
        self.dim = dim
        # one linear for time + one for periodic
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, dim - 1)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = next(self.parameters()).device
        t = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(-1)  # [seq_len,1]
        lin = self.linear(t)                  # [seq_len,1]
        per = torch.sin(self.periodic(t))     # [seq_len, dim-1]
        embeds = torch.cat([lin, per], dim=-1) # [seq_len, dim]
        return embeds.unsqueeze(0)


# 1) ALiBi: Attention with Linear Biases
@register_module("embedding", "alibi")
class ALiBiPositionalBias(BaseEmbedding):
    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__(d_model=num_heads)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # simple geometric slopes: 1/2^i
        slopes = 1.0 / (2.0 ** torch.arange(num_heads))
        self.register_buffer("slopes", slopes)

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise IndexError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        device = self.slopes.device
        # [1, num_heads, 1, 1]
        slopes = self.slopes.view(1, self.num_heads, 1, 1)
        # compute pairwise distances [seq_len, seq_len]
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None]
        diff = diff.abs().view(1, 1, seq_len, seq_len)
        # bias = -|i-j| * slope
        bias = -diff * slopes
        return bias  # [1, num_heads, seq_len, seq_len]


# 2) Bucketed Relative (T5-style)
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
        # compute pairwise distances clipped to [-max_distance, max_distance]
        positions = torch.arange(seq_len, device=device)
        diff = positions[None, :] - positions[:, None]
        diff = torch.clamp(diff, -self.max_distance, self.max_distance) + self.max_distance
        # map distances into buckets
        bucket_size = (2 * self.max_distance + 1) / self.num_buckets
        bucket = torch.floor(diff / bucket_size).long().clamp(0, self.num_buckets - 1)
        # lookup bias per (i,j) per head
        biases = self.relative_buckets(bucket)  # [seq_len, seq_len, num_heads]
        biases = biases.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, seq_len, seq_len]
        return biases


# 3) Convolutional Positional Encoding (refines sinusoidal)
@register_module("embedding", "conv_pos")
class ConvolutionalPositionalEmbedding(BaseEmbedding):
    def __init__(self, dim: int, kernel_size: int = 3, max_seq_len: int = 2048):
        super().__init__(d_model=dim)
        # reuse sinusoidal as base
        from temporal.registry.core import resolve
        base_cls = resolve("embedding", "sinusoidal")
        self.base = base_cls(dim, max_seq_len)
        # 1D conv to refine
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)

    def forward(self, batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor:
        # get sinusoidal embeddings [1, seq_len, dim]
        emb = self.base(batch_size, seq_len, past_key_values_length)
        # shape → [1, dim, seq_len]
        x = emb.permute(0, 2, 1)
        # refine via conv
        x = self.conv(x)
        # back to [1, seq_len, dim]
        return x.permute(0, 2, 1)


# 4) Time-Delta / Inter-Event Embeddings
@register_module("embedding", "timedelta")
class TimeDeltaEmbedding(BaseEmbedding):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__(d_model=dim)
        # small MLP: [1] → hidden_dim → dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        device = next(self.mlp.parameters()).device
        # assume uniform intervals: t = [0,1,2,...]
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)  # [seq_len,1]
        feats = self.mlp(t)  # [seq_len, dim]
        # broadcast as [1, seq_len, dim]
        return feats.unsqueeze(0)


@register_module("embedding", "stacked_embedding")
class StackedPositionalEmbedding(BaseEmbedding): # Or perhaps inherit directly from nn.Module
    def __init__(self, d_model: int, embedding_configs: list[dict]):
        """
        Initializes a wrapper to stack multiple positional embeddings.

        Args:
            d_model (int): The embedding dimension, must match the model.
            embedding_configs (list[dict]): A list of configurations for the
                                           embeddings to stack. Each dict should
                                           contain at least 'type' (the registered name)
                                           and optionally 'args' for the specific embedding.
                                           Example:
                                           [
                                               {'type': 'sinusoidal', 'args': {'max_seq_len': 1024}},
                                               {'type': 'timedelta', 'args': {'hidden_dim': 64}}
                                           ]
        """
        super().__init__(d_model=d_model) # If inheriting BaseEmbedding
        # super().__init__() # If inheriting nn.Module directly
        self.d_model = d_model
        self.embeddings = nn.ModuleList()

        for config in embedding_configs:
            embed_type = config["type"]
            embed_args = config.get("args", {})

            # Ensure the dimension matches
            if 'dim' in embed_args:
                 embed_args['dim'] = d_model
            if 'd_model' in embed_args:
                 embed_args['d_model'] = d_model
            # Add dim/d_model if not present, assuming it's a required arg named 'dim' or 'd_model'
            # (This might need adjustment based on specific embedding __init__ signatures)
            if 'dim' not in embed_args and 'd_model' not in embed_args:
                 # Try adding both common names, constructor should ignore unused ones
                 embed_args['dim'] = d_model
                 embed_args['d_model'] = d_model

            # --- Option 1: Using a hypothetical build_module function ---
            # module = build_module("embedding", embed_type, args=embed_args)
            # self.embeddings.append(module)

            # --- Option 2: Manual instantiation (requires listing all types) ---
            # This requires manually mapping type names to classes
            if embed_type == "sinusoidal":
                # Need to know exact args, assuming 'dim', 'max_seq_len'
                # Filter args relevant to SinusoidalPositionalEmbedding
                relevant_args = {k: v for k, v in embed_args.items() if k in ['dim', 'max_seq_len']}
                self.embeddings.append(SinusoidalPositionalEmbedding(**relevant_args))
            elif embed_type == "timedelta":
                 # Need to know exact args, assuming 'dim', 'hidden_dim'
                relevant_args = {k: v for k, v in embed_args.items() if k in ['dim', 'hidden_dim']}
                self.embeddings.append(TimeDeltaEmbedding(**relevant_args))
            # elif embed_type == "my_new_embedding":
            #     self.embeddings.append(MyNewPositionalEmbedding(d_model=d_model, **embed_args)) # Adapt args
            else:
                # Ideally, use a builder function or raise an error
                raise ValueError(f"Unsupported embedding type for stacking: {embed_type}")
                # Or try a generic build using registry if available

    def forward(self, batch_size: int, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Computes and combines embeddings from all stacked modules.

        Args:
            batch_size (int): The batch size.
            seq_len (int): The sequence length.
            **kwargs: Additional arguments potentially needed by specific embeddings
                      (like past_key_values_length).

        Returns:
            torch.Tensor: The combined positional embedding tensor of shape
                          [batch_size, seq_len, d_model].
        """
        combined_embedding = None
        device = None

        for embedding_module in self.embeddings:
            # Call the forward method of the individual embedding
            # Ensure the called signature is compatible
            try:
                # Pass all potentially relevant args
                pos_embed = embedding_module(batch_size=batch_size, seq_len=seq_len, **kwargs)
            except TypeError as e:
                 print(f"Warning: TypeError calling {type(embedding_module).__name__}.forward: {e}. Check signature compatibility.")
                 # Potentially try a simpler signature if applicable, or re-raise
                 # pos_embed = embedding_module(seq_len=seq_len) # Example fallback
                 raise e # Re-raise for now

            if pos_embed.shape[-1] != self.d_model:
                raise ValueError(f"Embedding {type(embedding_module).__name__} produced wrong dimension "
                                 f"{pos_embed.shape[-1]}, expected {self.d_model}")

            if device is None:
                device = pos_embed.device
                # Initialize combined embedding only once we know the device and shape basics
                combined_embedding = torch.zeros((batch_size, seq_len, self.d_model), device=device)

            # Combine - Summation is common
            combined_embedding = combined_embedding + pos_embed

        if combined_embedding is None:
             # Handle case where no embeddings were configured or ran
             print("Warning: No embeddings were configured or ran in StackedPositionalEmbedding.")
             # Need a reliable way to get device if nothing ran
             try:
                 fallback_device = next(self.parameters()).device
             except StopIteration:
                 fallback_device = 'cpu' # Default if the wrapper itself has no parameters
             combined_embedding = torch.zeros((batch_size, seq_len, self.d_model), device=fallback_device)


        return combined_embedding
