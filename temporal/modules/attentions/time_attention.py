import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# Suppose your base MHA class is here:
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention



# ----------------------------------------------------------------------
# 2) Utility classes / stubs
# ----------------------------------------------------------------------
class RotaryProjection(nn.Module):
    """
    Example stub for a rotary embedding / projection layer.
    """
    def __init__(self, max_len=100):
        super().__init__()
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In real usage, you'd apply rotary embeddings to x. 
        For example, you'd split x into even/odd dims, rotate by cos/sin, etc.
        Here we just pass x through.
        """
        return x


class QueryKeyProjection(nn.Module):
    """
    A custom Q/K projection block that might apply partial factor or rotary logic.
    """
    def __init__(self, dim, num_heads, proj_layer, proj_kwargs=None, partial_factor=(0.0, 0.5)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.partial_factor = partial_factor
        if proj_kwargs is None:
            proj_kwargs = {}
        # Create the "inner" projection, e.g. a RotaryProjection
        self.proj = proj_layer(**proj_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Real code might:
          - reshape x into [B, H, T, dim_per_head]
          - apply partial factor
          - pass through self.proj (rotary)
        For demo, we just pass x to self.proj and return.
        """
        x_proj = self.proj(x)
        return x_proj


class BinaryAttentionBias(nn.Module):
    """
    Example stub for a custom attention bias module.
    Possibly used to build a mask or additive bias across [B, H, T, T].
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Suppose we want a [B*H, T, T] shape for the bias, or [B, H, T, T].
        We'll guess that attn_scores => [B*H, T, T].
        For demonstration, we create a zero bias of the same shape.
        """
        return torch.zeros_like(attn_scores)


def expand_mask(
    attention_mask: torch.Tensor,
    tgt_len: int,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Convert a 2D [batch_size, src_len] attention mask
    into a 4D [batch_size, 1, tgt_len, src_len] mask
    => where 1 => keep, 0 => mask becomes 0 => keep, -1e9 => mask out.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask of shape [B, src_len], got {attention_mask.shape}.")

    bsz, src_len = attention_mask.shape
    expanded_mask = attention_mask[:, None, None, :]  # => [B,1,1,src_len]
    expanded_mask = expanded_mask.expand(bsz, 1, tgt_len, src_len)  # => [B,1,tgt_len,src_len]
    expanded_mask = expanded_mask.to(dtype=dtype)
    inverted_mask = (1.0 - expanded_mask) * -1e9
    return inverted_mask


# ----------------------------------------------------------------------
# 3) TimeSeriesAttention
# ----------------------------------------------------------------------
class TimeSeriesAttention(BaseMultiHeadAttention):
    """
    Example attention module that merges your "time series" logic with
    the standard multi-head base. It uses:
      - QueryKeyProjection for Q/K
      - BinaryAttentionBias for optional additive bias
      - expand_mask for 2D -> 4D mask
    Returns (attn_output, attn_probs, present_key_value).
    """

    def __init__(self, config: TimeSeriesConfig):
        """
        Inherits from BaseMultiHeadAttention. 
        We feed in config.d_model -> embed_dim, config.num_heads, etc.
        """
        super().__init__(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            is_decoder=config.is_decoder,
            bias=config.bias
        )

        self.mask_flag = config.mask_flag
        self.output_attention = config.output_attention
        self.covariate = config.covariate
        self.flash_attention = config.flash_attention

        # If config.scale is specified, we might override self.scaling:
        if config.scale is not None:
            self.scaling = config.scale

        # Custom QK projection
        self.qk_proj = QueryKeyProjection(
            dim=self.head_dim,         # or self.embed_dim, depends on your approach
            num_heads=config.num_heads,
            proj_layer=RotaryProjection,
            proj_kwargs=dict(max_len=config.max_len),
            partial_factor=(0.0, 0.5)
        )

        # Additional attention bias
        self.attn_bias = BinaryAttentionBias(dim=self.embed_dim, num_heads=config.num_heads)

    def compute_attention_scores(self, query_states, key_states):
        """
        Overwrite base method to incorporate your custom Q/K projection or bias if needed.
        The base class expects shape [B*H, T, head_dim]. 
        We'll do something minimal here:
          - reshape them into [B, H, T, head_dim] if needed
          - pass them into self.qk_proj
          - standard matmul
          - add custom bias
        """
        b_h, tgt_len, d = query_states.shape  # b_h = batch_size * num_heads
        # Reshape => [B, H, T, d]
        # If you want to handle big batch + heads, we find B using integer division: B = b_h // self.num_heads
        B = b_h // self.num_heads

        # [B, H, T, d]
        q_4d = query_states.view(B, self.num_heads, tgt_len, d)
        k_4d = key_states.view(B, self.num_heads, -1, d)

        # 1) Apply custom projection
        q_4d = self.qk_proj(q_4d)  # shape stays [B, H, T, d]
        k_4d = self.qk_proj(k_4d)  # shape [B, H, src_len, d]

        # 2) Flatten back => [B*H, T, d]
        q_4d_flat = q_4d.view(b_h, tgt_len, d)
        k_4d_flat = k_4d.view(b_h, -1, d)

        # 3) Standard dot-product
        attn_scores = torch.bmm(q_4d_flat, k_4d_flat.transpose(-1, -2))  # => [b_h, T, src_len]

        # 4) Add custom bias => we assume attn_scores => [b_h, T, src_len]
        #    attn_bias might also expect that shape
        bias_tensor = self.attn_bias(attn_scores)  # => same shape
        attn_scores = attn_scores + bias_tensor

        return attn_scores

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, tgt_len, d_model]
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None, # can be 2D or 4D
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Identical signature to your other MHA modules. We'll do:
         1) Q, K, V from hidden_states
         2) Optionally expand 2D -> 4D mask
         3) compute attn_probs => matmul => out
         4) Return (attn_output, attn_probs, present_key_value)
        """
        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        # 1) Q, K, V
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # 2) Reshape => [B, H, T, head_dim]
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        # 3) If caching
        if past_key_value is not None:
            pk, pv = past_key_value
            key_states = torch.cat([pk, key_states], dim=2)
            value_states = torch.cat([pv, value_states], dim=2)

        present_key_value = (key_states, value_states) if self.is_decoder else None

        # Flatten => [B*H, T, head_dim]
        query_states = query_states.view(bsz * self.num_heads, tgt_len, self.head_dim)
        key_states = key_states.view(bsz * self.num_heads, -1, self.head_dim)
        value_states = value_states.view(bsz * self.num_heads, -1, self.head_dim)

        # 4) If attention_mask is 2D [B, src_len], expand to 4D [B,1,tgt_len,src_len]
        #    If it's already 4D, do nothing
        src_len = key_states.size(1)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # expand => [B, 1, tgt_len, src_len]
                attention_mask = expand_mask(attention_mask, tgt_len, dtype=query_states.dtype)
            elif attention_mask.dim() == 4:
                # we trust it's correct shape
                pass
            else:
                raise ValueError("attention_mask must be 2D or 4D")

        # 5) attn_scores => [B*H, tgt_len, src_len]
        attn_scores = self.compute_attention_scores(query_states, key_states)

        # 6) add mask if present
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Mask shape mismatch. Expecting [B,1,{tgt_len},{src_len}], got {attention_mask.shape}")
            attn_scores = attn_scores.view(bsz, self.num_heads, tgt_len, src_len)
            attn_scores = attn_scores + attention_mask  # broadcast
            attn_scores = attn_scores.view(bsz * self.num_heads, tgt_len, src_len)

        # 7) softmax
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=attn_scores.dtype)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 8) Weighted sum => [B*H, tgt_len, head_dim]
        attn_output = torch.bmm(attn_probs, value_states)

        # 9) Reshape => [B, T, D]
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 10) Optionally return attn_probs => [B, H, tgt_len, src_len]
        if output_attentions or self.output_attention:
            attn_probs_4d = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_probs_4d, present_key_value
        else:
            return attn_output, None, present_key_value


# ----------------------------------------------------------------------
# 4) MultiScaleTimeAttention
# ----------------------------------------------------------------------
class MultiScaleTimeAttention(TimeSeriesAttention):
    """
    This class reuses the parent's forward logic as the "fine scale" pass
    and also uses a separate TimeSeriesAttention for coarse scale.
    """
    def __init__(
        self,
        config: TimeSeriesConfig,
        time_attn_large: TimeSeriesAttention,
        downsample_factor: int = 4
    ):
        """
        Args:
          config: config for the "fine-scale" TimeSeriesAttention (this class)
          time_attn_large: a separate TimeSeriesAttention for coarse scale
          downsample_factor: how many time steps to group for coarse
        """
        super().__init__(config)  # sets up "fine-scale" logic
        self.time_attn_large = time_attn_large
        self.downsample_factor = downsample_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: shape (B, L, d_model)

        Steps:
         1) fine-scale attention => super().forward
         2) downsample => coarse
         3) coarse-level attention => self.time_attn_large
         4) upsample
         5) combine
        """
        B, L, d_model = hidden_states.shape

        # 1) Fine-scale attention => call parent's forward
        #    We don't do cross-attn here, so pass key_value_states=None
        fine_out, _, _ = super().forward(hidden_states=hidden_states)

        # 2) Downsample for coarse-level
        #    group steps of size 'downsample_factor' => average
        grouped = rearrange(hidden_states, "b (k s) d -> b k s d", s=self.downsample_factor)
        coarse = grouped.mean(dim=2)  # => (B, k, d_model) where k = L // downsample_factor

        # 3) Coarse-level attention => use the 'time_attn_large'
        coarse_out, _, _ = self.time_attn_large(coarse)

        # 4) Upsample => expand coarse_out back to length L
        coarse_expanded = torch.repeat_interleave(coarse_out, repeats=self.downsample_factor, dim=1)
        # => shape (B, L, d_model)

        # 5) Combine
        output = fine_out + coarse_expanded  # (B, L, d_model)
        return output

        import math
