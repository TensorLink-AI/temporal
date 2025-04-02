import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class BaseAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.scaling = self.head_dim ** -0.5

        # Shared QKV projections and output projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape [B, T, D] -> [B, H, T, D/H]"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def compute_attention_scores(self, query_states, key_states):
        """Override this in subclasses for custom attention (e.g., relative pos)."""
        return torch.matmul(query_states, key_states.transpose(-1, -2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,  # <-- add this
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        # QKV projections
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz)

        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # Cache support
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if self.is_decoder else None

        # Flatten for batched matmul: [B, H, T, D] -> [B*H, T, D]
        query_states = query_states.view(bsz * self.num_heads, tgt_len, self.head_dim)
        key_states = key_states.view(bsz * self.num_heads, -1, self.head_dim)
        value_states = value_states.view(bsz * self.num_heads, -1, self.head_dim)

        # Compute attention scores
        attn_weights = self.compute_attention_scores(query_states, key_states)

        # Attention mask (broadcasted properly)
        if attention_mask is not None:
            expected_shape = (bsz, 1, tgt_len, key_states.size(1))
            if attention_mask.size() != expected_shape:
                raise ValueError(f"Expected attention_mask shape {expected_shape}, got {attention_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        # after computing attn_probs shape = [batch_size * num_heads, tgt_len, src_len]
        if head_mask is not None:
            # head_mask shape often [num_heads] or [batch_size, num_heads, 1, 1]
            # so you'd broadcast or reshape it to match [B*num_heads, 1, 1]
            # For simplest approach: shape [num_heads] -> expand to [B*num_heads, 1, 1].
            head_mask = head_mask.unsqueeze(1).unsqueeze(2)  # => [num_heads, 1, 1]
            head_mask = head_mask.expand(attn_probs.size(0), -1, -1)  # => [B*num_heads, 1, 1]
            attn_probs = attn_probs * head_mask

        # Weighted sum over values
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, -1)
            return attn_output, attn_probs, present_key_value
        else:
            return attn_output, None, present_key_value


class TimeSeriesAttention(BaseAttention):
    """
    Standard Multi-Head Attention for Time Series data.
    Inherits all logic from BaseAttention, including:
      - Query/Key/Value projections
      - Attention dropout
      - Optional caching for autoregressive decoding
      - Attention masks and optional output of attention weights
    """

    def __init__(self, config):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=config.is_decoder,  # ✅ Uses the config value instead of hardcoding
            bias=True,  # Bias remains default as True, unless changed in config
        )


def expand_decoder_mask(
    attention_mask: torch.Tensor,
    tgt_len: int,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Convert a 2D [batch_size, src_len] attention mask
    into a 4D [batch_size, 1, tgt_len, src_len] mask
    for Transformer multi-head attention.

    Args:
        attention_mask (Tensor): shape [B, src_len], 
            where 1.0 means "keep" and 0.0 means "mask" (or vice versa).
        tgt_len (int): number of target steps (e.g., 1 if decoding a single step).
        dtype: typically hidden_states.dtype

    Returns:
        A 4D expanded mask of shape [B, 1, tgt_len, src_len]
        with 0.0 where we keep, -inf where we mask (if using the
        typical additive attention approach).
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask of shape [B, src_len], got {attention_mask.shape}.")

    bsz, src_len = attention_mask.shape
    # Expand to [B, 1, 1, src_len]
    expanded_mask = attention_mask[:, None, None, :]  # => [B,1,1,src_len]

    # Expand along target dimension => [B,1,tgt_len,src_len]
    expanded_mask = expanded_mask.expand(bsz, 1, tgt_len, src_len)

    # Convert from [0,1] mask to additive form: 0.0 for keep, -1e9 for masked
    expanded_mask = expanded_mask.to(dtype=dtype)
    inverted_mask = (1.0 - expanded_mask) * -1e9

    return inverted_mask
