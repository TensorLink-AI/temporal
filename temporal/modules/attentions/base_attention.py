import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

def expand_mask(
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
            where 1.0 => keep and 0.0 => mask (or vice versa).
        tgt_len (int): number of target steps (e.g. 1).
        dtype: typically hidden_states.dtype

    Returns:
        4D expanded mask => [B, 1, tgt_len, src_len]
        0.0 => keep, -1e9 => mask in additive form
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask of shape [B, src_len], got {attention_mask.shape}.")

    bsz, src_len = attention_mask.shape
    # shape => [B, 1, 1, src_len]
    expanded_mask = attention_mask[:, None, None, :]
    # shape => [B, 1, tgt_len, src_len]
    expanded_mask = expanded_mask.expand(bsz, 1, tgt_len, src_len)
    expanded_mask = expanded_mask.to(dtype=dtype)
    inverted_mask = (1.0 - expanded_mask) * -1e9
    return inverted_mask

class BaseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        bias: bool = True,
        is_cross_attention: bool = False,

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention


        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.scaling = self.head_dim ** -0.5

        # QKV + output projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """ Reshape [B, T, D] -> [B, H, T, D/H] """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def compute_attention_scores(self, query_states, key_states):
        return torch.matmul(query_states, key_states.transpose(-1, -2))

    def forward(
        self,
        hidden_states: torch.Tensor,           # shape [B, tgt_len, embed_dim]
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None, # can be 2D or 4D
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        # 1) Q, K, V
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz)

        if is_cross_attention:
            # shape => [B, src_len, embed_dim], then => [B, H, src_len, head_dim]
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # 2) Past key-value (caching)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if self.is_decoder else None

        # 3) Flatten for matmul
        query_states = query_states.view(bsz * self.num_heads, tgt_len, self.head_dim)
        key_states = key_states.view(bsz * self.num_heads, -1, self.head_dim)
        value_states = value_states.view(bsz * self.num_heads, -1, self.head_dim)

        # 4) attention logits => shape [B*H, tgt_len, src_len]
        attn_weights = self.compute_attention_scores(query_states, key_states)

        # 5) Expand or apply attention_mask
        #    If mask is 2D, expand to 4D. If it's already 4D, do nothing.


        if attention_mask is not None:
            # Just verify it's already 4D
            if attention_mask.dim() != 4:
                raise ValueError(
                    f"Attention mask must be 4D [B,1,tgt_len,src_len], got {attention_mask.shape}"
                )

            expected_shape = (bsz, 1, tgt_len, key_states.size(1))
            if attention_mask.shape != expected_shape:
                raise ValueError(
                    f"Expected attention_mask shape {expected_shape}, got {attention_mask.shape}"
                )

            # shape => [B,H,tgt_len,src_len]
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)


        # 6) softmax & dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 7) optional head_mask
        if head_mask is not None:
            # shape [num_heads] or [B, num_heads, 1, 1], must expand to match attn_probs
            head_mask = head_mask.unsqueeze(1).unsqueeze(2)  # => [num_heads, 1, 1]
            head_mask = head_mask.expand(attn_probs.size(0), -1, -1)
            attn_probs = attn_probs * head_mask

        # 8) Weighted sum
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 9) Return
        if output_attentions:
            # reshape attn_probs => [B, H, tgt_len, src_len] for interpretability
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, -1)
            return attn_output, attn_probs, present_key_value
        else:
            return attn_output, None, present_key_value


class DotProductAttention(BaseMultiHeadAttention):
    """
    Standard Multi-Head Attention for Time Series data.
    Inherits logic from BaseAttention, but sets up from config.
    """
    def __init__(self, config):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=config.is_decoder,
            bias=True
        )



class FlashAttentionMultiHeadAttention(BaseMultiHeadAttention):
    """
    Multi-head attention using FlashAttention for a fused, efficient
    matmul+softmax+dropout kernel. 
    - Does *not* show cross-attention or outputting attention weights 
      (which is trickier to do with FlashAttention).
    - Minimal example to illustrate the approach.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        if flash_attn_func is None:
            raise ImportError("flash_attn is not installed or could not be imported.")

        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        # 1) Project Q, K, V
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # 2) Reshape => [B, T, H, D]
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        # 3) Past key-value
        # Note: Handling past_key_value caching with FlashAttention is non-trivial
        # because FlashAttn expects contiguous segments. This snippet omits that.
        present_key_value = None
        if past_key_value is not None:
            raise NotImplementedError(
                "past_key_value + FlashAttention not fully implemented in this snippet."
            )

        # 4) If you have an attention mask, convert it to a boolean or additive mask
        #    that FlashAttention supports. For standard "padding" masks, FlashAttention
        #    can handle them via `flash_attn_func` arguments or by zeroing out Q/K as needed.
        #    Typically it wants shape [B, T] or a causal flag. We'll do a minimal check:
        causal = False  # set True if you want causal (decoder) attention
        if attention_mask is not None:
            # Often you pass a boolean key_padding_mask or a block mask to flash_attn_func.
            # If it's an additive mask, you'd need to convert it. Minimal example:
            # shape => [B, 1, T_q, T_k], 0 => keep, -1e9 => mask
            # We'll just say if there's any -1e9, we treat that token as "masked."
            bool_mask = (attention_mask >= 0).squeeze(1)  # => [B, T_q, T_k]
        else:
            bool_mask = None

        # 5) Transpose for FlashAttn => [B, T, H, D] is usually accepted directly by flash_attn_func.
        #    (Check your version if it needs [B, H, T, D].)

        # 6) Call fused flash attention
        #    dropout_p => typically the same as self.dropout (in training mode).
        #    softmax_scale => optional, can set to self.scaling if you want:
        #                     scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,   # or 1.0 / math.sqrt(self.head_dim)
            causal=causal,
            key_padding_mask=bool_mask,  # If using a padding mask
        )
        # attn_output => [B, T, H, D]

        # 7) Reshape => [B, T, H, D] -> [B, T, H*D]
        attn_output = attn_output.reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        # 8) [Optional] head_mask not trivially integrated into flash_attn. 
        #    Typically you'd have to re-implement a portion of the kernel or do a separate pass.

        # 9) Return 
        #    FlashAttention doesn't natively return attn_probs. 
        #    Some advanced variants let you extract them, but that undermines memory savings.
        if output_attentions:
            # Not straightforward to get attn_probs from flash_attn without extra overhead.
            return attn_output, None, present_key_value
        else:
            return attn_output, None, present_key_value