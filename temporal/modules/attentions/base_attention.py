import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module


def expand_mask(attention_mask: torch.Tensor, tgt_len: int, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask [B, S], got {attention_mask.shape}")
    bsz, src_len = attention_mask.shape
    expanded = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    expanded = (1.0 - expanded.to(dtype)) * -1e9
    return expanded


class BaseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_decoder=False, is_cross_attention=False, bias=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, x, seq_len, bsz):
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def compute_attention_scores(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2))

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        bsz, tgt_len, _ = hidden_states.size()
        is_cross = key_value_states is not None

        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)

        if is_cross:
            k = self._shape(self.k_proj(key_value_states), -1, bsz)
            v = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            k = self._shape(self.k_proj(hidden_states), -1, bsz)
            v = self._shape(self.v_proj(hidden_states), -1, bsz)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v) if self.is_decoder else None

        q = q.view(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz * self.num_heads, -1, self.head_dim)
        v = v.view(bsz * self.num_heads, -1, self.head_dim)

        attn_weights = self.compute_attention_scores(q, k)

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        if head_mask is not None:
            attn_probs *= head_mask.unsqueeze(1).unsqueeze(2)

        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            return attn_output, attn_probs.view(bsz, self.num_heads, tgt_len, -1), present_key_value
        else:
            return attn_output, None, present_key_value

@register_module("attention", "full")
class DotProductAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_decoder=False, is_cross_attention=False, **kwargs):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=True,
        )

@register_module("attention", "flash")
class FlashAttentionMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_decoder=False, is_cross_attention=False, **kwargs):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, is_cross_attention)
        # check for flash_attn availability
        from flash_attn import flash_attn_func
        self.flash_attn_func = flash_attn_func

    def forward(...):
        ...
