import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from temporal.registry.core import register_module
from temporal.modules.attention.base_attention import BaseMultiHeadAttention


class RotaryProjection(nn.Module):
    def __init__(self, max_len=100):
        super().__init__()
        self.max_len = max_len

    def forward(self, x):
        return x  # no-op placeholder


class QueryKeyProjection(nn.Module):
    def __init__(self, dim, num_heads, proj_layer, proj_kwargs=None, partial_factor=(0.0, 0.5)):
        super().__init__()
        self.proj = proj_layer(**(proj_kwargs or {}))

    def forward(self, x):
        return self.proj(x)


class BinaryAttentionBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

    def forward(self, attn_scores):
        return torch.zeros_like(attn_scores)


def expand_mask(attention_mask, tgt_len, dtype):
    bsz, src_len = attention_mask.shape
    expanded = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    return (1.0 - expanded.to(dtype)) * -1e9


@register_module("attention", "timeseries")
class TimeSeriesAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        max_len: int = 1024,
        scale: Optional[float] = None,
        mask_flag: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, is_cross_attention, bias=bias)
        self.mask_flag = mask_flag
        self.scaling = scale if scale is not None else self.head_dim ** -0.5
        self.qk_proj = QueryKeyProjection(
            dim=self.head_dim,
            num_heads=num_heads,
            proj_layer=RotaryProjection,
            proj_kwargs={"max_len": max_len},
        )
        self.attn_bias = BinaryAttentionBias(embed_dim, num_heads)

    def compute_attention_scores(self, q, k):
        B = q.shape[0] // self.num_heads
        tgt_len = q.shape[1]
        q_4d = q.view(B, self.num_heads, tgt_len, self.head_dim)
        k_4d = k.view(B, self.num_heads, -1, self.head_dim)

        q_proj = self.qk_proj(q_4d)
        k_proj = self.qk_proj(k_4d)

        scores = torch.matmul(q_proj.view(-1, tgt_len, self.head_dim),
                              k_proj.view(-1, k_4d.size(2), self.head_dim).transpose(-1, -2))
        scores += self.attn_bias(scores)
        return scores

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
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(key_value_states if key_value_states is not None else hidden_states)
        v = self.v_proj(key_value_states if key_value_states is not None else hidden_states)

        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, -1, bsz)
        v = self._shape(v, -1, bsz)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present_key_value = (k, v) if self.is_decoder else None

        q = q.view(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz * self.num_heads, -1, self.head_dim)
        v = v.view(bsz * self.num_heads, -1, self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = expand_mask(attention_mask, tgt_len, dtype=q.dtype)
            elif attention_mask.dim() != 4:
                raise ValueError("Attention mask must be 2D or 4D.")

        scores = self.compute_attention_scores(q, k)

        if attention_mask is not None:
            scores = scores.view(bsz, self.num_heads, tgt_len, -1)
            scores += attention_mask
            scores = scores.view(bsz * self.num_heads, tgt_len, -1)

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_probs_4d = attn_probs.view(bsz, self.num_heads, tgt_len, -1) if output_attentions else None
        return attn_output, attn_probs_4d, present_key_value
