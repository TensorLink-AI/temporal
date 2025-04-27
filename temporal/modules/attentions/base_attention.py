import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None
    print("Warning: flash_attn is not installed. Flash attention will not be available.")

def expand_mask(attention_mask: torch.Tensor, tgt_len: int, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask of shape [B, S], got {attention_mask.shape}.")

    bsz, src_len = attention_mask.shape
    expanded = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    expanded = (1.0 - expanded.to(dtype)) * torch.finfo(dtype).min
    return expanded




class BaseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        **kwargs
    ):
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

    def _reshape_for_heads(self, x, bsz, seq_len):
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, seq_len, self.head_dim)

    def compute_attention_scores(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()
        is_cross = key_value_states is not None

        # === Q, K, V projection ===
        q = self.q_proj(hidden_states)
        k = self.k_proj(key_value_states if is_cross else hidden_states)
        v = self.v_proj(key_value_states if is_cross else hidden_states)

        # === Reshape to [B*H, T, D/H] ===
        q = self._reshape_for_heads(q * self.scaling, bsz, tgt_len)
        k = self._reshape_for_heads(k, bsz, -1)
        v = self._reshape_for_heads(v, bsz, -1)

        # === Handle past key/values ===
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        present_key_value = (k, v) if self.is_decoder else None

        # === Compute attention ===
        attn_scores = self.compute_attention_scores(q, k)

        if attention_mask is not None:
            attn_scores = attn_scores.view(bsz, self.num_heads, tgt_len, -1)
            attn_scores = attn_scores + attention_mask  # mask shape: [B, 1, T, S]
            attn_scores = attn_scores.view(bsz * self.num_heads, tgt_len, -1)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        if head_mask is not None:
            attn_probs *= head_mask.unsqueeze(1).unsqueeze(2)

        # === Compute final output ===
        attn_output = torch.bmm(attn_probs, v)  # [B*H, T, D/H]
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, -1)
            return attn_output, attn_probs, present_key_value
        else:
            return attn_output, None, present_key_value


# ------------------------------------------------------------------
# ✅ Implementation 1: Standard Dot Product Attention
# ------------------------------------------------------------------

@register_module("attention", "full")
class FullAttention(BaseMultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
@register_module("attention", "flash")
class FlashAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_decoder=False, is_cross_attention=False, bias=True, **kwargs):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, is_cross_attention, bias, **kwargs)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        if flash_attn_func is None:
            raise ImportError("flash_attn_func is not available.")

        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(key_value_states if key_value_states is not None else hidden_states)
        v = self.v_proj(key_value_states if key_value_states is not None else hidden_states)

        # [B, T, H, D] → flash_attn expects [B, T, H, D]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        # flash_attn_func: expects [B, T, H, D]
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scaling,
            causal=self.is_decoder
        )

        out = out.view(B, T, self.embed_dim)
        return self.out_proj(out), None, None  # no attn_probs returned

