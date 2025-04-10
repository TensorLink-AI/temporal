import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attention.base_attention import BaseMultiHeadAttention
from temporal.modules.norm.rmsnorm import RMSNorm  # or define one inline


def lambda_init_fn(depth: int):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(x.size(0), x.size(1) * n_rep, x.size(2), x.size(3))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True):
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    x1_rot = x1 * cos[..., :half] - x2 * sin[..., :half]
    x2_rot = x2 * cos[..., :half] + x1 * sin[..., :half]
    return torch.cat([x1_rot, x2_rot], dim=-1)


@register_module("attention", "diffwist")
class DiffWistAttentionWithCache(BaseMultiHeadAttention):
    """
    Registry-compatible DiffWist attention layer with gating, rotary, and caching.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        depth: int = 1,
        num_kv_heads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=False
        )

        self.depth = depth
        self.num_kv_heads = num_kv_heads or num_heads
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = self.embed_dim // self.num_heads // 2

        # Redefine projections to support gated and reduced dimension
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_pos: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ):
        B, T, _ = hidden_states.shape
        cos, sin = rel_pos

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, T, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, 2 * self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, 2 * self.head_dim)

        # Rotary
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose + repeat
        q = q.transpose(1, 2)                         # [B, 2*H, T, D]
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # [B, 2*H, T, D]
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # [B, 2*H, T, 2D]

        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present_key_value = (k, v)

        q = q * (self.head_dim ** -0.5)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)

        # Gating: split heads into [num_heads, 2, T, S]
        attn_weights = attn_weights.view(B, self.num_heads, 2, T, -1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        gated = attn_weights[:, :, 0] - (lambda_1 - lambda_2 + self.lambda_init) * attn_weights[:, :, 1]

        # Weighted sum
        v = v.view(B, self.num_heads, 2, -1, self.head_dim)
        attn_output = torch.einsum("bhts,bhstd->bhtd", gated, v)
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_probs = gated if output_attentions else None
        return attn_output, attn_probs, present_key_value
