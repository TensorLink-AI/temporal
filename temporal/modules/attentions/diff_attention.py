import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.modules.norm.rms_norm import RMSNorm



def lambda_init_fn(depth: int):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads for grouped-query attention.

    Args:
        x: [B, H_kv, T, D]
        n_rep: repetition factor to match query heads

    Returns:
        Tensor of shape [B, H_q, T, D] where H_q = H_kv * n_rep
    """
    B, H, T, D = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]                  # [B, H, 1, T, D]
        .expand(B, H, n_rep, T, D)           # [B, H, n_rep, T, D]
        .reshape(B, H * n_rep, T, D)         # [B, H_q, T, D]
    )
def reshape_for_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    [B, T, D] → [B, H, T, D/H]
    """
    B, T, D = x.shape
    head_dim = D // num_heads
    return x.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to a query or key tensor.

    Args:
        x: Tensor of shape [B, H, T, D]
        cos, sin: [T, D] or [B, H, T, D] broadcastable

    Returns:
        Tensor of same shape as x
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Rotary embedding requires even dimension, got {x.shape[-1]}")

    D = x.shape[-1]
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    cos = cos[..., :half]
    sin = sin[..., :half]

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)




@register_module("attention", "diffwist")
class DiffWistAttentionWithCache(BaseMultiHeadAttention):
    """
    DiffWist attention: rotary + grouped query + learnable gating with caching.
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
            bias=False,
        )

        self.depth = depth
        self.num_kv_heads = num_kv_heads or num_heads
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2

        # Redefine projections with modified dims
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
        hidden_states: torch.Tensor,                     # [B, T, D]
        rel_pos: Tuple[torch.Tensor, torch.Tensor],      # (cos, sin) for rotary
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ):
        B, T, _ = hidden_states.shape
        cos, sin = rel_pos

        # Project and reshape
        q = self.q_proj(hidden_states).view(B, T, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, 2 * self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, 2 * self.head_dim)

        # Apply rotary
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose + repeat
        q = q.transpose(1, 2)                         # [B, 2H, T, D]
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # [B, 2H, T, D]
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # [B, 2H, T, 2D]

        # Caching
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present_key_value = (k, v)

        # Attention
        q = q * (self.head_dim ** -0.5)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # [B, 2H, T, T_k]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=0.0)

        # Gating
        attn_weights = attn_weights.view(B, self.num_heads, 2, T, -1)
        λ1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        λ2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        gated = attn_weights[:, :, 0] - (λ1 - λ2 + self.lambda_init) * attn_weights[:, :, 1]

        # Weighted sum
        v = v.view(B, self.num_heads, 2, -1, self.head_dim)
        attn_output = torch.einsum("bhts,bhstd->bhtd", gated, v)

        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, gated if output_attentions else None, present_key_value
