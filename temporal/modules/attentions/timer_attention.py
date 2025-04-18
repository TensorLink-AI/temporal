import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention


class TimeMoeRotaryEmbedding(nn.Module):
    """
    Caches cos/sin rotary embeddings for position IDs.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_pos = max_position_embeddings

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])  # [1, 1, max_len, D]
        self.register_buffer("sin", emb.sin()[None, None, :, :])  # [1, 1, max_len, D]

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:, :, position_ids, :], self.sin[:, :, position_ids, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


@register_module("attention", "timer")
class TimerAttention(BaseMultiHeadAttention):
    """
    Time-aware attention using rotary position embeddings (RoPE).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        max_position_embeddings: int = 2048,
        **kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
        )

        self.rope = TimeMoeRotaryEmbedding(self.head_dim, max_position_embeddings)

    def compute_attention_scores(self, q, k):
        # Shape: q, k = [B*H, T, D]
        return torch.matmul(q, k.transpose(-1, -2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ):
        B, T, _ = hidden_states.shape
        device = hidden_states.device
        position_ids = position_ids if position_ids is not None else torch.arange(T, device=device).unsqueeze(0)

        # === Projections
        q = self.q_proj(hidden_states)
        k_input = key_value_states if key_value_states is not None else hidden_states
        k = self.k_proj(k_input)
        v = self.v_proj(k_input)

        # === Reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # === Rotary Embedding
        cos, sin = self.rope(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # === Flatten for base attention
        q = q.transpose(1, 2).reshape(B * self.num_heads, T, self.head_dim)
        k = k.transpose(1, 2).reshape(B * self.num_heads, -1, self.head_dim)

        return super().forward(
            hidden_states=hidden_states,
            key_value_states=None,  # We already used k/v directly
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
