import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention, expand_mask


class RotaryProjection(nn.Module):
    """(Replace with real rotary or plug in from utils.)"""
    def __init__(self, max_len=1000):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # ← placeholder


class BinaryAttentionBias(nn.Module):
    """(Replace with learnable or condition-based biasing if needed.)"""
    def __init__(self, dim, num_heads):
        super().__init__()

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(attn_scores)  # ← placeholder


@register_module("attention", "time")
class TimeAttention(BaseMultiHeadAttention):
    """
    Example time-aware attention using optional projection + additive bias.
    """

    def __init__(self, config):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=config.is_decoder,
            bias=config.bias
        )

        self.use_rotary = config.get("use_rotary", False)
        self.use_bias = config.get("use_attention_bias", False)

        self.qk_proj = RotaryProjection() if self.use_rotary else nn.Identity()
        self.attn_bias = BinaryAttentionBias(config.hidden_size, config.num_attention_heads) if self.use_bias else None

    def compute_attention_scores(self, q, k):
        # q, k shape: [B*H, T, D]
        # Apply optional rotary projection inline
        q = self.qk_proj(q)
        k = self.qk_proj(k)
        scores = torch.matmul(q, k.transpose(-1, -2))
        if self.attn_bias is not None:
            scores = scores + self.attn_bias(scores)
        return scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        return super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
