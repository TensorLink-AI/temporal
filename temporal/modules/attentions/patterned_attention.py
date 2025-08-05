import torch
import torch.nn as nn
from typing import Optional

from temporal.configs.attention_config import PatternedAttentionConfig
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.registry.core import register_module


def combine_masks(patt_mask: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is None:
        return patt_mask
    return patt_mask & attention_mask


@register_module("attention", "patterned")
class PatternedMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, cfg: PatternedAttentionConfig):
        super().__init__(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_layernorm=cfg.qk_layernorm,
        )
        p = cfg.pattern
        self.pattern_type = p.type
        self.window_size = p.window_size
        self.stride = p.stride or 1
        self.global_indices = p.global_indices or []

    def compute_pattern_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=self.q_proj.weight.device)
        for i in range(seq_len):
            if self.pattern_type in ("sliding", "local"):
                half = self.window_size // 2 if self.pattern_type == "sliding" else 0
                start = max(0, i - half)
                end = min(seq_len, start + self.window_size)
                mask[i, start:end] = True
            if self.pattern_type == "dilated":
                for j in range(0, seq_len, self.stride):
                    mask[i, j] = True
            for g in self.global_indices:
                mask[i, g] = True
                mask[g, i] = True
        return mask

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, T, E = hidden_states.size()
        
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        patt_mask = self.compute_pattern_mask(T)
        full_mask = combine_masks(patt_mask, attention_mask)
        
        return self.forward_with_mask(q, k, v, full_mask, **kwargs)
