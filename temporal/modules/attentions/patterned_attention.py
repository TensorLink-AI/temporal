import torch
import torch.nn as nn
from typing import Optional, Tuple

from temporal.configs.attention_config import PatternedAttentionConfig
from temporal.modules.attentions.base_attention import FullAttention
from temporal.registry.core import register_module


def combine_masks(
    patt_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Combines the pattern mask with an optional existing attention mask."""
    if attention_mask is None:
        # The pattern mask is boolean and additive (True means attend).
        # Attention scores are typically floats where -inf means "do not attend".
        # So, we convert our boolean mask to the correct float format.
        # True -> 0.0, False -> -inf
        return torch.where(patt_mask, 0.0, -torch.inf)
    
    # The provided attention_mask is already in the float format (-inf for masked positions).
    # We add our pattern mask to it.
    patt_mask_float = torch.where(patt_mask, 0.0, -torch.inf)
    return patt_mask_float + attention_mask


@register_module("attention", "patterned")
class PatternedMultiHeadAttention(FullAttention):
    """
    Multi-head attention that applies a fixed, predefined pattern (e.g., local,
    sliding, dilated) to the attention matrix.

    This class inherits from `FullAttention`, giving it full support for RoPE,
    ALiBi, and other features of the standard attention mechanism. It extends
    it by generating a pattern-based mask that is combined with any other
    masks (like causal or padding masks) during the forward pass.
    """
    def __init__(self, cfg: PatternedAttentionConfig):
        # Initialize the parent `FullAttention` with all relevant parameters from the config.
        super().__init__(
            embed_dim=cfg.d_model, # Assuming d_model is passed by the builder
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_layernorm=cfg.qk_layernorm,
            use_rope=cfg.use_rope,
            use_alibi=cfg.use_alibi,
            max_position_embeddings=cfg.max_position_embeddings,
            rope_base=cfg.rope_base,
        )
        p = cfg.pattern
        self.pattern_type = p.type
        self.window_size = p.window_size
        self.stride = p.stride or 1
        self.global_indices = p.global_indices or []

    def compute_pattern_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates the boolean attention mask based on the configured pattern."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            # Sliding and Local Patterns
            if self.pattern_type in ("sliding", "local"):
                # 'local' is a sliding window centered on the query
                half = self.window_size // 2 if self.pattern_type == "sliding" else 0
                start = max(0, i - half)
                end = min(seq_len, start + self.window_size)
                mask[i, start:end] = True
            
            # Dilated Pattern
            if self.pattern_type == "dilated":
                # Attend to every 'stride'-th token
                for j in range(0, seq_len, self.stride):
                    mask[i, j] = True
            
            # Global Indices (always attend to/from these tokens)
            for g in self.global_indices:
                if 0 <= g < seq_len:
                    mask[i, g] = True
                    mask[g, i] = True
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Overrides the forward pass to inject the pattern mask.
        
        1. Generates the pattern mask based on the sequence length.
        2. Combines it with any existing attention mask (e.g., for padding or causality).
        3. Calls the parent `FullAttention.forward` method with the combined mask.
        """
        B, T, _ = hidden_states.size()
        
        # 1. Generate the pattern mask.
        patt_mask = self.compute_pattern_mask(T, device=hidden_states.device)
        
        # 2. Combine with any user-provided mask.
        # The resulting mask will be of shape [T, T] or broadcastable.
        full_mask = combine_masks(patt_mask, attention_mask)
        
        # Add dimensions to make it broadcastable with attention scores: [B, H, T, T]
        # The parent forward expects a mask that can be added to scores [B, H, T_q, T_k]
        if full_mask.dim() == 2:
            full_mask = full_mask.unsqueeze(0).unsqueeze(0) # -> [1, 1, T, T]
        elif full_mask.dim() == 3:
            full_mask = full_mask.unsqueeze(1) # -> [B, 1, T, T]

        # 3. Call the parent's forward method with the new, combined mask.
        return super().forward(hidden_states=hidden_states, attention_mask=full_mask, **kwargs)
