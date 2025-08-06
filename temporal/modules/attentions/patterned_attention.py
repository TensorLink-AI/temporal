import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from temporal.configs.attention_config import PatternedAttentionConfig, AttentionPatternConfig
from temporal.modules.attentions.base_attention import FullAttention
from temporal.registry.core import register_module


def combine_masks(
    patt_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Combines the pattern mask with an optional existing attention mask."""
    # Convert the boolean pattern mask to the float format expected by PyTorch's attention mechanism.
    # True (attend) becomes 0.0, and False (do not attend) becomes -infinity.
    patt_mask_float = torch.where(patt_mask, 0.0, -torch.inf)

    if attention_mask is None:
        return patt_mask_float
    
    # The provided attention_mask is already in the float format.
    # We add our pattern mask to it; adding -inf to any position masks it.
    return patt_mask_float + attention_mask


@register_module("attention", "patterned")
class PatternedMultiHeadAttention(FullAttention):
    """
    Multi-head attention that applies a fixed, predefined pattern (e.g., local,
    sliding, dilated) to the attention matrix.

    This class inherits from `FullAttention`, giving it full support for RoPE,
    ALiBi, and other features. It extends `FullAttention` by generating a
    pattern-based mask that is combined with any other masks during the forward pass.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        pattern: Dict,  # The builder passes the nested config as a dict
        dropout: float = 0.1,
        bias: bool = True,
        qk_layernorm: bool = False,
        use_rope: bool = False,
        use_alibi: bool = False,
        max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        is_causal: bool = False,
        **kwargs,
    ):
        # Initialize the parent `FullAttention` with all relevant parameters.
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            qk_layernorm=qk_layernorm,
            use_rope=use_rope,
            use_alibi=use_alibi,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
        )
        # Instantiate the pattern config from the dictionary.
        p = AttentionPatternConfig.from_dict(pattern)
        self.pattern_type = p.type
        self.window_size = p.window_size
        self.stride = p.stride or 1
        self.dilation = p.dilation or 1
        self.global_indices = p.global_indices or []
        self.is_causal = is_causal

    def compute_pattern_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates the boolean attention mask based on the configured pattern."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            # --- STRIDING LOGIC ---
            # If the query index 'i' is not on a stride boundary, skip it.
            if self.stride > 1 and i % self.stride != 0:
                continue
            
            # Sliding and Local Patterns
            if self.pattern_type in ("sliding", "local"):
                half = self.window_size // 2 if self.pattern_type == "sliding" else 0
                start = max(0, i - half)
                end = min(seq_len, start + self.window_size)
                # Apply dilation within the window
                for j in range(start, end, self.dilation):
                    mask[i, j] = True
            
            # Dilated Pattern (Global Strided Attention)
            if self.pattern_type == "dilated":
                # Use stride for global, fixed-step attention
                for j in range(0, seq_len, self.dilation):
                    mask[i, j] = True
            
            # Global Indices
            for g in self.global_indices:
                if 0 <= g < seq_len:
                    mask[i, g] = True
                    mask[g, i] = True
        if self.is_causal:
            # Apply a causal mask to prevent attention to future positions
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            mask = mask & causal_mask
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Overrides the forward pass to inject the pattern mask.
        """
        B, T, _ = hidden_states.size()
        
        patt_mask = self.compute_pattern_mask(T, device=hidden_states.device)
        
        full_mask = combine_masks(patt_mask, attention_mask)
        
        if full_mask.dim() == 2:
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        elif full_mask.dim() == 3:
            full_mask = full_mask.unsqueeze(1)

        return super().forward(hidden_states=hidden_states, attention_mask=full_mask, **kwargs)
