import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from temporal.configs.attention_config import PatternedAttentionConfig, AttentionPatternConfig
from temporal.modules.attentions.base_attention import FullAttention
from temporal.registry.core import register_module


def combine_masks(
    patt_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Combines the pattern mask with an optional existing attention mask."""
    patt_mask_float = torch.where(patt_mask, 0.0, -torch.inf)

    if attention_mask is None:
        return patt_mask_float
    
    return patt_mask_float + attention_mask


@register_module("attention", "patterned")
class PatternedMultiHeadAttention(FullAttention):
    """
    Multi-head attention that applies a fixed, predefined pattern (e.g., local,
    sliding, dilated) to the attention matrix.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        pattern: Dict[str, Any],
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        **kwargs,
    ):
        # Pass all relevant arguments, including kwargs, to the parent class.
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            **kwargs,
        )
        
        # Instantiate the pattern config from the dictionary.
        p = AttentionPatternConfig.from_dict(pattern)
        self.pattern_type = p.type
        self.window_size = p.window_size
        self.stride = p.stride or 1
        self.dilation = p.dilation or 1
        self.global_indices = p.global_indices or []
        
        # is_causal can be passed via kwargs or inferred
        self.is_causal = kwargs.get('is_causal', is_decoder and not is_cross_attention)

    def compute_pattern_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates the boolean attention mask based on the configured pattern."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            if self.stride > 1 and i % self.stride != 0:
                continue
            
            if self.pattern_type in ("sliding", "local"):
                half = self.window_size // 2 if self.pattern_type == "sliding" else 0
                start = max(0, i - half)
                end = min(seq_len, start + self.window_size)
                for j in range(start, end, self.dilation):
                    mask[i, j] = True
            
            if self.pattern_type == "dilated":
                for j in range(0, seq_len, self.dilation):
                    mask[i, j] = True
            
            for g in self.global_indices:
                if 0 <= g < seq_len:
                    mask[i, g] = True
                    mask[g, i] = True
        
        if self.is_causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            mask = mask & causal_mask
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Overrides the forward pass to inject the pattern mask."""
        B, T, _ = hidden_states.size()
        
        patt_mask = self.compute_pattern_mask(T, device=hidden_states.device)
        full_mask = combine_masks(patt_mask, attention_mask)
        
        if full_mask.dim() == 2:
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        elif full_mask.dim() == 3:
            full_mask = full_mask.unsqueeze(1)

        return super().forward(hidden_states=hidden_states, attention_mask=full_mask, **kwargs)