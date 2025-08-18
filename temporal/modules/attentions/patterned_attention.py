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
            """Generates the boolean attention mask using efficient vectorized operations."""
            # Create tensors representing row and column indices for vectorized comparisons
            row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
            col_indices = torch.arange(seq_len, device=device).unsqueeze(0)

            # Generate the primary pattern mask based on type
            if self.pattern_type == "sliding":
                half_window = self.window_size // 2
                # A key is valid if its column index is within half_window of the query's row index
                mask = torch.abs(row_indices - col_indices) <= half_window

            elif self.pattern_type == "local":
                # A key is valid if its index is before the query's and within the window size
                distance = row_indices - col_indices
                mask = (distance >= 0) & (distance < self.window_size)
            
            elif self.pattern_type == "dilated":
                # Corrected: A key is valid if its distance from the query is a multiple of dilation
                distance = torch.abs(row_indices - col_indices)
                mask = (distance % self.dilation == 0)

            else:
                raise ValueError(f"Unknown pattern type: '{self.pattern_type}'")

            # Apply global tokens: they can see all tokens, and all tokens can see them
            if self.global_indices:
                global_indices_tensor = torch.tensor(self.global_indices, device=device, dtype=torch.long)
                mask[global_indices_tensor, :] = True
                mask[:, global_indices_tensor] = True
            
            # Apply causality on top of the existing pattern if required
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
        """
        Overrides the forward pass to inject the pattern mask.
        The mask is created to ensure no rows are entirely masked out, preventing NaNs.
        """
        B, T, _ = hidden_states.size()
        
        patt_mask = self.compute_pattern_mask(T, device=hidden_states.device)
        full_mask = combine_masks(patt_mask, attention_mask)
        
        if full_mask.dim() == 2:
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        elif full_mask.dim() == 3:
            full_mask = full_mask.unsqueeze(1)

        return super().forward(hidden_states=hidden_states, attention_mask=full_mask, **kwargs)
