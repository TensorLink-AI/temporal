# Modified temporal/modules/attentions/hybrid_attention.py
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from temporal.registry.core import register_module, resolve

@register_module("attention", "hybrid")
class HybridAttention(nn.Module):
    """
    Implements a hybrid multi-head attention mechanism.

    This module allows different groups of attention heads to use different
    attention kernel implementations (e.g., 'full', 'flash'). The outputs from
    these heterogeneous groups are then fused together to produce a single
    output tensor.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        head_splits: Optional[List[int]] = None,
        head_types: Optional[List[str]] = None,
        head_agg: str = "concat",
        head_agg_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        if not head_splits or not head_types:
            raise ValueError("HybridAttention requires 'head_splits' and 'head_types'.")
        if sum(head_splits) != num_heads:
            raise ValueError(f"Sum of head_splits ({sum(head_splits)}) must equal num_heads ({num_heads}).")
        if len(head_splits) != len(head_types):
            raise ValueError("Length of head_splits must match length of head_types.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.head_splits = head_splits
        self.group_count = len(head_splits)

        # This list will store the embedding dimension for each group.
        self.group_embed_dims = [self.head_dim * split for split in head_splits]
        
        # The output projection is always needed for the 'concat' aggregator.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias) if head_agg == "concat" else None

        # Group-specific attention kernels
        self.head_groups = nn.ModuleList()
        for i, (split, attn_type) in enumerate(zip(head_splits, head_types)):
            group_embed_dim = self.group_embed_dims[i]
            try:
                kernel_cls = resolve("attention", attn_type)
                kernel = kernel_cls(
                    embed_dim=group_embed_dim,
                    num_heads=split,
                    dropout=dropout,
                    is_decoder=is_decoder,
                    is_cross_attention=is_cross_attention,
                    bias=bias,
                    **kwargs
                )
                self.head_groups.append(kernel)
            except Exception as e:
                print(f"Error resolving/instantiating attention kernel '{attn_type}': {e}")
                raise e

        # Aggregation logic
        self.head_agg = head_agg
        self.head_aggregator = None
        if self.head_agg != "concat":
            try:
                agg_cls = resolve("head_agg", head_agg)
                self.head_aggregator = agg_cls(
                    embed_dim=embed_dim,
                    num_groups=self.group_count,
                    **(head_agg_kwargs or {})
                )
            except Exception as e:
                print(f"Error resolving/instantiating head aggregator '{head_agg}': {e}")
                raise e

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None, # Not used
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]], Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
        
        is_cross_attn = key_value_states is not None
        key_value_source = key_value_states if is_cross_attn else hidden_states
        
        # --- MAJOR FIX START ---
        # Split the input tensors along the feature dimension for each group.
        # This provides each sub-module with the 3D tensor it expects.
        hidden_states_split = torch.split(hidden_states, self.group_embed_dims, dim=-1)
        key_value_states_split = torch.split(key_value_source, self.group_embed_dims, dim=-1)
        # --- MAJOR FIX END ---
        
        group_outputs = []
        all_attn_probs = [] if output_attentions else None
        all_present_key_values = [] if use_cache else None

        for i, kernel in enumerate(self.head_groups):
            group_hidden_state = hidden_states_split[i]
            group_kv_state = key_value_states_split[i] if is_cross_attn else None
            group_past_kv = past_key_value[i] if past_key_value else None

            # Pass the 3D tensor chunks to the respective kernels
            output, attn_probs, present_kv = kernel(
                hidden_states=group_hidden_state,
                key_value_states=group_kv_state,
                past_key_value=group_past_kv,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            group_outputs.append(output)
            
            if output_attentions:
                all_attn_probs.append(attn_probs)
            if use_cache:
                all_present_key_values.append(present_kv)

        # Fuse the outputs from all groups
        if self.head_agg == "concat":
            fused_output = torch.cat(group_outputs, dim=-1)
            fused_output = self.out_proj(fused_output)
        else:
            # Assumes the custom aggregator knows how to handle the list of tensors
            fused_output = self.head_aggregator(group_outputs)
            
        return fused_output, all_attn_probs if output_attentions else None, all_present_key_values if use_cache else None