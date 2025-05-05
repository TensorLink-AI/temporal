# Modified temporal/modules/attentions/hybrid_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from temporal.registry.core import register_module, resolve
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention # Use base for standard args

# Assume attention kernels registered under "attention" expect standard BaseMultiHeadAttention args
# Assume head aggregators registered under "head_agg" expect specific inputs


@register_module("attention", "hybrid")
class HybridAttention(nn.Module): # Inherit from nn.Module
    """
    Multi-head attention with per-group attention kernel types.

    Each group of heads uses a different attention kernel, and the group outputs
    are fused either by concatenation + projection or by a configured head aggregator.
    """

    def __init__(
        self,
        embed_dim: int, # Required base arg from builder
        num_heads: int, # Required base arg from builder
        dropout: float = 0.1, # Optional base arg from builder
        is_decoder: bool = False, # Optional base arg from builder
        is_cross_attention: bool = False, # Optional base arg from builder
        bias: bool = True, # Optional base arg from builder
        head_splits: Optional[List[int]] = None, # Hybrid specific
        head_types: Optional[List[str]] = None, # Hybrid specific
        head_agg: str = "concat", # Hybrid specific: 'concat' or registered aggregator key
        head_agg_kwargs: Optional[dict] = None, # Hybrid specific: kwargs for aggregator
        # Capture other potential base args or user kwargs
        **kwargs
    ):
        super().__init__()
        if not head_splits or not head_types:
             raise ValueError("HybridAttention requires 'head_splits' and 'head_types'.")
        if sum(head_splits) != num_heads:
            raise ValueError(f"Sum of head_splits ({sum(head_splits)}) must equal num_heads ({num_heads}).")
        if len(head_splits) != len(head_types):
             raise ValueError("Length of head_splits must match length of head_types.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        self.head_splits = head_splits
        self.head_types = head_types
        self.group_count = len(head_splits)

        if embed_dim % num_heads != 0:
             raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) for HybridAttention.")
        self.head_dim = embed_dim // num_heads

        # Projections (Standard MHA style)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Output projection is only used if head_agg is 'concat'
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias) if head_agg == "concat" else None

        # Group-specific attention kernels
        self.head_groups = nn.ModuleList()
        for split, attn_type in zip(head_splits, head_types):
            # --- MODIFICATION START ---
            # Calculate the embedding dimension for this specific group based on
            # the global head_dim and the number of heads in this group (split).
            group_embed_dim = self.head_dim * split
            # --- MODIFICATION END ---
            try:
                kernel_cls = resolve("attention", attn_type) # Resolve standard attention modules
                # Pass standard MHA arguments. The kernel's __init__ should accept these.
                kernel = kernel_cls(
                    # --- MODIFICATION START ---
                    embed_dim=group_embed_dim, # Pass the calculated group embed_dim
                    # --- MODIFICATION END ---
                    num_heads=split, # Pass the number of heads for THIS group
                    dropout=dropout,
                    is_decoder=is_decoder,
                    is_cross_attention=is_cross_attention,
                    bias=bias,
                    **kwargs # Pass along any other kwargs
                )
            except Exception as e:
                print(f"Error resolving/instantiating attention kernel '{attn_type}' for HybridAttention group: {e}")
                raise e
            self.head_groups.append(kernel)

        # Aggregation logic
        self.head_agg = head_agg
        self.head_aggregator = None
        if self.head_agg == "concat":
            self.fuse = self._concat_fuse
        else:
            try:
                 # Resolve the head aggregator module
                 agg_cls = resolve("head_agg", head_agg)
                 # Instantiate the aggregator. Its __init__ must handle necessary args.
                 # Common aggregators might need embed_dim, num_groups, etc.
                 self.head_aggregator = agg_cls(
                      embed_dim=embed_dim, # Provide embed_dim for potential use
                      num_groups=self.group_count, # Inform aggregator about number of inputs
                      **(head_agg_kwargs or {})
                 )
                 self.fuse = self._aggregate_fuse
            except Exception as e:
                 print(f"Error resolving/instantiating head aggregator '{head_agg}': {e}")
                 raise e

    def _concat_fuse(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Input head_outputs: list of [B, H_group, T, D_head] from each kernel
        all_heads = torch.cat(head_outputs, dim=1)  # [B, H, T, D_head]
        # Transpose and reshape for output projection
        # [B, H, T, D_head] -> [B, T, H, D_head] -> [B, T, D]
        out = all_heads.transpose(1, 2).contiguous().view(all_heads.shape[0], all_heads.shape[2], self.embed_dim)
        if self.out_proj is None:
             # Should not happen if head_agg is 'concat', but safety check
             raise RuntimeError("Output projection is None for concat fuse mode.")
        return self.out_proj(out)

    def _aggregate_fuse(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Input head_outputs: list of [B, H_group, T, D_head]
        # The aggregator module is responsible for handling this list input.
        if self.head_aggregator is None:
            raise RuntimeError("Head aggregator is None but aggregation requested.")
        # Pass the list directly to the aggregator's forward method
        return self.head_aggregator(head_outputs)

    def forward(
        self,
        hidden_states: torch.Tensor,               # Query: [B, T, D]
        key_value_states: Optional[torch.Tensor] = None, # Key/Value source for cross-attn: [B, S, D]
        past_key_value: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None, # List per group, inner tuple can be None
        attention_mask: Optional[torch.Tensor] = None, # Additive mask: [B, 1, T, S]
        head_mask: Optional[torch.Tensor] = None, # Not used in this implementation
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]], Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:

        B, T, D = hidden_states.shape
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        S = kv_source.size(1)

        # Project Q, K, V globally first
        q_proj = self.q_proj(hidden_states) # [B, T, D]
        k_proj = self.k_proj(kv_source)     # [B, S, D]
        v_proj = self.v_proj(kv_source)     # [B, S, D]

        # Reshape Q, K, V for multi-head processing [B, H, T/S, D_head]
        q_heads = q_proj.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_proj.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_proj.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Slice Q, K, V and route to appropriate attention kernel groups
        group_outputs = []
        all_attn_probs = [] if output_attentions else None
        present_key_values = [] if use_cache else None
        head_offset = 0

        for i, (split, kernel) in enumerate(zip(self.head_splits, self.head_groups)):
            # Slice Q, K, V for the current group
            q_slice = q_heads[:, head_offset : head_offset + split] # [B, H_group, T, D_head]
            k_slice = k_heads[:, head_offset : head_offset + split] # [B, H_group, S, D_head]
            v_slice = v_heads[:, head_offset : head_offset + split] # [B, H_group, S, D_head]

            # Handle past_key_value for this group if provided
            group_past_kv = None
            if use_cache and past_key_value is not None and i < len(past_key_value):
                 group_past_kv = past_key_value[i]

            # Forward pass through the group's attention kernel
            # Assuming kernel follows standard (output, attn_probs, present_kv) signature
            # Note: The kernel receives Q/K/V already split by heads.
            # Its internal logic should handle these [B, H_group, T/S, D_head] inputs.
            # We pass the *full* attention_mask, the kernel might need to adjust/ignore it if needed.
            kernel_result = kernel(
                 hidden_states=q_slice, # Pass Q-slice
                 key_value_states=k_slice if is_cross_attn else None, # Pass K-slice if cross-attn
                 # If kernel *needs* separate K, V inputs based on BaseMultiHeadAttention structure:
                 # query=q_slice,
                 # key=k_slice, # Need to adjust base class forward or kernel forward
                 # value=v_slice,
                 attention_mask=attention_mask,
                 past_key_value=group_past_kv,
                 output_attentions=output_attentions,
                 use_cache=use_cache,
            )

            # Extract results from the kernel's output tuple
            group_attn_output = kernel_result[0] # Expected shape [B, H_group, T, D_head]
            group_outputs.append(group_attn_output)

            if output_attentions:
                 # Ensure kernel actually returned attn_probs
                 attn_prob = kernel_result[1] if len(kernel_result) > 1 else None
                 all_attn_probs.append(attn_prob) # Store attn_probs [B, H_group, T, S] or None
            if use_cache:
                 # Ensure kernel actually returned present_kv
                 present_kv = kernel_result[2] if len(kernel_result) > 2 else None
                 present_key_values.append(present_kv) # Store present_kv tuple or None

            head_offset += split

        # Fuse the outputs from all groups using the selected method
        final_output = self.fuse(group_outputs) # [B, T, D]

        # Return combined results
        # Note: attn_probs and present_key_values are lists (one element per group)
        return final_output, all_attn_probs, present_key_values
