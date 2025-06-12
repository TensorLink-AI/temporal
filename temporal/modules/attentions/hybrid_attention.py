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
class HybridAttention(nn.Module):
    """Implements a hybrid multi-head attention mechanism.

    This module allows different groups of attention heads to use different
    attention kernel implementations (e.g., 'full', 'flash'). The outputs from
    these heterogeneous groups are then fused together to produce a single
    output tensor.

    The fusion can be a simple concatenation followed by a linear projection,
    or a more complex, learnable aggregation module.

    Attributes:
        embed_dim (int): The total embedding dimension.
        num_heads (int): The total number of attention heads.
        head_dim (int): The dimension of each individual attention head.
        head_splits (List[int]): A list specifying the number of heads in each group.
        head_types (List[str]): A list of strings specifying the attention kernel
            type for each group.
        group_count (int): The number of attention groups.
        q_proj (nn.Linear): The global query projection layer.
        k_proj (nn.Linear): The global key projection layer.
        v_proj (nn.Linear): The global value projection layer.
        out_proj (Optional[nn.Linear]): The final output projection layer, used
            only when `head_agg` is 'concat'.
        head_groups (nn.ModuleList): A list of the instantiated attention kernel
            modules for each group.
        head_aggregator (Optional[nn.Module]): The instantiated head aggregation module.
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
        """Initializes the HybridAttention module.

        Args:
            embed_dim (int): The total embedding dimension.
            num_heads (int): The total number of attention heads.
            dropout (float): The dropout rate for the attention kernels.
            is_decoder (bool): Flag indicating if the module is used in a decoder.
            is_cross_attention (bool): Flag indicating if it's a cross-attention module.
            bias (bool): Whether to use a bias in the projection layers.
            head_splits (Optional[List[int]]): A list defining the number of heads
                for each attention group. The sum must equal `num_heads`.
            head_types (Optional[List[str]]): A list of registered attention kernel
                keys, one for each group.
            head_agg (str): The method for aggregating group outputs. Can be 'concat'
                or a key for a registered 'head_agg' module.
            head_agg_kwargs (Optional[dict]): A dictionary of keyword arguments to
                pass to the head aggregator's constructor.
            **kwargs: Additional arguments passed to the attention kernel constructors.
        """
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
        """Fuses head outputs by concatenation and a linear projection.

        Args:
            head_outputs (List[torch.Tensor]): A list of output tensors from each
                attention group. Each tensor has the shape
                `[batch, group_heads, seq_len, head_dim]`.

        Returns:
            torch.Tensor: The fused output tensor of shape `[batch, seq_len, embed_dim]`.
        """
        all_heads = torch.cat(head_outputs, dim=1)  # [B, H, T, D_head]
        # Transpose and reshape for output projection
        # [B, H, T, D_head] -> [B, T, H, D_head] -> [B, T, D]
        fused_output = all_heads.transpose(1, 2).contiguous().view(all_heads.shape[0], all_heads.shape[2], self.embed_dim)
        if self.out_proj is None:
             # Should not happen if head_agg is 'concat', but safety check
             raise RuntimeError("Output projection is None for concat fuse mode.")
        return self.out_proj(fused_output)

    def _aggregate_fuse(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Fuses head outputs using a registered head aggregation module.

        Args:
            head_outputs (List[torch.Tensor]): A list of output tensors from each
                attention group. Each tensor has the shape
                `[batch, group_heads, seq_len, head_dim]`.

        Returns:
            torch.Tensor: The fused output tensor. The shape depends on the
                aggregator implementation but is typically
                `[batch, seq_len, embed_dim]`.
        """
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
        """Performs the forward pass for the HybridAttention module.

        Args:
            hidden_states (torch.Tensor): The query tensor of shape `[B, T, D]`.
            key_value_states (Optional[torch.Tensor]): The key/value source tensor
                for cross-attention, shape `[B, S, D]`. If None, self-attention is performed.
            past_key_value (Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]):
                A list of cached key-value states, one for each attention group.
                Used for efficient decoding.
            attention_mask (Optional[torch.Tensor]): An additive mask applied to the
                attention scores, shape `[B, 1, T, S]`.
            head_mask (Optional[torch.Tensor]): A mask for heads. Not used in this
                implementation.
            output_attentions (bool): If True, returns the attention probabilities
                for each group.
            use_cache (bool): If True, returns the updated key-value states for caching.

        Returns:
            Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]], Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
                - The final fused attention output tensor of shape `[B, T, D]`.
                - A list of attention probabilities from each group (if `output_attentions`).
                - A list of updated key-value states from each group (if `use_cache`).
        """
        B, T, D = hidden_states.shape
        is_cross_attn = key_value_states is not None
        key_value_source = key_value_states if is_cross_attn else hidden_states
        S = key_value_source.size(1)

        # Project Q, K, V globally first
        projected_q = self.q_proj(hidden_states) # [B, T, D]
        projected_k = self.k_proj(key_value_source)     # [B, S, D]
        projected_v = self.v_proj(key_value_source)     # [B, S, D]

        # Reshape Q, K, V for multi-head processing [B, H, T/S, D_head]
        q_by_heads = projected_q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_by_heads = projected_k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v_by_heads = projected_v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Slice Q, K, V and route to appropriate attention kernel groups
        group_attention_outputs = []
        all_group_attention_probs = [] if output_attentions else None
        all_group_present_key_values = [] if use_cache else None
        head_offset = 0

        for i, (split, kernel) in enumerate(zip(self.head_splits, self.head_groups)):
            # Slice Q, K, V for the current group
            q_slice = q_by_heads[:, head_offset : head_offset + split] # [B, H_group, T, D_head]
            k_slice = k_by_heads[:, head_offset : head_offset + split] # [B, H_group, S, D_head]
            v_slice = v_by_heads[:, head_offset : head_offset + split] # [B, H_group, S, D_head]

            # Handle past_key_value for this group if provided
            group_past_key_value = None
            if use_cache and past_key_value is not None and i < len(past_key_value):
                 group_past_key_value = past_key_value[i]

            # Forward pass through the group's attention kernel
            # Assuming kernel follows standard (output, attn_probs, present_kv) signature
            # Note: The kernel receives projected Q/K/V that are already split by heads.
            # This implementation assumes the sub-kernel's `forward` method can handle
            # `hidden_states` being passed as `q_slice` and `key_value_states` as `k_slice`.
            # This differs from a standard BaseMultiHeadAttention which expects raw hidden states.
            # We pass the *full* attention_mask, the kernel might need to adjust/ignore it if needed.
            group_kernel_result = kernel(
                 hidden_states=q_slice, # Pass Q-slice
                 key_value_states=k_slice if is_cross_attn else None, # Pass K-slice if cross-attn
                 # If kernel *needs* separate K, V inputs based on BaseMultiHeadAttention structure:
                 # query=q_slice,
                 # key=k_slice, # Need to adjust base class forward or kernel forward
                 # value=v_slice,
                 attention_mask=attention_mask,
                 past_key_value=group_past_key_value,
                 output_attentions=output_attentions,
                 use_cache=use_cache,
            )

            # Extract results from the kernel's output tuple
            group_output_tensor = group_kernel_result[0] # Expected shape [B, H_group, T, D_head]
            group_attention_outputs.append(group_output_tensor)

            if output_attentions:
                 # Ensure kernel actually returned attn_probs
                 group_probs_tensor = group_kernel_result[1] if len(group_kernel_result) > 1 else None
                 all_group_attention_probs.append(group_probs_tensor) # Store attn_probs [B, H_group, T, S] or None
            if use_cache:
                 # Ensure kernel actually returned present_kv
                 group_kv_cache = group_kernel_result[2] if len(group_kernel_result) > 2 else None
                 all_group_present_key_values.append(group_kv_cache) # Store present_kv tuple or None

            head_offset += split

        # Fuse the outputs from all groups using the selected method
        fused_output = self.fuse(group_attention_outputs) # [B, T, D]

        # Return combined results
        return fused_output, all_group_attention_probs, all_group_present_key_values
