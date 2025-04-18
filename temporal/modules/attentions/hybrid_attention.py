import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from temporal.registry.core import register_module, resolve


@register_module("attention", "hybrid")
class HybridMultiHeadAttention(nn.Module):
    """
    Multi-head attention with per-group attention kernel types.

    Each group of heads uses a different attention kernel, and the group outputs
    are fused either by concatenation + projection or by a configured head aggregator.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        head_splits: Optional[List[int]] = None,
        head_types: Optional[List[str]] = None,
        head_agg: str = "concat",
        head_agg_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        assert head_splits and head_types, "Must provide head_splits and head_types"
        assert sum(head_splits) == num_heads, "head_splits must sum to num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_splits = head_splits
        self.head_types = head_types
        self.head_dim = embed_dim // num_heads
        self.group_count = len(head_splits)

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Group-specific attention kernels
        self.head_groups = nn.ModuleList([
            resolve("attention_kernel", attn_type)(
                num_heads=split,
                head_dim=self.head_dim,
                dropout=dropout
            )
            for split, attn_type in zip(head_splits, head_types)
        ])

        # Aggregation logic
        if head_agg == "concat":
            self.fuse = self._concat_fuse
            self.head_aggregator = None
        else:
            agg_cls = resolve("head_agg", head_agg)
            self.head_aggregator = agg_cls(
                input_size=self.head_dim * split,  # use flattened group
                num_heads=self.group_count,
                output_size=self.embed_dim,
                **(head_agg_kwargs or {})
            )
            self.fuse = self._aggregate_fuse

    def _concat_fuse(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        # head_outputs: list of [B, H_group, T, D_head]
        all_heads = torch.cat(head_outputs, dim=1)  # [B, H, T, D]
        out = all_heads.transpose(1, 2).reshape(all_heads.shape[0], all_heads.shape[2], -1)  # [B, T, H*D]
        return self.out_proj(out)

    def _aggregate_fuse(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Flatten each group: [B, H_group, T, D] → [B, T, H_group*D]
        flattened = [
            h.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[2], -1)
            for h in head_outputs
        ]
        return self.head_aggregator(flattened)  # expects list[B, T, D]

    def forward(
        self,
        hidden_states: torch.Tensor,               # [B, T, D]
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:

        B, T, _ = hidden_states.shape
        k_input = key_value_states if key_value_states is not None else hidden_states

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(k_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(k_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Slice and route per head group
        group_outputs = []
        offset = 0
        for split, attn in zip(self.head_splits, self.head_groups):
            q_slice = q[:, offset:offset+split]  # [B, split, T, D]
            k_slice = k[:, offset:offset+split]
            v_slice = v[:, offset:offset+split]
            out = attn(q_slice, k_slice, v_slice, mask=attention_mask)
            group_outputs.append(out)
            offset += split

        return self.fuse(group_outputs)  # [B, T, D]
