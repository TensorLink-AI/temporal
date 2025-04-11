from temporal.registry.core import resolve

class HybridMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        head_splits,
        head_types,
        dropout=0.1,
        head_agg: Optional[str] = "concat",
        head_agg_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        ...
        self.head_splits = head_splits
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_groups = nn.ModuleList([
            resolve("attention_kernel", t)(
                num_heads=h, head_dim=self.head_dim, dropout=dropout
            )
            for h, t in zip(head_splits, head_types)
        ])

        # === NEW: Aggregation logic
        if head_agg == "concat":
            self.fuse_heads = self._concat_project
        else:
            agg_cls = resolve("head_agg", head_agg)
            self.head_aggregator = agg_cls(
                hidden_size=self.head_dim,
                num_heads=sum(head_splits),
                output_size=embed_dim,
                **(head_agg_kwargs or {})
            )
            self.fuse_heads = self._aggregate_heads

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _concat_project(self, head_outputs: list[torch.Tensor]):
        # [B, H, T, D] → [B, T, embed_dim]
        concat = torch.cat(head_outputs, dim=1)
        return self.out_proj(concat.transpose(1, 2).reshape(concat.shape[0], concat.shape[2], -1))

    def _aggregate_heads(self, head_outputs: list[torch.Tensor]):
        # reshape each to [B, T, D]
        per_head = [h.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[2], -1) for h in head_outputs]
        return self.head_aggregator(per_head)

    def forward(self, hidden_states, attention_mask=None):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        head_outputs = []
        offset = 0
        for attn, h in zip(self.head_groups, self.head_splits):
            q_split = q[:, offset:offset+h]
            k_split = k[:, offset:offset+h]
            v_split = v[:, offset:offset+h]
            out = attn(q_split, k_split, v_split, attention_mask)
            head_outputs.append(out)
            offset += h

        return self.fuse_heads(head_outputs)
