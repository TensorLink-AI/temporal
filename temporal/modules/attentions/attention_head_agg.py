import torch
from torch import nn
from typing import List, Optional
import torch
from torch import nn
from typing import List, Optional

class HeadAggregator(nn.Module):
    """
    Extended head aggregator supporting multiple approaches:

    'gated'         : per-head gating (sigmoid)
    'attention'     : single query attention over heads
    'std'        : concatenation + linear projection (similar to standard MHA)
    'weighted_mean' : learnable scalar weights per head
    'head2head'     : meta-attention across heads
    'moe'           : mixture-of-experts style gating across heads
    'low_rank'      : low-rank factorization to fuse heads
    'small_mlp'     : small MLP to combine heads
    'se'            : squeeze-and-excitation across heads
    """

    def __init__(
        self,
        method: str = "mean",
        hidden_size: Optional[int] = None,  # Usually dimension of each head output (Q)
        num_heads: Optional[int] = None,
        output_size: Optional[int] = None,  # If method does final projection
        # Extra parameters for advanced methods:
        aggregator_attn_heads: int = 4,     # For head2head multi-head attention
        moe_hidden_size: int = 64,          # Hidden size for MoE gating MLP
        low_rank_dim: int = 64,            # Rank for low_rank aggregator
        mlp_hidden_factor: float = 2.0,     # MLP hidden factor -> hidden = factor * output_size
    ):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size

        # --------------------
        # Existing methods
        # --------------------
        if method == "gated":
            assert hidden_size is not None, "Gated aggregation requires hidden_size"
            self.gate = nn.Linear(hidden_size, 1)

        elif method == "attention":
            # Single query for "attention over heads"
            assert hidden_size is not None and num_heads is not None, \
                "Attention requires hidden_size and num_heads"
            self.query = nn.Parameter(torch.randn(hidden_size))
            self.softmax = nn.Softmax(dim=0)

        elif method == "std":
            assert output_size is not None and num_heads is not None, \
                "Fusion requires num_heads and output_size"
            self.head_fusion = nn.Linear(num_heads * output_size, output_size)

        elif method == "weighted_mean":
            assert num_heads is not None, "Weighted mean requires num_heads"
            self.head_weights = nn.Parameter(torch.ones(num_heads))
            self.softmax = nn.Softmax(dim=0)

        # --------------------
        # NEW methods
        # --------------------
        elif method == "head2head":
            """
            Meta-attention across the head dimension.
            We'll treat each head's output as a 'token' in a small multi-head attention block.
            Suppose each head output has shape (B, T, hidden_size).
            We will combine them into (B*T, H, hidden_size) and run multi-head self-attention.
            """
            assert hidden_size is not None, \
                "head2head aggregator requires hidden_size"
            self.aggregator_attn_heads = aggregator_attn_heads
            self.head2head_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=aggregator_attn_heads,
                batch_first=True
            )

        elif method == "moe":
            """
            Mixture-of-Experts aggregator over heads:
            For each (batch, time) step, produce gating across heads.
            We'll do a small 2-layer MLP gating for demonstration.
            If you want top-k gating or something discrete, you'd need
            additional logic (and possibly a non-differentiable step).
            """
            assert hidden_size is not None and num_heads is not None, \
                "moe aggregator requires hidden_size and num_heads"
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size, moe_hidden_size),
                nn.ReLU(),
                nn.Linear(moe_hidden_size, 1)  # 1 gating scalar per head
            )
            self.softmax = nn.Softmax(dim=0)

        elif method == "low_rank":
            """
            Low-rank factorization for combining heads:
            1. Concatenate heads -> (B, T, num_heads*output_size)
            2. Linear(W1) -> (B, T, low_rank_dim)
            3. Linear(W2) -> (B, T, output_size)
            """
            assert output_size is not None and num_heads is not None, \
                "low_rank aggregator requires num_heads and output_size"
            self.low_rank_projection1 = nn.Linear(num_heads * output_size, low_rank_dim, bias=False)
            self.low_rank_projection2 = nn.Linear(low_rank_dim, output_size, bias=False)

        elif method == "small_mlp":
            """
            Pass concatenated heads (B, T, num_heads*Q) through a small MLP
            to get shape (B, T, Q) again.
            """
            assert output_size is not None and num_heads is not None, \
                "small_mlp aggregator requires num_heads and output_size"
            hidden_dim = int(mlp_hidden_factor * output_size)
            self.mlp = nn.Sequential(
                nn.Linear(num_heads * output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_size),
            )

        elif method == "se":
            """
            Squeeze-and-Excitation style aggregator across heads:
            1. Squeeze heads into a global descriptor -> (H)
            2. MLP -> gating vector of shape (H)
            3. Scale each head by gating, sum across heads
            """
            assert num_heads is not None and hidden_size is not None, \
                "se aggregator requires num_heads and hidden_size"
            # For simplicity, we do a single MLP from num_heads->num_heads or hidden_size->hidden_size
            # We'll do gating across heads: shape (H).
            # If you prefer gating each hidden_size channel, you'd do a different shape.
            self.se_mlp = nn.Sequential(
                nn.Linear(num_heads, num_heads // 2),
                nn.ReLU(),
                nn.Linear(num_heads // 2, num_heads)
            )
            self.sigmoid = nn.Sigmoid()

        else:
            raise ValueError(f"Unknown head aggregation method: {method}")

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        head_outputs: List of length H (num_heads), each of shape (B, T, Q)
        returns:      (B, T, Q) aggregated
        """
        # (H, B, T, Q)
        stacked = torch.stack(head_outputs, dim=0)

        # --------------------
        # Original methods
        # --------------------

        if self.method == "gated":
            # gates: (H, B, T, 1)
            gates = torch.stack([self.gate(h) for h in head_outputs], dim=0)
            weights = torch.sigmoid(gates)          # (H, B, T, 1)
            weighted = stacked * weights
            return weighted.sum(dim=0) / (weights.sum(dim=0) + 1e-8)

        elif self.method == "attention":
            # shape (H, B, T, Q)
            scores = torch.stack([
                torch.einsum("btq,q->bt", h, self.query) for h in head_outputs
            ])  # (H, B, T)
            weights = self.softmax(scores).unsqueeze(-1)  # (H, B, T, 1)
            weighted = stacked * weights
            return weighted.sum(dim=0)

        elif self.method == "std":
            # (B, T, Q * H)
            fused_input = torch.cat(head_outputs, dim=-1)
            return self.head_fusion(fused_input)

        elif self.method == "weighted_mean":
            weights = self.softmax(self.head_weights)  # (H,)
            weighted = stacked * weights.view(-1, 1, 1, 1)
            return weighted.sum(dim=0)

        # --------------------
        # New methods
        # --------------------
        elif self.method == "head2head":
            # Flatten (B, T) into one "batch" dimension so we can do self-attention across H
            # stacked: (H, B, T, Q)
            H, B, T, Q = stacked.shape
            x = stacked.permute(1,2,0,3).reshape(B*T, H, Q)  # (B*T, H, Q)
            # We do "self-attention across heads" -> Key/Value are the same:
            attn_out, _ = self.head2head_attn(x, x, x)      # (B*T, H, Q)
            # Example aggregator: average along the "head" dimension:
            out = attn_out.mean(dim=1)  # (B*T, Q)
            # Reshape back to (B, T, Q)
            return out.view(B, T, Q)

        elif self.method == "moe":
            # stacked: (H, B, T, Q)
            # We'll do a gating for each head, for each (B, T).
            # 1) Squeeze each head-output from (B, T, Q) -> (B, T, Q) => we apply an MLP along Q
            # 2) Then we do a softmax across heads
            # For simplicity, let's do gating(h) by averaging Q -> shape (B, T, 1)
            # Then pass into the gate_net -> shape (B, T, 1). In practice you might do more.
            H, B, T, Q = stacked.shape
            # gates_i = self.gate_net( mean over Q ) => shape (B, T, 1)
            # Collect gating for each head in a single tensor: (H, B, T, 1)
            gating = []
            for i in range(H):
                # average across Q => shape (B, T, 1)
                h_mean = head_outputs[i].mean(dim=-1, keepdim=True)  # (B, T, 1)
                g = self.gate_net(h_mean)                            # (B, T, 1)
                gating.append(g)
            gating = torch.stack(gating, dim=0)  # (H, B, T, 1)

            # Softmax over heads (dim=0) => get gating weights (H, B, T, 1)
            weights = self.softmax(gating)
            # Weighted sum
            weighted = stacked * weights
            return weighted.sum(dim=0)  # (B, T, Q)

        elif self.method == "low_rank":
            # stacked: (H, B, T, Q)
            # 1) Permute to (B, T, H, Q)
            # 2) reshape -> (B, T, H*Q)
            # 3) linear -> (B, T, low_rank_dim)
            # 4) linear -> (B, T, Q)
            H, B, T, Q = stacked.shape
            fused_input = stacked.permute(1, 2, 0, 3).reshape(B, T, H*Q)  # (B, T, H*Q)
            out = self.low_rank_projection1(fused_input)                 # (B, T, low_rank_dim)
            out = self.low_rank_projection2(out)                         # (B, T, Q)
            return out

        elif self.method == "small_mlp":
            # stacked: (H, B, T, Q)
            H, B, T, Q = stacked.shape
            fused_input = stacked.permute(1, 2, 0, 3).reshape(B, T, H*Q)  # (B, T, H*Q)
            out = self.mlp(fused_input)                                  # (B, T, Q)
            return out

        elif self.method == "se":
            """
            Squeeze heads (H, B, T, Q) -> we produce a gating factor per head.
            We'll do a global average across (B, T, Q), leaving shape (H,).
            Then feed that into se_mlp -> produce gating (H,).
            Multiply each head by gating[h], then sum across heads.
            """
            H, B, T, Q = stacked.shape
            # (H, B, T, Q) => average across B, T, Q => shape(H,)
            squeeze = stacked.mean(dim=(1,2,3))  # (H,)
            scale = self.se_mlp(squeeze)         # (H,)
            scale = self.sigmoid(scale)
            # shape(H,) -> reshape(H,1,1,1)
            scaled = stacked * scale.view(H, 1, 1, 1)
            return scaled.sum(dim=0)  # (B, T, Q)

        else:
            raise ValueError(f"Unknown head aggregation method: {self.method}")
