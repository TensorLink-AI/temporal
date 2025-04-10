import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from temporal.registry.core import register_module


# =========================
# Mean Aggregator
# =========================
@register_module("head_agg", "mean")
class MeanAggregator(nn.Module):
    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(head_outputs, dim=0).mean(dim=0)


# =========================
# Gated Aggregator
# =========================
@register_module("head_agg", "gated")
class GatedAggregator(nn.Module):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(head_outputs, dim=0)
        gates = torch.stack([self.gate(h) for h in head_outputs], dim=0)
        weights = torch.sigmoid(gates)
        return (stacked * weights).sum(dim=0) / (weights.sum(dim=0) + 1e-8)


# =========================
# Weighted Mean Aggregator
# =========================
@register_module("head_agg", "weighted_mean")
class WeightedMeanAggregator(nn.Module):
    def __init__(self, num_heads: int, **kwargs):
        super().__init__()
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(head_outputs, dim=0)
        weights = self.softmax(self.head_weights)
        weighted = stacked * weights.view(-1, 1, 1, 1)
        return weighted.sum(dim=0)


# =========================
# Squeeze-and-Excitation Aggregator
# =========================
@register_module("head_agg", "se")
class SEAggregator(nn.Module):
    def __init__(self, num_heads: int, **kwargs):
        super().__init__()
        self.se_mlp = nn.Sequential(
            nn.Linear(num_heads, max(1, num_heads // 2)),
            nn.ReLU(),
            nn.Linear(max(1, num_heads // 2), num_heads)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(head_outputs, dim=0)  # (H, B, T, Q)
        squeeze = stacked.mean(dim=(1, 2, 3))        # (H,)
        scale = self.sigmoid(self.se_mlp(squeeze))  # (H,)
        scaled = stacked * scale.view(-1, 1, 1, 1)
        return scaled.sum(dim=0)


# =========================
# Mixture-of-Experts Aggregator
# =========================
@register_module("head_agg", "moe")
class MoEAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, moe_hidden_size: int = 64, **kwargs):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, moe_hidden_size),
            nn.ReLU(),
            nn.Linear(moe_hidden_size, 1)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(head_outputs, dim=0)  # (H, B, T, Q)
        gating = []
        for h in head_outputs:
            h_mean = h.mean(dim=-1, keepdim=True)
            g = self.gate_net(h_mean)
            gating.append(g)
        gating = torch.stack(gating, dim=0)         # (H, B, T, 1)
        weights = self.softmax(gating)
        return (stacked * weights).sum(dim=0)

        # =========================
# Head2Head Aggregator
# =========================
@register_module("head_agg", "head2head")
class Head2HeadAggregator(nn.Module):
    def __init__(self, hidden_size: int, aggregator_attn_heads: int = 4, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=aggregator_attn_heads,
            batch_first=True
        )

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        # stacked: (H, B, T, Q)
        H, B, T, Q = torch.stack(head_outputs, dim=0).shape
        x = torch.stack(head_outputs, dim=0).permute(1, 2, 0, 3).reshape(B * T, H, Q)  # (B*T, H, Q)
        out, _ = self.attn(x, x, x)  # self-attention over heads
        out = out.mean(dim=1)        # average over heads
        return out.view(B, T, Q)


# =========================
# Low-Rank Aggregator
# =========================
@register_module("head_agg", "low_rank")
class LowRankAggregator(nn.Module):
    def __init__(self, num_heads: int, output_size: int, low_rank_dim: int = 64, **kwargs):
        super().__init__()
        self.low_rank_projection1 = nn.Linear(num_heads * output_size, low_rank_dim, bias=False)
        self.low_rank_projection2 = nn.Linear(low_rank_dim, output_size, bias=False)

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        H, B, T, Q = torch.stack(head_outputs, dim=0).shape
        fused_input = torch.stack(head_outputs, dim=0).permute(1, 2, 0, 3).reshape(B, T, H * Q)  # (B, T, H*Q)
        return self.low_rank_projection2(self.low_rank_projection1(fused_input))


# =========================
# Small MLP Aggregator
# =========================
@register_module("head_agg", "small_mlp")
class SmallMLPAggregator(nn.Module):
    def __init__(self, num_heads: int, output_size: int, mlp_hidden_factor: float = 2.0, **kwargs):
        super().__init__()
        hidden_dim = int(mlp_hidden_factor * output_size)
        self.mlp = nn.Sequential(
            nn.Linear(num_heads * output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        H, B, T, Q = torch.stack(head_outputs, dim=0).shape
        fused = torch.stack(head_outputs, dim=0).permute(1, 2, 0, 3).reshape(B, T, H * Q)  # (B, T, H*Q)
        return self.mlp(fused)
