class HeadAggregator(nn.Module):
    def __init__(
        self,
        method: str = "mean",
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        output_size: Optional[int] = None,  # Q
    ):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if method == "gated":
            assert hidden_size is not None, "Gated aggregation requires hidden_size"
            self.gate = nn.Linear(hidden_size, 1)

        elif method == "attention":
            assert hidden_size is not None and num_heads is not None, "Attention requires hidden_size and num_heads"
            self.query = nn.Parameter(torch.randn(hidden_size))
            self.softmax = nn.Softmax(dim=0)

        elif method == "fusion":
            assert output_size is not None and num_heads is not None, "Fusion requires num_heads and output_size (Q)"
            self.head_fusion = nn.Linear(num_heads * output_size, output_size)  # Using head_fusion instead of fusion



        elif method == "weighted_mean":
            assert num_heads is not None, "Weighted mean requires num_heads"
            self.head_weights = nn.Parameter(torch.ones(num_heads))  # Learnable weights for heads
            self.softmax = nn.Softmax(dim=0)  # Normalize weights

        elif method == "mean":
            pass  # Simple mean

        else:
            raise ValueError(f"Unknown head aggregation method: {method}")

    def forward(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(head_outputs, dim=0)  # (H, B, T, Q)

        if self.method == "mean":
            return stacked.mean(dim=0)  # (B, T, Q)

        elif self.method == "gated":
            gates = torch.stack([self.gate(h) for h in head_outputs], dim=0)  # (H, B, T, 1)
            weights = torch.sigmoid(gates)
            weighted = stacked * weights
            return weighted.sum(dim=0) / (weights.sum(dim=0) + 1e-8)

        elif self.method == "attention":
            scores = torch.stack([
                torch.einsum("btq,q->bt", h, self.query) for h in head_outputs
            ])  # (H, B, T)
            weights = self.softmax(scores).unsqueeze(-1)  # (H, B, T, 1)
            weighted = stacked * weights
            return weighted.sum(dim=0)

        elif self.method == "fusion":
            fused_input = torch.cat(head_outputs, dim=-1)  # (B, T, Q * H)
            return self.head_fusion(fused_input)  # (B, T, Q)

        elif self.method == "weighted_mean":
            weights = self.softmax(self.head_weights)  # Normalize weights (H,)
            weighted = stacked * weights.view(-1, 1, 1, 1)  # Apply weights (H, B, T, Q)
            return weighted.sum(dim=0)  # (B, T, Q)

        else:
            raise ValueError(f"Unknown head aggregation method: {self.method}")
