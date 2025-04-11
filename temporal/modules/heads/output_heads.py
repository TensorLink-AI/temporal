from torch import nn
from temporal.registry.core import register_module

@register_module("output_head", "gaussian")
class GaussianHead(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.out = nn.Linear(hidden_size, 2)  # mean and log_std

    def forward(self, x):
        return self.out(x)  # shape: [B, T, 2]
import torch
import torch.nn as nn
from temporal.registry.core import register_module


# ============================================================
# 1. Standard Linear Output Head
# ============================================================

@register_module("output_head", "linear")
class LinearOutputHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs):
        super().__init__()
        self.proj = n output_sizen.Linear(hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # shape: [B, T, Q]

    def get_loss_fn(self):
        return TimeSeriesLoss(loss_type=self.loss_type)

# ============================================================
# 2. T-Distribution Output Head
# ============================================================

@register_module("output_head", "t_distribution")
class TDistributionHead(BaseOutputHead):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)  # Outputs: μ, log(σ), log(ν)

    def forward(self, x):
        return self.fc(x)

    def get_loss_fn(self):
        return TDistributionLoss()



# ============================================================
# 3. Multi-Quantile Output Head
# ============================================================

@register_module("output_head", "multi_quantile")
class MultiQuantileHead(BaseOutputHead):
    def __init__(self, input_dim, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.fc = nn.Linear(input_dim, len(quantiles))

    def forward(self, x):
        return self.fc(x)

    def get_loss_fn(self):
        return QuantileLoss(self.quantiles)
