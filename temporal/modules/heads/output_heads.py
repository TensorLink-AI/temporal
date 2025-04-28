import torch
import torch.nn as nn

from temporal.registry.core import register_module
from temporal.modules.heads.base_output_head import BaseOutputHead
from temporal.modules.losses.losses import TimeSeriesLoss


# -------------------------------------------------------
# ✅ 1. LinearOutputHead
# -------------------------------------------------------

@register_module("output_head", "linear")
class LinearOutputHead(BaseOutputHead):
    def __init__(self, hidden_size: int, output_size: int = 1, loss_type: str = "mse", **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size, output_size)
        self.loss_type = loss_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_loss_fn(self):
        return TimeSeriesLoss(loss_type=self.loss_type)


# -------------------------------------------------------
# ✅ 2. GaussianHead
# -------------------------------------------------------

@register_module("output_head", "gaussian")
class GaussianHead(BaseOutputHead):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 2)  # mean and log std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_loss_fn(self):
        return TimeSeriesLoss(loss_type="gaussian")  # optional: use GaussianNLLLoss directly


# -------------------------------------------------------
# ✅ 3. TDistributionHead
# -------------------------------------------------------

@register_module("output_head", "t_distribution")
class TDistributionHead(BaseOutputHead):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 3)  # μ, log σ, log ν

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_loss_fn(self):
        return TDistributionLoss()


# -------------------------------------------------------
# ✅ 4. MultiQuantileHead
# -------------------------------------------------------

@register_module("output_head", "multi_quantile")
class MultiQuantileHead(BaseOutputHead):
    def __init__(self, hidden_size: int, quantiles: list = [0.1, 0.5, 0.9], **kwargs):
        super().__init__()
        self.quantiles = quantiles
        self.proj = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_loss_fn(self):
        return QuantileLoss(quantiles=self.quantiles)
