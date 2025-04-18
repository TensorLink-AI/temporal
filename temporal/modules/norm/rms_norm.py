import torch
import torch.nn as nn
from temporal.registry.core import register_module


@register_module("normalization", "rms")
import torch
import torch.nn as nn
from temporal.registry.core import register_module


@register_module("normalization", "rms")
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True, **kwargs):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, ..., D] where D = normalized_shape
        Returns:
            Normalized tensor of same shape as x
        """
        # Compute root mean square norm over last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms
        if self.weight is not None:
            out = out * self.weight
        return out
