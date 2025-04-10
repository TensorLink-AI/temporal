import torch
import torch.nn as nn
from temporal.registry.core import register_module


@register_module("normalization", "rms")
class RMSNorm(nn.Module):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(kwargs.get("normalized_shape", 1)))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)
