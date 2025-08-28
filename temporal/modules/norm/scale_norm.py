import torch
import torch.nn as nn
from temporal.registry.core import register_module


@register_module("normalization", "scale")
class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        return self.g * x / (norm + self.eps)
