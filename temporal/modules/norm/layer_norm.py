import torch.nn as nn
from temporal.registry.core import register_module

@register_module("normalization", "layer")
class LayerNorm(nn.LayerNorm):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(normalized_shape=kwargs.get("normalized_shape", None), eps=eps)
