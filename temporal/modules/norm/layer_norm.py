import torch.nn as nn
from temporal.registry.core import register_module




@register_module("normalization", "layer")
class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        """
        Wrapper around torch.nn.LayerNorm with registry support.

        Args:
            normalized_shape (int or list): usually hidden_size
            eps (float): epsilon for numerical stability
            elementwise_affine (bool): whether to learn scale and shift
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

