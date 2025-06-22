
import torch
import torch.nn as nn

from temporal.models.mixin.adaptive_patching import AdaptivePatching, PatchMerging
from temporal.registry import register_block


@register_block("adaptive_patch_transformer")
class AdaptivePatchTransformerBlock(nn.Module):
    """
    An adaptive patch transformer block that applies adaptive patching before a transformer layer
    and merges patches after.

    This block is generic and can wrap any module (e.g., an encoder or decoder layer)
    that operates on sequence data.

    Args:
        transformer_layer (nn.Module): The transformer layer (encoder, decoder, etc.) to apply.
        expansion_factor (int): The factor to expand patches and reduce feature dim.
    """

    def __init__(self, transformer_layer: nn.Module, expansion_factor: int):
        super().__init__()
        self.adaptive_patching = AdaptivePatching(expansion_factor)
        self.transformer_layer = transformer_layer
        self.patch_merging = PatchMerging(expansion_factor)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D].
            **kwargs: Additional keyword arguments to be passed to the inner transformer layer.

        Returns:
            torch.Tensor: The output tensor, with the same shape as the input [B, N, D].
        """
        # Note: The inner `transformer_layer` might return a tuple (e.g., with attentions).
        # We assume the primary tensor output is the first element.
        x_patched = self.adaptive_patching(x)
        
        # Pass patched input and any other arguments to the wrapped layer
        layer_output = self.transformer_layer(x_patched, **kwargs)

        # Handle both Tensor and tuple outputs from the inner layer
        if isinstance(layer_output, tuple):
            x_processed = layer_output[0]
            other_outputs = layer_output[1:]
            
            # Merge the patches back
            merged_output = self.patch_merging(x_processed)
            return (merged_output,) + other_outputs
        else:
            x_processed = layer_output
            merged_output = self.patch_merging(x_processed)
            return merged_output
