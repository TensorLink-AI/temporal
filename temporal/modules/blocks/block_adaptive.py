
import torch
import torch.nn as nn

from temporal.models.mixin.adaptive_patching import AdaptivePatching, PatchMerging
from temporal.registry.core import register_module


@register_module("block", "adaptive_patch_transformer")
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
        self.expansion_factor = expansion_factor

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        past_key_value=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D].
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, 1, T_q, T_kv].
            past_key_value (tuple, optional): Past key-value state. Not supported.
            **kwargs: Additional keyword arguments to be passed to the inner transformer layer.

        Returns:
            torch.Tensor: The output tensor, with the same shape as the input [B, N, D].
        """
        if past_key_value is not None:
            raise NotImplementedError("KV caching not yet supported with adaptive patching.")

        x_patched = self.adaptive_patching(x)

        if attention_mask is not None:
            # Expand both query and key/value length
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=2)  # query length
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=3)  # key/value length

        # Pass patched input and any other arguments to the wrapped layer
        layer_output = self.transformer_layer(
            x_patched,
            attention_mask=attention_mask,
            past_key_value=None,
            **kwargs,
        )

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
