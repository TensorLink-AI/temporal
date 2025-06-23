
import torch
import torch.nn as nn
import copy

from temporal.models.mixin.adaptive_patching import AdaptivePatching, PatchMerging
from temporal.registry.core import register_module
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig

@register_module("block", "adaptive_patch_transformer")
class AdaptivePatchTransformerBlock(nn.Module):
    """
    An adaptive patch transformer block that wraps another transformer layer (like an
    encoder or decoder), applying adaptive patching before the layer and merging
    patches after.

    This block is responsible for its own construction, including creating the
    correctly dimensioned inner layer.

    Args:
        builder (ModuleBuilder): The main module builder.
        wrapped_block_type (str): The type of block to wrap (e.g., 'default_encoder').
        expansion_factor (int): The factor to expand patches and reduce feature dim.
        attention_config (dict, optional): Specific attention config for the inner block.
        ffn_config (dict, optional): Specific FFN config for the inner block.
        kwargs: Additional keyword arguments for the inner block.
    """
    def __init__(
        self,
        builder: ModuleBuilder,
        wrapped_block_type: str,
        expansion_factor: int,
        attention_config: dict = None,
        ffn_config: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.adaptive_patching = AdaptivePatching(expansion_factor)
        self.patch_merging = PatchMerging(expansion_factor)

        # --- Inner Layer Construction ---
        # Create a deep copy of the main config to create an isolated environment for the inner block.
        temp_config = copy.deepcopy(builder.config)
        # Adjust d_model for the inner, wrapped layer.
        temp_config.d_model = builder.config.d_model // expansion_factor

        # Create a new, temporary builder with the modified config.
        temp_builder = ModuleBuilder(temp_config)

        # Define the configuration for the inner layer.
        inner_layer_cfg = TransformerBlockConfig(
            block_type=wrapped_block_type,
            attention_config=attention_config, # Pass along any specific configs
            ffn_config=ffn_config,
            kwargs=kwargs,
        )

        # Create a temporary block builder to construct the inner layer in isolation.
        temp_block_builder = BlockBuilder(temp_config, temp_builder)
        self.transformer_layer = temp_block_builder.build_block(inner_layer_cfg)
        # --- End Inner Layer Construction ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_value=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, N, D].
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_value (tuple, optional): Past key-value state. Not supported.
            **kwargs: Additional keyword arguments to be passed to the inner transformer layer.

        Returns:
            torch.Tensor: The output tensor, with the same shape as the input [B, N, D].
        """
        if past_key_value is not None:
            raise NotImplementedError("KV caching not yet supported with adaptive patching.")

        x_patched = self.adaptive_patching(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=2)
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=3)

        layer_output = self.transformer_layer(
            hidden_states=x_patched,
            attention_mask=attention_mask,
            past_key_value=None,
            **kwargs,
        )

        if isinstance(layer_output, tuple):
            x_processed = layer_output[0]
            other_outputs = layer_output[1:]
            merged_output = self.patch_merging(x_processed)
            return (merged_output,) + other_outputs
        else:
            merged_output = self.patch_merging(layer_output)
            return merged_output
