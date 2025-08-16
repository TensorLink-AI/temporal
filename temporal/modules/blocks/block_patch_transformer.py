import torch
import torch.nn as nn
import copy
from dataclasses import replace 

from temporal.models.mixin.adaptive_patching import PatchSplitting, PatchMerging
from temporal.registry.core import register_module
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.outputs import DecoderLayerOutput
from temporal.configs.transformer_block_config import transformer_block_config_from_dict, AdaptivePatchTransformerBlockConfig
from temporal.configs.feedforward_config import FeedForwardConfig # Import FeedForwardConfig

@register_module("block", "patch_transform_block")
class PatchTransformBlock(nn.Module):
    """
    A transformer block that wraps another transformer layer, applying patch
    splitting/merging before and after.
    """
    def __init__(
        self,
        config: AdaptivePatchTransformerBlockConfig, # Corrected: receive specific block config
        builder: ModuleBuilder,
        **kwargs,
    ):
        super().__init__()
        # No need to check config type, it's already typed in the signature
        # No need to get config from builder, it's passed as an argument

        self.expansion_factor = config.expansion_factor
        self.order = config.order
        
        original_model_config = builder.config # Access the original model config from the builder
        d_model = original_model_config.d_model

        if self.order == 'split_first':
            inner_dim = d_model // self.expansion_factor
            if d_model % self.expansion_factor != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by expansion_factor ({self.expansion_factor})")
            
            self.patch_splitting = PatchSplitting(input_dim=d_model, expansion_factor=self.expansion_factor, use_mlp=True)
            self.patch_merging = PatchMerging(input_dim=inner_dim, merge_factor=self.expansion_factor, use_mlp=True)
            
            temp_config = replace(original_model_config, d_model=inner_dim)
        
        elif self.order == 'merge_first':
            if self.expansion_factor != 2:
                raise ValueError(f"For 'merge_first' order with an MLP, expansion_factor must be 2. Got {self.expansion_factor}.")
            
            inner_dim = d_model * 2
            
            self.patch_merging = PatchMerging(input_dim=d_model, merge_factor=self.expansion_factor, use_mlp=True)
            self.patch_splitting = PatchSplitting(input_dim=inner_dim, expansion_factor=self.expansion_factor, use_mlp=True)
            
            temp_config = replace(original_model_config, d_model=inner_dim)

        temp_builder = ModuleBuilder(temp_config)

        # Adapt FFN config based on inner_dim if it's the default 'standard' type
        adapted_ffn_config_dict = config.ffn_config.to_dict()
        # The default `intermediate_size` for `StandardFeedForward` (type='standard')
        # is typically 4 * d_model. We should update it relative to the inner_dim.
        # We need to be careful if intermediate_size was explicitly set in the config.
        # A safer check is to see if it matches the default value if it were derived from the original d_model.
        
        # Get the default intermediate_size if a StandardFeedForwardConfig was created with original d_model
        # This requires an instance of FeedForwardConfig to get the default, or know its default logic
        # The default for StandardFeedForwardConfig.intermediate_size is 2048.
        # If the user explicitly set it, we should respect that.
        # If it's the default, we calculate it based on inner_dim.
        
        # Assuming the default `intermediate_size` in FeedForwardConfig is what we want to override if present
        # If the user has explicitly set intermediate_size in their config, we should respect it.
        # If it's still the default, then we calculate it based on inner_dim.

        if config.ffn_config.type == "standard":
            # Check if intermediate_size was NOT explicitly set by the user to a non-default value
            # This is a bit tricky, as dataclass defaults can be hard to distinguish from user-provided defaults.
            # A common pattern for this is to check if it's the 'default_factory' created value or similar.
            # For simplicity, if it's the default (e.g., 2048) assume it needs adapting.
            # If it's not present in the dict (meaning it's using the default_factory), it also needs adapting.
            
            # The most robust way is to re-create the default FeedForwardConfig with the current d_model
            # and see if the user's config's intermediate_size matches that default.
            # However, since FeedForwardConfig itself doesn't directly depend on d_model in its __init__ (it's passed during build),
            # we have to rely on a common convention, like 4*d_model.

            # If the intermediate_size was NOT explicitly set by the user, or if it matches the 'typical' default derived from original d_model (e.g., 4 * original_d_model)
            # and the type is standard, we adjust.
            # For simplicity, if the intermediate_size is exactly what FeedForwardConfig's default_factory would produce (e.g., related to the outer d_model's default 4*d_model, like 2048)
            # we assume it should be adapted.

            # This check is more robust: if the current intermediate_size is the same as the default from the FeedForwardConfig class definition
            # OR if it was dynamically set based on the ORIGINAL d_model (e.g., 4 * original_model_config.d_model), then adjust.
            # The default intermediate_size for standard FFN is 2048. If that's what's in the config, we override it.
            default_standard_ffn_intermediate_size = 2048 # This should ideally come from FeedForwardConfig's default
            if (config.ffn_config.intermediate_size == default_standard_ffn_intermediate_size) or \
               (config.ffn_config.intermediate_size is None): # if it's None, it means the default_factory in the config class would apply
                adapted_ffn_config_dict["intermediate_size"] = inner_dim * 4


        inner_layer_cfg_dict = {
            "type": config.wrapped_block_type,
            "attention_config": config.attention_config.to_dict(),
            "ffn_config": adapted_ffn_config_dict,
            "kwargs": config.kwargs,
        }
        inner_layer_cfg = transformer_block_config_from_dict(inner_layer_cfg_dict)
        self.transformer_layer = temp_builder._build("block", inner_layer_cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_value=None,
        **kwargs,
    ) -> DecoderLayerOutput:
        if past_key_value is not None:
            raise NotImplementedError("KV caching not yet supported with patch transform blocks.")

        if self.order == 'split_first':
            return self.forward_split_first(hidden_states, attention_mask, **kwargs)
        else:
            return self.forward_merge_first(hidden_states, attention_mask, **kwargs)

    def _extract_layer_output(self, layer_output):
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        if hasattr(layer_output, 'last_hidden_state') and layer_output.last_hidden_state is not None:
            return layer_output.last_hidden_state 
        if hasattr(layer_output, 'hidden_states') and layer_output.hidden_states is not None:
            return layer_output.hidden_states[-1] if isinstance(layer_output.hidden_states, (list, tuple)) else layer_output.hidden_states
        if isinstance(layer_output, tuple) and len(layer_output) > 0 and isinstance(layer_output[0], torch.Tensor):
            return layer_output[0]
        raise TypeError(f"Unsupported output type from transformer_layer: {type(layer_output)}")

    def forward_split_first(self, hidden_states, attention_mask, **kwargs):
        x_split = self.patch_splitting(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=-1).repeat_interleave(self.expansion_factor, dim=-2)

        layer_output = self.transformer_layer(hidden_states=x_split, attention_mask=attention_mask, **kwargs)
        x_processed = self._extract_layer_output(layer_output)
        merged_output = self.patch_merging(x_processed)

        return DecoderLayerOutput(
            hidden_states=merged_output,
            self_attention_weights=getattr(layer_output, 'attentions', [None])[0],
            past_key_value=getattr(layer_output, 'past_key_values', None),
        )

    def forward_merge_first(self, hidden_states, attention_mask, **kwargs):
        x_merged = self.patch_merging(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask[..., ::self.expansion_factor, ::self.expansion_factor]

        layer_output = self.transformer_layer(hidden_states=x_merged, attention_mask=attention_mask, **kwargs)
        x_processed = self._extract_layer_output(layer_output)
        split_output = self.patch_splitting(x_processed)

        return DecoderLayerOutput(
            hidden_states=split_output,
            self_attention_weights=getattr(layer_output, 'attentions', [None])[0],
            past_key_value=getattr(layer_output, 'past_key_values', None),
        )