import torch
import torch.nn as nn
import copy
from dataclasses import replace # FIX: Import the 'replace' function

from temporal.models.mixin.adaptive_patching import PatchSplitting, PatchMerging
from temporal.registry.core import register_module
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.models.outputs import DecoderLayerOutput
from temporal.configs.transformer_block_config import TransformerBlockConfig

@register_module("block", "patch_transform_block")
class PatchTransformBlock(nn.Module):
    """
    A transformer block that wraps another transformer layer, applying patch
    splitting/merging before and after.
    """
    def __init__(
        self,
        builder: ModuleBuilder,
        wrapped_block_type: str,
        expansion_factor: int,
        order: str = 'split_first',
        attention_config: dict = None,
        ffn_config: dict = None,
        **kwargs,
    ):
        super().__init__()
        if order not in ['split_first', 'merge_first']:
            raise ValueError(f"order must be one of 'split_first' or 'merge_first', but got {order}")
        
        self.expansion_factor = expansion_factor
        self.order = order
        
        original_config = builder.config
        d_model = original_config.d_model

        if self.order == 'split_first':
            # --- Configure for Split -> Transform -> Merge ---
            inner_dim = d_model // expansion_factor
            if d_model % expansion_factor != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by expansion_factor ({expansion_factor})")
            
            self.patch_splitting = PatchSplitting(input_dim=d_model, expansion_factor=expansion_factor, use_mlp=True)
            self.patch_merging = PatchMerging(input_dim=inner_dim, merge_factor=expansion_factor, use_mlp=True)
            
            # FIX: Create a new config with the modified d_model instead of assigning to a frozen one.
            temp_config = replace(original_config, d_model=inner_dim)
        
        elif self.order == 'merge_first':
            # --- Configure for Merge -> Transform -> Split ---
            if self.expansion_factor != 2:
                raise ValueError(f"For 'merge_first' order with an MLP, expansion_factor must be 2. Got {self.expansion_factor}.")
            
            inner_dim = d_model * 2
            
            self.patch_merging = PatchMerging(input_dim=d_model, merge_factor=expansion_factor, use_mlp=True)
            self.patch_splitting = PatchSplitting(input_dim=inner_dim, expansion_factor=expansion_factor, use_mlp=True)
            
            # FIX: Create a new config with the modified d_model.
            temp_config = replace(original_config, d_model=inner_dim)

        # --- Inner Layer Construction ---
        temp_builder = ModuleBuilder(temp_config)
        inner_layer_cfg = TransformerBlockConfig(
            type=wrapped_block_type, # FIX: Use 'type' instead of 'block_type' for consistency
            attention_config=attention_config or {"type": "full"},
            ffn_config=ffn_config or {"type": "standard", "intermediate_size": inner_dim * 4},
            kwargs=kwargs,
        )
        # Assuming BlockBuilder is not needed and we can build directly
        self.transformer_layer = temp_builder.build_block(inner_layer_cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_value=None,
        **kwargs,
    ) -> DecoderLayerOutput: # Return a consistent output object
        if past_key_value is not None:
            raise NotImplementedError("KV caching not yet supported with patch transform blocks.")

        if self.order == 'split_first':
            return self.forward_split_first(hidden_states, attention_mask, **kwargs)
        else: # 'merge_first'
            return self.forward_merge_first(hidden_states, attention_mask, **kwargs)

    def _extract_layer_output(self, layer_output):
        """Helper to robustly extract hidden states from various output types."""
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        if hasattr(layer_output, 'last_hidden_state') and layer_output.last_hidden_state is not None:
            return layer_output.last_hidden_state 
        if hasattr(layer_output, 'hidden_states') and layer_output.hidden_states is not None:
            # hidden_states is often a tuple of all layer outputs, take the last one
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