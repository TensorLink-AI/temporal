import torch
import torch.nn as nn
import copy

# We assume these are in a file like `temporal.models.mixin.patch_utils`
# and have been updated to include the 2-layer MLP option.
from temporal.models.mixin.adaptive_patching import PatchSplitting, PatchMerging
from temporal.registry.core import register_module
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig
from temporal.models.outputs import DecoderLayerOutput

@register_module("block", "patch_transform_block")
class PatchTransformBlock(nn.Module):
    """
    A transformer block that wraps another transformer layer, applying patch
    splitting/merging before and after.

    This block supports two orders of operation via the `order` parameter:
    1. 'split_first': (Default) Splits patches to a higher resolution, processes
       with the inner transformer, then merges them back.
       (D -> D/K -> D)
    2. 'merge_first': Merges patches to a lower resolution, processes, then
       splits them back to the original resolution. This requires the
       expansion_factor to be 2. (D -> 2*D -> D)

    Args:
        builder (ModuleBuilder): The main module builder.
        wrapped_block_type (str): The type of block to wrap (e.g., 'default_encoder').
        expansion_factor (int): The factor K to split/merge patches.
        order (str): The order of operations. One of ['split_first', 'merge_first'].
        attention_config (dict, optional): Specific attention config for the inner block.
        ffn_config (dict, optional): Specific FFN config for the inner block.
        kwargs: Additional keyword arguments for the inner block.
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
        
        # Create a deep copy of the config to build the inner layer in isolation.
        temp_config = copy.deepcopy(builder.config)
        d_model = temp_config.d_model

        if self.order == 'split_first':
            # --- Configure for Split -> Transform -> Merge ---
            inner_dim = d_model // expansion_factor
            if d_model % expansion_factor != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by expansion_factor ({expansion_factor})")
            
            self.patch_splitting = PatchSplitting(input_dim=d_model, expansion_factor=expansion_factor, use_mlp=True)
            self.patch_merging = PatchMerging(input_dim=inner_dim, merge_factor=expansion_factor, use_mlp=True)
            
            temp_config.d_model = inner_dim
        
        elif self.order == 'merge_first':
            # --- Configure for Merge -> Transform -> Split ---
            # NOTE: This order requires expansion_factor==2 for the dimensions to align,
            # because the MLP in PatchMerging projects the dimension to 2*input_dim.
            if self.expansion_factor != 2:
                raise ValueError(
                    f"For 'merge_first' order with an MLP, expansion_factor must be 2. Got {self.expansion_factor}."
                )
            
            inner_dim = d_model * 2
            
            # Merging takes D and outputs 2*D.
            self.patch_merging = PatchMerging(input_dim=d_model, merge_factor=expansion_factor, use_mlp=True)
            # Splitting must take 2*D and output D.
            self.patch_splitting = PatchSplitting(input_dim=inner_dim, expansion_factor=expansion_factor, use_mlp=True)
            
            temp_config.d_model = inner_dim

        # --- Inner Layer Construction ---
        temp_builder = ModuleBuilder(temp_config)
        inner_layer_cfg = TransformerBlockConfig(
            block_type=wrapped_block_type,
            attention_config=attention_config,
            ffn_config=ffn_config,
            kwargs=kwargs,
        )
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
        if past_key_value is not None:
            raise NotImplementedError("KV caching not yet supported with patch transform blocks.")

        if self.order == 'split_first':
            return self.forward_split_first(hidden_states, attention_mask, **kwargs)
        else: # 'merge_first'
            return self.forward_merge_first(hidden_states, attention_mask, **kwargs)

    def forward_split_first(self, hidden_states, attention_mask, **kwargs):
        # Path: Split -> Transform -> Merge
        # Shape: [B, N, D] -> [B, N*K, D/K] -> [B, N*K, D/K] -> [B, N, D]
        
        # 1. Split patches
        x_split = self.patch_splitting(hidden_states)

        # 2. Expand attention mask if provided
        if attention_mask is not None:
            # Assumes mask is [B, 1, N, N] or similar
            B, H, N, W = attention_mask.shape
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=2)
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=3)

        # 3. Process with inner transformer
        layer_output = self.transformer_layer(
            hidden_states=x_split,
            attention_mask=attention_mask,
            past_key_value=None,
            **kwargs,
        )
        x_processed = None
        if isinstance(layer_output, torch.Tensor):
            x_processed = layer_output
        elif hasattr(layer_output, 'last_hidden_state') and layer_output.last_hidden_state is not None:
            x_processed = layer_output.last_hidden_state 
        elif hasattr(layer_output, 'hidden_states') and layer_output.hidden_states is not None:
            x_processed = layer_output.hidden_states
        elif isinstance(layer_output, tuple) and len(layer_output) > 0 and isinstance(layer_output[0], torch.Tensor):
            x_processed = layer_output[0]
        else:
            raise TypeError(
                f"Unsupported output type from transformer_layer: {type(layer_output)}. "
                f"Expected torch.Tensor, or object with 'last_hidden_state'/'hidden_states' attribute."
            )
        # 4. Merge patches back
        aux_hidden_states = None 
        self_attention_weights = None
        cross_attention_weights = None
        past_key_value = None # Singular name
        aux_loss = None

        # Try to extract auxiliary info from inner layer's output (`layer_output`)
        # Use hasattr and getattr for robustness against different layer_output types
        # and to handle cases where outputs are not requested (and thus None).

        # Main hidden_states will be the merged_output. Aux_hidden_states are intermediates.
        aux_hidden_states = getattr(layer_output, 'hidden_states', None)

        # Attention weights mapping: layer_output might have 'attentions' (plural)
        # or 'self_attention_weights' (singular)
        if hasattr(layer_output, 'attentions') and layer_output.attentions is not None:
            # If attentions is a tuple of (self_attn, cross_attn), assume first is self_attn
            if isinstance(layer_output.attentions, tuple) and len(layer_output.attentions) > 0:
                self_attention_weights = layer_output.attentions[0]
            else: # If it's a single tensor or not a tuple, assume it's self-attention
                self_attention_weights = layer_output.attentions
        elif hasattr(layer_output, 'self_attention_weights'):
            self_attention_weights = layer_output.self_attention_weights

        # Cross-attention weights mapping
        if hasattr(layer_output, 'cross_attentions') and layer_output.cross_attentions is not None:
            if isinstance(layer_output.cross_attentions, tuple) and len(layer_output.cross_attentions) > 0:
                cross_attention_weights = layer_output.cross_attentions[0]
            else:
                cross_attention_weights = layer_output.cross_attentions
        elif hasattr(layer_output, 'cross_attention_weights'):
            cross_attention_weights = layer_output.cross_attention_weights
        
        # Past Key Values mapping (singular 'past_key_value' expected by DecoderLayerOutput)
        if hasattr(layer_output, 'past_key_values') and layer_output.past_key_values is not None: # From BaseModelOutput... (plural)
            past_key_value = layer_output.past_key_values
        elif hasattr(layer_output, 'past_key_value') and layer_output.past_key_value is not None: # From custom/other layers (singular)
            past_key_value = layer_output.past_key_value

        # Auxiliary Loss mapping
        aux_loss = getattr(layer_output, 'aux_loss', None)

        return DecoderLayerOutput(
            hidden_states=merged_output, # The processed main hidden state after patch merging
            self_attention_weights=self_attention_weights,
            cross_attention_weights=cross_attention_weights,
            past_key_value=past_key_value, 
            aux_loss=aux_loss
        )

    def forward_merge_first(self, hidden_states, attention_mask, **kwargs):
        # Path: Merge -> Transform -> Split
        # Shape: [B, N, D] -> [B, N/K, 2*D] -> [B, N/K, 2*D] -> [B, N, D]
        
        # 1. Merge patches
        x_merged = self.patch_merging(hidden_states)

        # 2. Downsample attention mask if provided
        if attention_mask is not None:
            # A simple way to downsample is to subsample
            attention_mask = attention_mask[:, :, ::self.expansion_factor, ::self.expansion_factor]

        # 3. Process with inner transformer
        layer_output = self.transformer_layer(
            hidden_states=x_merged,
            attention_mask=attention_mask,
            past_key_value=None,
            **kwargs,
        )

        # 4. Split patches back
        x_processed = None
        if isinstance(layer_output, torch.Tensor):
            x_processed = layer_output
        elif hasattr(layer_output, 'last_hidden_state') and layer_output.last_hidden_state is not None:
            x_processed = layer_output.last_hidden_state 
        elif hasattr(layer_output, 'hidden_states') and layer_output.hidden_states is not None:
            x_processed = layer_output.hidden_states
        elif isinstance(layer_output, tuple) and len(layer_output) > 0 and isinstance(layer_output[0], torch.Tensor):
            x_processed = layer_output[0]
        else:
            raise TypeError(
                f"Unsupported output type from transformer_layer: {type(layer_output)}. "
                f"Expected torch.Tensor, or object with 'last_hidden_state'/'hidden_states' attribute."
            )
        split_output = self.patch_splitting(x_processed)

        aux_hidden_states = None 
        self_attention_weights = None
        cross_attention_weights = None
        past_key_value = None 
        aux_loss = None

        # Try to extract auxiliary info from inner layer's output (`layer_output`)
        aux_hidden_states = getattr(layer_output, 'hidden_states', None)

        if hasattr(layer_output, 'attentions') and layer_output.attentions is not None:
            if isinstance(layer_output.attentions, tuple) and len(layer_output.attentions) > 0:
                self_attention_weights = layer_output.attentions[0]
            else:
                self_attention_weights = layer_output.attentions
        elif hasattr(layer_output, 'self_attention_weights'):
            self_attention_weights = layer_output.self_attention_weights

        if hasattr(layer_output, 'cross_attentions') and layer_output.cross_attentions is not None:
            if isinstance(layer_output.cross_attentions, tuple) and len(layer_output.cross_attentions) > 0:
                cross_attention_weights = layer_output.cross_attentions[0]
            else:
                cross_attention_weights = layer_output.cross_attentions
        elif hasattr(layer_output, 'cross_attention_weights'):
            cross_attention_weights = layer_output.cross_attention_weights
        
        if hasattr(layer_output, 'past_key_values') and layer_output.past_key_values is not None:
            past_key_value = layer_output.past_key_values 
        elif hasattr(layer_output, 'past_key_value') and layer_output.past_key_value is not None:
            past_key_value = layer_output.past_key_value

        aux_loss = getattr(layer_output, 'aux_loss', None)

        return DecoderLayerOutput(
            hidden_states=split_output, # The processed main hidden state after patch splitting
            self_attention_weights=self_attention_weights,
            cross_attention_weights=cross_attention_weights,
            past_key_value=past_key_value,
            aux_loss=aux_loss
        )
