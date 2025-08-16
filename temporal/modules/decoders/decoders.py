import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, List, Union

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_block_config import TransformerBlockConfig
from temporal.models.outputs import DecoderLayerOutput

class TimeSeriesTransformerDecoder(nn.Module):
    """
    A flexible Transformer decoder built from a list of block configurations.

    This module serves as the main decoder component in a Transformer-based
    time series model. It dynamically constructs a stack of decoder layers
    based on a list of `TransformerBlockConfig` objects.

    The decoder is responsible for:
    - Sequentially processing an already-embedded sequence through its layers.
    - Handling the Key-Value (KV) cache for efficient autoregressive generation.
    
    The calling model, such as `TransformerTemporalModel`, is responsible for
    all preprocessing, including value and positional embeddings and normalization,
    via the `InputPreprocessor`.

    Attributes:
        config: The main configuration object for the model.
        layers (nn.ModuleList): The stack of decoder layers.
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: List[TransformerBlockConfig],
        **kwargs,
    ):
        """
        Initializes the TimeSeriesTransformerDecoder.

        Args:
            config: The main model configuration object.
            builder (ModuleBuilder): A helper class that constructs the various
                sub-modules for the layers based on the config.
            block_configs (List[TransformerBlockConfig]): A list of configurations,
                where each configuration defines a single decoder layer in the stack.
            **kwargs: Catches unused arguments to ensure backward compatibility.
        """
        super().__init__()
        self.config = config
        self.layerdrop = getattr(config, "decoder_layerdrop", 0.0)

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(block_cfg) for block_cfg in block_configs
        ])

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # Correct type hint
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            use_cache: bool = False,
            return_dict: bool = True,
            x_raw: Optional[torch.Tensor] = None,
        ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:

            all_hidden_states_collector = () if output_hidden_states else None
            all_self_attentions_collector = () if output_attentions else None
            all_cross_attentions_collector = () if output_attentions else None
            
            # --- FIX: Initialize the new cache as a tuple ---
            next_decoder_cache = () if use_cache else None
            total_aux_loss = None

            for idx, layer in enumerate(self.layers):
                if self.training and torch.rand([]).item() < self.layerdrop:
                    if use_cache:
                        # Append a placeholder for the dropped layer
                        next_decoder_cache += (None,)
                    continue

                layer_past_key_value = past_key_values[idx] if past_key_values is not None else None

                if output_hidden_states:
                    all_hidden_states_collector += (hidden_states,)

                layer_outputs: DecoderLayerOutput = layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=layer_past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    x_raw=x_raw,
                )

                hidden_states = layer_outputs.hidden_states

                if use_cache:
                    # --- FIX: Append to the tuple directly ---
                    next_decoder_cache += (layer_outputs.past_key_value,)

                if output_attentions:
                    if layer_outputs.self_attention_weights is not None:
                        all_self_attentions_collector += (layer_outputs.self_attention_weights,)
                    if layer_outputs.cross_attention_weights is not None:
                        all_cross_attentions_collector += (layer_outputs.cross_attention_weights,)

                if layer_outputs.aux_loss is not None:
                    if total_aux_loss is None:
                        total_aux_loss = layer_outputs.aux_loss
                    else:
                        total_aux_loss += layer_outputs.aux_loss

            if output_hidden_states:
                all_hidden_states_collector += (hidden_states,)
            
            # --- FIX: The cache is already a tuple, no conversion needed ---
            next_cache = next_decoder_cache

            if not return_dict:
                return tuple(v for v in [
                    hidden_states, all_hidden_states_collector, all_self_attentions_collector, all_cross_attentions_collector, next_cache
                ] if v is not None)

            output = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states_collector,
                attentions=all_self_attentions_collector,
                cross_attentions=all_cross_attentions_collector,
                past_key_values=next_cache,
            )
            if total_aux_loss is not None:
                output.aux_loss = total_aux_loss
            return output