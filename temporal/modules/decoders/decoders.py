import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, List, Union

# Import ModuleBuilder from the new helper file
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig


class TimeSeriesTransformerDecoder(nn.Module):
    """
    Flexible Transformer decoder using block configs.

    Supports:
    - Self + Cross attention
    - Caching for autoregressive decoding
    - Optional return of attention weights
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: List[TransformerBlockConfig],
    ):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = getattr(config, "decoder_layerdrop", 0.0)

        self.layernorm_embedding = builder.build_normalization()
        self.value_embedding = builder.build_value_embedding()
        self.positional_embedding = builder.build_positional_embedding()


        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(block_cfg) for block_cfg in block_configs
        ])

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T, F] raw features
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[Tuple, Tuple]]] = None,  # List of (self_kv, cross_kv)
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:

        # === Embedding ===
        # Embed raw float input features
        hidden_states = self.value_embedding(input_ids)  # Corrected variable name from input_values to input_ids [B, T, D]

        # Generate sinusoidal position encoding based on shape
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        pos_embed = self.positional_embedding(position_ids)  # [B, T, D]

        # Sum + norm + dropout
        hidden_states = hidden_states + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)


        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        next_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if self.training and torch.rand([]).item() < self.layerdrop:
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_kv = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_kv,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_key_values.append(layer_outputs[3])

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attns += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                all_hidden_states,
                all_self_attns,
                all_cross_attns,
                next_key_values,
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
            past_key_values=next_key_values,
        )