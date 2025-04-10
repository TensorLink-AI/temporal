import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from temporal.models.builder import ModuleBuilder
from temporal.models.block_builder import BlockBuilder  # ✅ new
from temporal.configs.transformer_block_config import TransformerBlockConfig  # or wherever you defined it


class TimeSeriesTransformerDecoder(nn.Module):
    """
    Transformer decoder for time series, using flexible blocks.

    Args:
        config: TransformerTimeSeriesConfig
        builder: ModuleBuilder
        block_configs: List[TransformerBlockConfig] for each decoder layer
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: list[TransformerBlockConfig],
    ):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = config.decoder_layerdrop

        self.value_embedding = builder.build_embedding()
        self.positional_embedding = builder.build_embedding()
        self.layernorm_embedding = builder.build_normalization()

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(block_cfg) for block_cfg in block_configs
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPastAndCrossAttentions:

        hidden_states = self.value_embedding(inputs_embeds)
        position_embeddings = self.positional_embedding(inputs_embeds)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        for layer in self.layers:
            if self.training and torch.rand([]).item() < self.layerdrop:
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            past_key_values=None,
        )
