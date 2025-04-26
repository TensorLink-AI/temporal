import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List

# Import ModuleBuilder from the new helper file
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig


class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, config, builder: ModuleBuilder, block_configs: List[TransformerBlockConfig]):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = getattr(config, "encoder_layerdrop", 0.0)

        #self.input_projection = builder.build_embedding()  # Value embedding: raw features → hidden
        self.layernorm_embedding = builder.build_normalization()
        self.value_embedding = builder.build_value_embedding()
        self.positional_embedding = builder.build_positional_embedding()
        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(cfg) for cfg in block_configs
        ])

    def forward(
        self,
        input_values: torch.FloatTensor,  # [B, L, F]
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
    
        hidden_states = self.value_embedding(input_values)  # [B, L, D]

        # Get position IDs for sinusoidal or learned embeddings
        B, L, _ = input_values.shape
        position_ids = torch.arange(L, device=input_values.device).unsqueeze(0).expand(B, L)
        pos_embed = self.positional_embedding(position_ids)  # [B, L, D]

        hidden_states = hidden_states + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)



        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if self.training and torch.rand([]).item() < self.layerdrop:
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
