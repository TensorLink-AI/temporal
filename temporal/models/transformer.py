import torch
import torch.nn as nn
from transformers import PreTrainedModel
from temporal.models.builder import ModuleBuilder, expand_attention_config
from temporal.modules.encoders.transformer_encoder import TimeSeriesTransformerEncoder
from temporal.modules.decoders.transformer_decoder import TimeSeriesTransformerDecoder
from temporal.modules.loss import TimeSeriesLoss
from temporal.configs.transformer_hf_config import TransformerTimeSeriesConfig


class TimeSeriesTransformerModel(PreTrainedModel):
    config_class = TransformerTimeSeriesConfig

    def __init__(self, config: TransformerTimeSeriesConfig):
        super().__init__(config)
        self.config = config
        builder = ModuleBuilder(config)

        encoder = None
        decoder = None

        # === Encoder
        if config.architecture.layout in ("encoder", "encoder-decoder"):
            encoder_attn_cfgs = expand_attention_config(
                config.attention_blocks.encoder_attention,
                config.architecture.num_encoder_layers
            )
            encoder = TimeSeriesTransformerEncoder(
                config=config,
                builder=builder,
                attention_configs=encoder_attn_cfgs
            )

        # === Decoder
        if config.architecture.layout in ("decoder", "encoder-decoder"):
            decoder_self_attn_cfgs = expand_attention_config(
                config.attention_blocks.decoder_attention,
                config.architecture.num_decoder_layers
            )
            decoder_cross_attn_cfgs = expand_attention_config(
                config.attention_blocks.decoder_cross_attention,
                config.architecture.num_decoder_layers
            )
            decoder = TimeSeriesTransformerDecoder(
                config=config,
                builder=builder,
                self_attention_configs=decoder_self_attn_cfgs,
                cross_attention_configs=decoder_cross_attn_cfgs
            )

        # === Output heads + aggregator
        output_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_quantiles)
            for _ in range(config.output_token_lengths)
        ])
        head_aggregator = builder.build_head_aggregator()
        loss_fn = TimeSeriesLoss(config, loss_type=config.loss_type)

        self.encoder = encoder
        self.decoder = decoder
        self.output_heads = output_heads
        self.head_aggregator = head_aggregator
        self.loss_fn = loss_fn

    def forward(
        self,
        input_ids,
        decoder_input_ids=None,
        attention_mask=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        encoder_outputs = self.encoder(
            inputs_embeds=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=None,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        sequence_output = decoder_outputs.last_hidden_state

        head_outputs = [head(sequence_output) for head in self.output_heads]
        predictions = self.head_aggregator(head_outputs)

        loss = self.loss_fn(predictions, labels) if labels is not None else None

        if not return_dict:
            return (predictions, loss)

        return {
            "loss": loss,
            "logits": predictions,
            "encoder_hidden_states": encoder_outputs.hidden_states,
            "decoder_hidden_states": decoder_outputs.hidden_states
        }
