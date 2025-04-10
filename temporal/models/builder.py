import torch.nn as nn

from temporal.registry.core import resolve
from temporal.registry.generate import resolve_generate
from temporal.models.base_model import BaseTimeSeriesModel
from temporal.models.block_builder import BlockBuilder
from temporal.modules.encoders.transformer_encoder import TimeSeriesTransformerEncoder
from temporal.modules.decoders.transformer_decoder import TimeSeriesTransformerDecoder
from temporal.modules.loss import TimeSeriesLoss
from temporal.configs.transformer_config import TransformerTimeSeriesConfig
from temporal.configs.subconfigs import (
    AttentionConfig,
    FeedForwardConfig,
    EmbeddingConfig,
    HeadAggregationConfig,
    NormalizationConfig,
    TransformerBlockConfig,
)


class ModuleBuilder:
    def __init__(self, config):
        self.config = config

    def build_attention(self, attn_cfg: AttentionConfig):
        cls = resolve("attention", attn_cfg.attention_type)
        return cls(
            embed_dim=self.config.hidden_size,
            num_heads=attn_cfg.num_heads,
            dropout=attn_cfg.dropout,
            **attn_cfg.kwargs,
        )

    def build_feedforward(self):
        cfg: FeedForwardConfig = self.config.feedforward_config
        cls = resolve("feedforward", cfg.type)
        return cls(
            hidden_size=self.config.hidden_size,
            intermediate_size=cfg.intermediate_size,
            activation=cfg.activation,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_embedding(self):
        cfg: EmbeddingConfig = self.config.embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            input_size=self.config.feature_size,
            hidden_size=self.config.hidden_size,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_head_aggregator(self):
        cfg: HeadAggregationConfig = self.config.head_agg_config
        cls = resolve("head_agg", cfg.type)
        return cls(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.output_token_lengths,
            output_size=self.config.num_quantiles,
            **cfg.kwargs,
        )

    def build_normalization(self):
        cfg = self.config.norm_config
        cls = resolve("normalization", cfg.norm_type)
        return cls(eps=cfg.eps, normalized_shape=self.config.hidden_size)


def build_time_series_transformer(
    config: TransformerTimeSeriesConfig,
    generate_strategy: str = None
) -> BaseTimeSeriesModel:
    builder = ModuleBuilder(config)
    block_builder = BlockBuilder(config, builder)

    # === Create block configs if not already present ===
    if not hasattr(config, "encoder_blocks"):
        config.encoder_blocks = [
            TransformerBlockConfig(
                block_type="default_encoder",
                attention_config=config.attention_blocks.encoder_attention,
                ffn_config=config.feedforward_config,
            ) for _ in range(config.architecture.num_encoder_layers)
        ]

    if not hasattr(config, "decoder_blocks"):
        config.decoder_blocks = [
            TransformerBlockConfig(
                block_type="default_decoder",
                attention_config=config.attention_blocks.decoder_attention,
                ffn_config=config.feedforward_config,
                kwargs={"cross_attention_config": config.attention_blocks.decoder_cross_attention}
            ) for _ in range(config.architecture.num_decoder_layers)
        ]

    # === Encoder ===
    encoder = None
    encoder_output_dim = config.hidden_size
    if config.architecture.layout in ("encoder", "encoder-decoder"):
        encoder = TimeSeriesTransformerEncoder(
            config=config,
            builder=builder,
            block_configs=config.encoder_blocks
        )
        if hasattr(encoder, "output_dim"):
            encoder_output_dim = encoder.output_dim

    # === Decoder ===
    decoder = None
    decoder_input_dim = config.hidden_size
    if config.architecture.layout in ("decoder", "encoder-decoder"):
        decoder = TimeSeriesTransformerDecoder(
            config=config,
            builder=builder,
            block_configs=config.decoder_blocks
        )
        if hasattr(decoder, "input_dim"):
            decoder_input_dim = decoder.input_dim

    # === Compatibility check
    if encoder and decoder:
        if encoder_output_dim != decoder_input_dim:
            raise ValueError(
                f"[build] Encoder and decoder hidden size mismatch:\n"
                f"    encoder output dim: {encoder_output_dim}\n"
                f"    decoder input dim:  {decoder_input_dim}\n"
                "Consider adding a projection layer or aligning hidden sizes."
            )

    # === Output heads + loss ===
    output_heads = nn.ModuleList([
        nn.Linear(config.hidden_size, config.num_quantiles)
        for _ in range(config.output_token_lengths)
    ])
    
    head_aggregator = builder.build_head_aggregator()
    loss_fn = TimeSeriesLoss(config, loss_type=config.loss_type)

    model = BaseTimeSeriesModel(
        config=config,
        encoder=encoder,
        decoder=decoder,
        output_heads=output_heads,
        head_aggregator=head_aggregator,
        loss_fn=loss_fn,
    )

    # === Inject .generate() behavior if needed ===
    if generate_strategy is not None:
        wrapper_cls = resolve_generate(generate_strategy)
        model.__class__ = wrapper_cls

    return model
