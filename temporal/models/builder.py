import torch.nn as nn

from temporal.registry.core import resolve
from temporal.registry.generate import resolve_generate
from temporal.models.base_model import BaseTimeSeriesModel
from temporal.models.block_builder import BlockBuilder
from temporal.models.output_head_builder import OutputHeadBuilder
from temporal.modules.encoders.transformer_encoder import TimeSeriesTransformerEncoder
from temporal.modules.decoders.transformer_decoder import TimeSeriesTransformerDecoder
from temporal.configs.transformer_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_config import (
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

    def build_attention(self, attn_cfg):
        cls = resolve("attention", attn_cfg.attention_type)
        return cls(
            embed_dim=self.config.hidden_size,
            num_heads=attn_cfg.num_heads,
            dropout=attn_cfg.dropout,
            **attn_cfg.kwargs,
        )

    def build_feedforward(self, ffn_cfg=None):
        cfg = ffn_cfg or self.config.feedforward_config
        cls = resolve("feedforward", cfg.type)
        return cls(
            hidden_size=self.config.hidden_size,
            intermediate_size=cfg.intermediate_size,
            activation=cfg.activation,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_value_embedding(self):
        cfg = self.config.value_embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            input_size=self.config.feature_size,
            hidden_size=self.config.hidden_size,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_positional_embedding(self):
        cfg = self.config.positional_embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            num_positions=self.config.context_length + self.config.prediction_length,
            embedding_dim=self.config.hidden_size,
            **cfg.kwargs,
        )


    def build_head_aggregator(self):
        cfg = self.config.head_agg_config
        cls = resolve("head_agg", cfg.type)

        # Determine input size = output head output size
        output_size = self.config.output_head_config.output_size
        if output_size is None:
            raise ValueError("output_head_config.output_size must be set to build head aggregator.")

        return cls(
            input_size=output_size,  # Q
            num_heads=self.config.output_token_lengths,
            output_size=output_size,  # Optional for some aggregators
            **cfg.kwargs
        )

    def build_normalization(self):
        cfg = self.config.norm_config
        cls = resolve("normalization", cfg.norm_type)
        return cls(eps=cfg.eps, normalized_shape=self.config.hidden_size)


def build_time_series_transformer(config, generate_strategy=None):
    if hasattr(config, "validate_config"):
        config.validate_config()  # ✅ fail-fast if invalid
    else:
        raise ValueError("Config object does not implement validate_config()")
    builder = ModuleBuilder(config)
    block_builder = BlockBuilder(config, builder)
    output_head_builder = OutputHeadBuilder(config)

    # === Encoder ===
    encoder = None
    if config.architecture.layout in ("encoder", "encoder-decoder"):
        if not getattr(config, "encoder_blocks", None):
            raise ValueError("Missing encoder_blocks in config")
        encoder = TimeSeriesTransformerEncoder(
            config=config,
            builder=builder,
            block_configs=config.encoder_blocks
        )

    # === Decoder ===
    decoder = None
    if config.architecture.layout in ("decoder", "encoder-decoder"):
        if not getattr(config, "decoder_blocks", None):
            raise ValueError("Missing decoder_blocks in config")
        decoder = TimeSeriesTransformerDecoder(
            config=config,
            builder=builder,
            block_configs=config.decoder_blocks
        )

    # === Output head + loss
    output_head, loss_fn = output_head_builder.build()

    # === Head aggregator (optional)
    head_aggregator = None
    if config.output_token_lengths > 1:
        head_aggregator = builder.build_head_aggregator()

    # === Determine final model class from model_type (safe dispatch)
    model_cls = resolve_generate(config.model_type) if generate_strategy is None else resolve_generate(generate_strategy)

    model = model_cls(
        config=config,
        encoder=encoder,
        decoder=decoder,
        output_head=output_head,
        loss_fn=loss_fn,
        head_aggregator=head_aggregator
    )

    return model
