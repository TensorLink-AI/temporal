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
    """
    Constructs individual submodules for a transformer-based time series model.

    Uses the registry to resolve implementation classes for attention, feedforward,
    embedding, normalization, and head aggregation layers based on configuration.

    Args:
        config (TransformerTimeSeriesConfig): Configuration object with model hyperparameters.
    """
    def __init__(self, config):
        self.config = config

    def build_attention(self, attn_cfg: AttentionConfig) -> nn.Module:
        """
        Build an attention module from its configuration.

        Args:
            attn_cfg (AttentionConfig): Configuration for the attention mechanism.

        Returns:
            nn.Module: Instantiated attention layer.
        """
        cls = resolve("attention", attn_cfg.attention_type)
        return cls(
            embed_dim=self.config.hidden_size,
            num_heads=attn_cfg.num_heads,
            dropout=attn_cfg.dropout,
            **attn_cfg.kwargs,
        )

    def build_feedforward(self, ffn_cfg: FeedForwardConfig = None) -> nn.Module:
        """
        Build a feed-forward network module from its configuration.

        Args:
            ffn_cfg (FeedForwardConfig, optional): Configuration for the feed-forward layer.
                Defaults to the config.feedforward_config if None.

        Returns:
            nn.Module: Instantiated feed-forward network.
        """
        cfg = ffn_cfg or self.config.feedforward_config
        cls = resolve("feedforward", cfg.type)
        return cls(
            hidden_size=self.config.hidden_size,
            intermediate_size=cfg.intermediate_size,
            activation=cfg.activation,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_value_embedding(self) -> nn.Module:
        """
        Build the value embedding module for input features.

        Returns:
            nn.Module: Instantiated embedding layer for values.
        """
        cfg = self.config.value_embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            input_size=self.config.feature_size,
            hidden_size=self.config.hidden_size,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_positional_embedding(self) -> nn.Module:
        """
        Build the positional embedding module for time indices.

        Returns:
            nn.Module: Instantiated positional embedding layer.
        """
        cfg = self.config.positional_embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            num_positions=self.config.context_length + self.config.prediction_length,
            embedding_dim=self.config.hidden_size,
            **cfg.kwargs,
        )

    def build_head_aggregator(self) -> nn.Module:
        """
        Build a head aggregator to combine multiple output tokens.

        Returns:
            nn.Module: Instantiated head aggregation module.

        Raises:
            ValueError: If output_head_config.output_size is not set.
        """
        cfg = self.config.head_agg_config
        cls = resolve("head_agg", cfg.type)

        output_size = self.config.output_head_config.output_size
        if output_size is None:
            raise ValueError("output_head_config.output_size must be set to build head aggregator.")

        return cls(
            input_size=output_size,
            num_heads=self.config.output_token_lengths,
            output_size=output_size,
            **cfg.kwargs,
        )

    def build_normalization(self) -> nn.Module:
        """
        Build a normalization layer based on configuration.

        Returns:
            nn.Module: Instantiated normalization layer.
        """
        cfg = self.config.norm_config
        cls = resolve("normalization", cfg.norm_type)
        return cls(eps=cfg.eps, normalized_shape=self.config.hidden_size)



def build_time_series_transformer(
    config: TransformerTimeSeriesConfig,
    generate_strategy: str = None
) -> BaseTimeSeriesModel:
    """
    Construct a full transformer-based time series model from configuration.

    This function validates the config, builds submodules via ModuleBuilder,
    assembles encoder, decoder, output head, and optional head aggregator,
    and resolves the final model class for instantiation.

    Args:
        config (TransformerTimeSeriesConfig): Model configuration.
        generate_strategy (str, optional): Override for model_type resolution.
            If provided, resolved via the generation registry.

    Returns:
        BaseTimeSeriesModel: Instantiated transformer time series model.

    Raises:
        ValueError: If config lacks required block definitions or generate registry fails.
    """
    # Validate configuration
    if hasattr(config, "validate_config"):
        config.validate_config()
    else:
        raise ValueError("Config object does not implement validate_config()")

    # Initialize builders
    builder = ModuleBuilder(config)
    block_builder = BlockBuilder(config, builder)
    output_head_builder = OutputHeadBuilder(config)

    # Build encoder if applicable
    encoder = None
    if config.architecture.layout in ("encoder", "encoder-decoder"):
        if not getattr(config, "encoder_blocks", None):
            raise ValueError("Missing encoder_blocks in config")
        encoder = TimeSeriesTransformerEncoder(
            config=config,
            builder=builder,
            block_configs=config.encoder_blocks,
        )

    # Build decoder if applicable
    decoder = None
    if config.architecture.layout in ("decoder", "encoder-decoder"):
        if not getattr(config, "decoder_blocks", None):
            raise ValueError("Missing decoder_blocks in config")
        decoder = TimeSeriesTransformerDecoder(
            config=config,
            builder=builder,
            block_configs=config.decoder_blocks,
        )

    # Build output head and loss function
    output_head, loss_fn = output_head_builder.build()

    # Build head aggregator if output_token_lengths > 1
    head_aggregator = None
    if config.output_token_lengths > 1:
        head_aggregator = builder.build_head_aggregator()

    # Resolve model class and instantiate
    model_type = generate_strategy or config.model_type
    model_cls = resolve_generate(model_type)

    model = model_cls(
        config=config,
        encoder=encoder,
        decoder=decoder,
        output_heads=output_head,
        loss_fn=loss_fn,
        head_aggregator=head_aggregator,
    )

    return model
