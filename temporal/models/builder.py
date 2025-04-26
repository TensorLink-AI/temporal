import torch.nn as nn

from temporal.registry.core import resolve
from temporal.registry.generate import resolve_generate
from temporal.models.base_model import BaseTemporalModel
from temporal.models.block_builder import BlockBuilder
from temporal.models.output_head_builder import OutputHeadBuilder
from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.modules.decoders.decoders import TimeSeriesTransformerDecoder
from temporal.configs.transformer_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_config import (
    AttentionConfig,
    FeedForwardConfig,
    EmbeddingConfig,
    HeadAggregationConfig,
    NormalizationConfig,
    TransformerBlockConfig,
)

# Import ModuleBuilder from the new helper file to break circular dependency
from temporal.models.module_builder_helper import ModuleBuilder


def build_time_series_transformer(
    config: TransformerTimeSeriesConfig,
    generate_strategy: str = None
) -> BaseTemporalModel:
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
