import torch.nn as nn
from typing import Optional

from temporal.registry.core import resolve
from temporal.registry.generate import resolve_generate
from temporal.models.base_model import BaseTemporalModel
# from temporal.models.block_builder import BlockBuilder # No longer directly used
from temporal.models.output_head_builder import OutputHeadBuilder
from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.modules.decoders.decoders import TimeSeriesTransformerDecoder

# Import the new TransformerTimeSeriesConfig from its new location
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.models.module_builder_helper import ModuleBuilder


def build_time_series_transformer(
    config: TransformerTimeSeriesConfig,
    generate_strategy: Optional[str] = None
) -> BaseTemporalModel:
    """
    Constructs a full transformer-based time series model from a configuration object.

    This function serves as the main entry point for model creation. It takes a
    comprehensive configuration object and orchestrates the entire build process,
    including:
    1.  Validating the configuration (now handled by dataclass __post_init__).
    2.  Initializing the necessary builders (`ModuleBuilder`, `OutputHeadBuilder`).
    3.  Building the encoder and/or decoder stacks based on the specified architecture.
    4.  Building the output head(s).
    5.  Building the primary loss function.
    6.  Resolving the final model class (which defines the `generate` strategy)
        and instantiating it with all the built components.

    Args:
        config (TransformerTimeSeriesConfig): The complete configuration object
            that defines the model's architecture, layers, and components.
        generate_strategy (Optional[str]): If provided, this string overrides the
            `model_type` from the config to select a specific generation class
            from the `generate_registry`.

    Returns:
        BaseTemporalModel: An instantiated and fully constructed time series model.

    Raises:
        ValueError: If the configuration is invalid or missing required sections
            (e.g., `encoder_blocks` for an encoder-based architecture).
        RuntimeError: If the loss function cannot be built from the configuration.
    """
    # Config validation is now handled by TransformerTimeSeriesConfig's __post_init__
    # upon its creation. So, no explicit call needed here.

    # Initialize the primitive and composite builders.
    builder = ModuleBuilder(config)
    # BlockBuilder is no longer a separate class, its logic is integrated into Encoder/Decoder modules
    output_head_builder = OutputHeadBuilder(config, builder) # Pass builder to OutputHeadBuilder

    # Conditionally build the encoder based on the architecture layout.
    encoder = None
    if config.architecture.layout in ("encoder", "encoder-decoder"):
        if not config.encoder_blocks:
            raise ValueError("Config specifies an encoder, but 'encoder_blocks' are not defined.")
        encoder = TimeSeriesTransformerEncoder(
            config=config,
            builder=builder,
            block_configs=config.encoder_blocks,
        )

    # Conditionally build the decoder.
    decoder = None
    if config.architecture.layout in ("decoder", "encoder-decoder"):
        if not config.decoder_blocks:
            raise ValueError("Config specifies a decoder, but 'decoder_blocks' are not defined.")
        decoder = TimeSeriesTransformerDecoder(
            config=config,
            builder=builder,
            block_configs=config.decoder_blocks,
        )

    # Build the output head(s).
    output_head = output_head_builder.build()

    # Build the main loss function from the loss configuration.
    # loss_config is now a dataclass, check its type directly
    if not config.loss_config or not config.loss_config.type:
        raise ValueError("Config must have a 'loss_config' with a 'type' key.")
    loss_fn = builder.build_loss(config.loss_config) # Pass the loss_config dataclass
    if loss_fn is None:
        raise RuntimeError(f"Failed to build the loss function from config: {config.loss_config.type}")

    # Build the head aggregator if multiple output tokens are used.
    head_aggregator = None
    if config.output_token_lengths > 1 and config.head_agg_config:
        head_aggregator = builder.build_head_aggregator(config.head_agg_config) # Pass the head_agg_config dataclass

    # Resolve the final model class, which wraps the components and defines
    # the high-level forward and generate logic.
    model_type = generate_strategy or config.model_type
    model_cls = resolve_generate(model_type)

    # Instantiate the final model with all its constructed parts.
    model = model_cls(
        config=config,
        encoder=encoder,
        decoder=decoder,
        output_heads=output_head,
        loss_fn=loss_fn,
        head_aggregator=head_aggregator,
    )

    return model
