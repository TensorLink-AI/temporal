import pytest
import torch.nn as nn
from temporal.models.builder import build_time_series_transformer
from temporal.models.base_model import BaseTemporalModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.loss_config import LossConfig
from temporal.configs.output_head_config import OutputHeadConfig


# --- Fixtures ---


@pytest.fixture
def valid_encoder_decoder_config():
    """Provides a valid configuration for a standard encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        # FIX: Explicitly add ffn_config with required 'intermediate_size'
        encoder_blocks=[
            EncoderBlockConfig(
                type="default_encoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        # FIX: Use a valid, registered loss type
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )


@pytest.fixture
def valid_decoder_only_config():
    """Provides a valid configuration for a decoder-only model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(type="transformer_architecture", layout="decoder"),
        # FIX: Explicitly add ffn_config with required 'intermediate_size'
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        # FIX: Use a valid, registered loss type
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )


# --- Test Cases ---


def test_build_model_from_valid_config(valid_encoder_decoder_config):
    """
    Tests that a complete model can be built from a valid configuration.
    """
    model = build_time_series_transformer(valid_encoder_decoder_config)

    assert isinstance(model, BaseTemporalModel)
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.output_heads is not None
    assert model.loss_fn is not None
    assert model.config == valid_encoder_decoder_config


def test_build_decoder_only_model(valid_decoder_only_config):
    """
    Tests that a decoder-only model is built correctly, with no encoder.
    """
    model = build_time_series_transformer(valid_decoder_only_config)

    assert isinstance(model, BaseTemporalModel)
    assert model.encoder is None
    assert model.decoder is not None
    assert len(model.decoder.layers) == len(valid_decoder_only_config.decoder_blocks)


def test_build_raises_for_missing_encoder_blocks():
    """
    Tests that the builder raises a ValueError if the layout requires an encoder
    but no encoder_blocks are provided.
    """
    # FIX: Add required ffn_config to the decoder block
    bad_config = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        # Missing encoder_blocks
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    with pytest.raises(
        ValueError,
        match="Config specifies an encoder, but 'encoder_blocks' are not defined.",
    ):
        build_time_series_transformer(bad_config)


def test_build_raises_for_missing_decoder_blocks():
    """
    Tests that the builder raises a ValueError if the layout requires a decoder
    but no decoder_blocks are provided.
    """
    bad_config = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(type="transformer_architecture", layout="decoder"),
        # Missing decoder_blocks
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    with pytest.raises(
        ValueError,
        match="Config specifies a decoder, but 'decoder_blocks' are not defined.",
    ):
        build_time_series_transformer(bad_config)


def test_build_raises_for_missing_loss_config():
    """
    Tests that the builder raises a ValueError if the loss_config is missing.
    """
    # FIX: Create a dict from a valid config, remove the key, then create a new config
    base_config_dict = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
        encoder_blocks=[
            EncoderBlockConfig(
                type="default_encoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
    ).to_dict()

    # Manually remove the default loss_config
    base_config_dict.pop("loss_config", None)
    bad_config = TransformerTimeSeriesConfig.from_dict(base_config_dict)

    with pytest.raises(
        ValueError,
        match="Config must have a 'loss_config' dictionary with a 'type' key.",
    ):
        build_time_series_transformer(bad_config)


def test_build_with_custom_registered_components(valid_decoder_only_config):
    """
    Tests that the builder can successfully use custom, registered components.
    """
    from temporal.registry.core import register_module

    # Define and register a custom block
    @register_module("block", "custom_test_block")
    class CustomBlock(nn.Module):
        def __init__(self, config, builder):
            super().__init__()
            self.layer = nn.Linear(config.d_model, config.d_model)

        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    # FIX: Create a new config from a dict instead of using .copy()
    custom_config_dict = valid_decoder_only_config.to_dict()
    custom_config_dict["decoder_blocks"] = [
        DecoderBlockConfig(
            type="custom_test_block",
            ffn_config=StandardFeedForwardConfig(
                type="standard", intermediate_size=32
            ),
        ).to_dict()
    ]
    custom_config = TransformerTimeSeriesConfig.from_dict(custom_config_dict)

    # This should build without errors
    model = build_time_series_transformer(custom_config)
    assert isinstance(model.decoder.layers[0], CustomBlock)