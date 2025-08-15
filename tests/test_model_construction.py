# tests/test_model_construction.py

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
from temporal.configs.loss_config import LossConfig


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
        encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
        loss_config=LossConfig(type="mse"),
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
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
        loss_config=LossConfig(type="mse"),
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
    bad_config = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        # Missing encoder_blocks
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
        loss_config=LossConfig(type="mse"),
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
        loss_config=LossConfig(type="mse"),
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
    bad_config = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
    )
    # Manually remove the default loss_config
    bad_config.loss_config = None
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

    # Update the config to use the custom block
    custom_config = valid_decoder_only_config.copy(deep=True)
    custom_config.decoder_blocks = [
        DecoderBlockConfig(type="custom_test_block")
    ]

    # This should build without errors
    model = build_time_series_transformer(custom_config)
    assert isinstance(model.decoder.layers[0], CustomBlock)
