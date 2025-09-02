import pytest
import torch.nn as nn
from dataclasses import dataclass
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
    TransformerBlockConfig,
)
from temporal.registry import register_module, register_config_type, CONFIG_REGISTRY
from temporal.configs.base_config import BaseConfig


@pytest.fixture
def valid_encoder_decoder_config():
    """Provides a valid, complete config for an encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        feature_size=3,
        d_model=16,
        context_length=10,
        prediction_length=5,
        architecture=TransformerArchitectureConfig(
            type="transformer_architecture",
            layout="encoder-decoder",
            num_encoder_layers=1,
            num_decoder_layers=1,
        ),
        encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
    )


@pytest.fixture
def valid_decoder_only_config():
    """Provides a valid config for a decoder-only model."""
    return TransformerTimeSeriesConfig(
        feature_size=3,
        d_model=16,
        context_length=10,
        prediction_length=5,
        architecture=TransformerArchitectureConfig(
            type="transformer_architecture", layout="decoder", num_decoder_layers=1
        ),
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
    )


def test_build_model_from_valid_config(valid_encoder_decoder_config):
    """
    Tests that a complete model can be built from a valid configuration.
    """
    model = build_time_series_transformer(valid_encoder_decoder_config)
    assert model is not None
    assert hasattr(model, "encoder") and model.encoder is not None
    assert hasattr(model, "decoder") and model.decoder is not None


def test_build_decoder_only_model(valid_decoder_only_config):
    """
    Tests that a decoder-only model is built correctly, with no encoder.
    """
    model = build_time_series_transformer(valid_decoder_only_config)
    assert model is not None
    assert not hasattr(model, "encoder") or model.encoder is None
    assert hasattr(model, "decoder") and model.decoder is not None


def test_build_with_custom_registered_components(valid_decoder_only_config):
    """
    Tests that the builder can successfully use custom, registered components.
    """

    @register_config_type("custom_test_block")
    @dataclass(frozen=True)
    class CustomBlockConfig(TransformerBlockConfig):
        d_model: int = 16

    @register_module("block", "custom_test_block")
    class CustomBlock(nn.Module):
        def __init__(self, config, builder):
            super().__init__()
            self.layer = nn.Linear(config.d_model, config.d_model)

        def forward(self, hidden_states, **kwargs):
            return {"hidden_states": self.layer(hidden_states)}

    custom_config_dict = valid_decoder_only_config.to_dict()
    custom_config_dict["decoder_blocks"] = [
        {"type": "custom_test_block", "d_model": 16}
    ]

    custom_config = TransformerTimeSeriesConfig.from_dict(custom_config_dict)
    model = build_time_series_transformer(custom_config)
    assert isinstance(model.decoder.layers[0], CustomBlock)
    
    # Clean up registry
    del CONFIG_REGISTRY["custom_test_block"]