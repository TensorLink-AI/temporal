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
from temporal.registry.core import register_module
from temporal.configs.base_config import register_config_type, BaseConfig
from dataclasses import dataclass

@register_module("loss", "mse")
class DummyMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, preds, targets):
        return ((preds - targets) ** 2).mean()

@pytest.fixture
def valid_encoder_decoder_config():
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
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
        loss_config=LossConfig(type="mse"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )

@pytest.fixture
def valid_decoder_only_config():
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(type="transformer_architecture", layout="decoder"),
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        loss_config=LossConfig(type="mse"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )

def test_build_model_from_valid_config(valid_encoder_decoder_config):
    model = build_time_series_transformer(valid_encoder_decoder_config)
    assert isinstance(model, BaseTemporalModel)
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.output_heads is not None
    assert model.loss_fn is not None
    assert model.config == valid_encoder_decoder_config

def test_build_decoder_only_model(valid_decoder_only_config):
    model = build_time_series_transformer(valid_decoder_only_config)
    assert isinstance(model, BaseTemporalModel)
    assert model.encoder is None
    assert model.decoder is not None
    assert len(model.decoder.layers) == len(valid_decoder_only_config.decoder_blocks)

def test_build_raises_for_missing_encoder_blocks():
    bad_config = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        loss_config=LossConfig(type="mse"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    with pytest.raises(
        ValueError,
        match="Config specifies an encoder, but 'encoder_blocks' are not defined.",
    ):
        build_time_series_transformer(bad_config)

def test_build_raises_for_missing_decoder_blocks():
    bad_config = TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture=ArchitectureConfig(type="transformer_architecture", layout="decoder"),
        loss_config=LossConfig(type="mse"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    with pytest.raises(
        ValueError,
        match="Config specifies a decoder, but 'decoder_blocks' are not defined.",
    ):
        build_time_series_transformer(bad_config)

def test_build_with_missing_loss_config_succeeds(valid_encoder_decoder_config):
    bad_config_dict = valid_encoder_decoder_config.to_dict()
    del bad_config_dict["loss_config"]
    config_without_loss = TransformerTimeSeriesConfig.from_dict(bad_config_dict)
    model = build_time_series_transformer(config_without_loss)
    assert model.loss_fn is None

def test_build_with_custom_registered_components(valid_decoder_only_config):
    @dataclass(frozen=True)
    @register_config_type("custom_test_block")
    class CustomBlockConfig(BaseConfig):
        d_model: int = 16

    @register_module("block", "custom_test_block")
    class CustomBlock(nn.Module):
        def __init__(self, config, builder):
            super().__init__()
            self.layer = nn.Linear(config.d_model, config.d_model)

        def forward(self, hidden_states, **kwargs):
            return {"hidden_states": self.layer(hidden_states)}

    custom_config_dict = valid_decoder_only_config.to_dict()
    custom_config_dict["decoder_blocks"] = [{"type": "custom_test_block", "d_model": 16}]
    
    custom_config = TransformerTimeSeriesConfig.from_dict(custom_config_dict)
    
    model = build_time_series_transformer(custom_config)
    
    assert isinstance(model.decoder.layers[0], CustomBlock)
