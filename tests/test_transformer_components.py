import pytest
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import (
    TransformerTimeSeriesConfig,
    DecoderBlockConfig,
)
from temporal.configs.architecture_config import TransformerArchitectureConfig


@pytest.fixture
def basic_config():
    return TransformerTimeSeriesConfig(
        feature_size=1,
        d_model=16,
        context_length=10,
        prediction_length=5,
        architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="decoder"),
        # Use proper config objects (not dicts)
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
    )


def test_multi_feature_forward_pass(basic_config):
    config_dict = basic_config.to_dict()
    config_dict["feature_size"] = 3
    config_dict.setdefault("value_embedding_config", {}).update({"feature_size": 3})
    config_dict.setdefault("output_head_config", {}).update({"output_size": 3})
    # decoder layout already has a block in the fixture; no extra needed

    multi_feature_config = TransformerTimeSeriesConfig.from_dict(config_dict)
    model = build_time_series_transformer(multi_feature_config)
    assert model is not None
