import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig

@pytest.fixture
def basic_config():
    """Provides a basic, valid config that can be modified by tests."""
    return TransformerTimeSeriesConfig(
        feature_size=1,
        d_model=16,
        context_length=10,
        prediction_length=5,
        # This was the missing required argument
        architecture=TransformerArchitectureConfig(type="transformer_architecture")
    )

def test_multi_feature_forward_pass(basic_config):
    config_dict = basic_config.to_dict()
    config_dict["feature_size"] = 3
    config_dict["output_head_config"]["output_size"] = 3
    config_dict["value_embedding_config"]["feature_size"] = 3
    multi_feature_config = TransformerTimeSeriesConfig.from_dict(config_dict)
    model = build_time_series_transformer(multi_feature_config)
    encoder_inputs = torch.randn(2, multi_feature_config.context_length, multi_feature_config.feature_size)
    decoder_inputs = torch.randn(2, multi_feature_config.prediction_length, multi_feature_config.feature_size)
    output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
    assert output["point"].shape == (2, multi_feature_config.prediction_length, multi_feature_config.feature_size)
