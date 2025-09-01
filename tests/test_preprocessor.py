import pytest
import torch
import logging
from unittest.mock import patch
from temporal.models.preprocessor import InputPreprocessor
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig, SinusoidalPositionalEmbeddingConfig, TimeSeriesPatchEmbeddingConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.output_head_config import OutputHeadConfig

@pytest.fixture
def base_config():
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=4,
        value_embedding_config=TimeSeriesValueEmbeddingConfig(type="value", feature_size=4),
        positional_embedding_config=SinusoidalPositionalEmbeddingConfig(
            type="sinusoidal", max_seq_len=100
        ),
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )

@pytest.fixture
def module_builder(base_config):
    return ModuleBuilder(base_config)

@pytest.fixture
def preprocessor(base_config, module_builder):
    return InputPreprocessor(base_config, module_builder)

def test_preprocessor_basic_pass(preprocessor):
    batch_size, seq_len, feature_size = 2, 10, 4
    input_values = torch.randn(batch_size, seq_len, feature_size)
    output = preprocessor.process(input_values)
    assert "hidden_states" in output
    assert "attention_mask" in output
    assert output["hidden_states"].shape == (
        batch_size,
        seq_len,
        preprocessor.config.d_model,
    )

def test_shape_validation_success(preprocessor):
    input_values = torch.randn(2, 10, 4)
    preprocessor.process(input_values, validate_shapes=True)

def test_shape_validation_failure(preprocessor, monkeypatch):
    input_values = torch.randn(2, 10, 4)
    def mock_pos_embedding(*args, **kwargs):
        return torch.randn(1, 5, preprocessor.config.d_model)
    monkeypatch.setattr(preprocessor.positional_embedding, "forward", mock_pos_embedding)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        preprocessor.process(input_values, validate_shapes=True)

def test_verbose_output(preprocessor, caplog):
    with caplog.at_level(logging.INFO):
        input_values = torch.randn(2, 10, 4)
        preprocessor.process(input_values, verbose=True)
    assert "Preprocessor: Processing input" in caplog.text

def test_patched_preprocessor_padding(base_config):
    patched_config_dict = base_config.to_dict()
    patched_config_dict["value_embedding_config"] = TimeSeriesPatchEmbeddingConfig(
        type="patch", patch_size=4, feature_size=4
    ).to_dict()
    patched_config = TransformerTimeSeriesConfig.from_dict(patched_config_dict)
    builder = ModuleBuilder(patched_config)
    preprocessor = InputPreprocessor(patched_config, builder)
    input_values = torch.randn(2, 11, 4)
    output = preprocessor.process(input_values, verbose=True)
    assert output["hidden_states"].shape[1] == 3
