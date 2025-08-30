import pytest
import torch
from temporal.models.preprocessor import InputPreprocessor
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
# FIX: Import specific embedding configs instead of the base class
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig, SinusoidalPositionalEmbeddingConfig, TimeSeriesPatchEmbeddingConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.output_head_config import OutputHeadConfig


@pytest.fixture
def base_config():
    """Provides a base config for the preprocessor."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=4,
        # FIX: Instantiate the correct, specific config classes
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
    """Tests a standard forward pass without validation."""
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
    """Tests that shape validation passes when shapes are correct."""
    input_values = torch.randn(2, 10, 4)
    preprocessor.process(input_values, validate_shapes=True)


def test_shape_validation_failure(preprocessor, monkeypatch):
    """
    Tests that shape validation raises an AssertionError on a mismatch.
    """
    input_values = torch.randn(2, 10, 4)

    def mock_pos_embedding(*args, **kwargs):
        return torch.randn(1, 5, preprocessor.config.d_model)

    monkeypatch.setattr(preprocessor.positional_embedding, "forward", mock_pos_embedding)

    with pytest.raises(AssertionError, match="Shape mismatch"):
        preprocessor.process(input_values, validate_shapes=True)


def test_verbose_output(preprocessor, capsys):
    """Tests that the verbose flag prints shape information."""
    input_values = torch.randn(2, 10, 4)

    preprocessor.process(input_values, verbose=True)

    captured = capsys.readouterr()
    assert "[Preprocessor] Initial input shape" in captured.out
    assert "[Preprocessor] Value embedding shape" in captured.out
    assert "[Preprocessor] Positional embedding shape" in captured.out
    assert "[Preprocessor] Final hidden_states shape" in captured.out


def test_patched_preprocessor_padding(base_config):
    """Tests that the preprocessor correctly pads for patch embedding."""
    # FIX: Correctly create the patched config from the base config's dictionary
    patched_config_dict = base_config.to_dict()
    patched_config_dict["value_embedding_config"] = TimeSeriesPatchEmbeddingConfig(
        type="patch", patch_size=4, feature_size=4
    ).to_dict()
    patched_config = TransformerTimeSeriesConfig.from_dict(patched_config_dict)

    builder = ModuleBuilder(patched_config)
    preprocessor = InputPreprocessor(patched_config, builder)

    input_values = torch.randn(2, 11, 4)

    output = preprocessor.process(input_values, verbose=True)

    # After padding to 12, 12/4 = 3 patches
    assert output["hidden_states"].shape[1] == 3