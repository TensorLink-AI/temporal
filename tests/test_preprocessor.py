import pytest
import torch
from temporal.models.preprocessor import InputPreprocessor
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.embedding_config import EmbeddingConfig, TimeSeriesPatchEmbeddingConfig
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
        value_embedding_config=EmbeddingConfig(type="value"),
        positional_embedding_config=EmbeddingConfig(
            type="sinusoidal", kwargs={"max_seq_len": 100}
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
    # This should run without error
    preprocessor.process(input_values, validate_shapes=True)


def test_shape_validation_failure(preprocessor, monkeypatch):
    """
    Tests that shape validation raises an AssertionError on a mismatch.
    We use monkeypatch to simulate a misconfigured positional embedding.
    """
    input_values = torch.randn(2, 10, 4)

    # Simulate a positional embedding that returns an incorrect shape
    def mock_pos_embedding(*args, **kwargs):
        return torch.randn(1, 5, preprocessor.config.d_model)  # Mismatched seq_len

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
    # FIX: Create a new config from a dict instead of using deepcopy and modifying it.
    patched_config_dict = base_config.to_dict()
    patched_config_dict["value_embedding_config"] = TimeSeriesPatchEmbeddingConfig(
        patch_size=4, feature_size=4
    ).to_dict()
    patched_config = TransformerTimeSeriesConfig.from_dict(patched_config_dict)

    builder = ModuleBuilder(patched_config)
    preprocessor = InputPreprocessor(patched_config, builder)

    # Sequence length is not a multiple of patch_size
    input_values = torch.randn(2, 11, 4)

    output = preprocessor.process(input_values, verbose=True)

    # The output hidden states should have a sequence length of ceil(11/4) = 3
    assert output["hidden_states"].shape[1] == 3