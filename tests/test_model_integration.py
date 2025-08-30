import pytest
import torch
import tempfile
import os
from temporal.models.builder import build_time_series_transformer
from temporal.models.transformer_model import TransformerTemporalModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.loss_config import TimeSeriesLossConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig


@pytest.fixture
def generation_config():
    """Provides a standard config for an encoder-decoder model suitable for generation."""
    ffn_config = StandardFeedForwardConfig(type="standard", intermediate_size=32)
    
    return TransformerTimeSeriesConfig(
        feature_size=1,
        d_model=16,
        context_length=10,
        prediction_length=5,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        loss_config=TimeSeriesLossConfig(loss_type="mse"),
        use_cache=True,
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
        encoder_blocks=[
            EncoderBlockConfig(type="default_encoder", ffn_config=ffn_config),
            EncoderBlockConfig(type="default_encoder", ffn_config=ffn_config),
        ],
        decoder_blocks=[
            DecoderBlockConfig(type="default_decoder", ffn_config=ffn_config),
            DecoderBlockConfig(type="default_decoder", ffn_config=ffn_config),
        ],
    )


@pytest.fixture
def generation_model(generation_config):
    """Provides a model instance for generation and serialization tests."""
    torch.manual_seed(0)
    return build_time_series_transformer(generation_config)


def test_model_serialization(generation_model: TransformerTemporalModel):
    """
    Tests that the model can be saved and reloaded correctly.
    """
    model = generation_model
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        assert os.path.isfile(os.path.join(tmpdir, "config.json"))
        assert os.path.isfile(os.path.join(tmpdir, "pytorch_model.bin"))

        reloaded_model = TransformerTemporalModel.from_pretrained(tmpdir)
        reloaded_model.eval()

        original_dict = model.config.to_dict()
        reloaded_dict = reloaded_model.config.to_dict()

        for key, value in original_dict.items():
            assert key in reloaded_dict, f"Key '{key}' missing from reloaded config"
            if key == "kwargs":
                continue
            assert reloaded_dict[key] == value, f"Config mismatch for key '{key}'"

def test_generation_output_shape(generation_model, generation_config):
    """
    Tests the `.generate()` method to ensure it produces outputs of the correct shape.
    """
    model = generation_model
    model.eval()

    batch_size = 2
    past_values = torch.randn(
        batch_size, generation_config.context_length, generation_config.feature_size
    )

    with torch.no_grad():
        generated_sequence = model.generate(encoder_inputs=past_values)

    expected_shape = (
        batch_size,
        generation_config.prediction_length,
        generation_config.feature_size,
    )
    assert generated_sequence.shape == expected_shape, (
        f"Generated sequence shape is incorrect. Expected {expected_shape}, got {generated_sequence.shape}"
    )

    batch_size = 4
    past_values_b4 = torch.randn(
        batch_size, generation_config.context_length, generation_config.feature_size
    )
    with torch.no_grad():
        generated_sequence_b4 = model.generate(encoder_inputs=past_values_b4)
        expected_shape_b4 = (
            batch_size,
            generation_config.prediction_length,
            generation_config.feature_s