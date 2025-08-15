# tests/test_forward.py

import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)

# --- Fixtures ---


@pytest.fixture(scope="module")
def encoder_decoder_config():
    """Provides a config for a standard encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture={"layout": "encoder-decoder"},
        encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
        loss_config={"type": "mse"},
    )


@pytest.fixture(scope="module")
def decoder_only_config():
    """Provides a config for a decoder-only model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture={"layout": "decoder"},
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
        loss_config={"type": "mse"},
    )


@pytest.fixture(scope="module")
def encoder_decoder_model(encoder_decoder_config):
    """Builds an encoder-decoder model from the config."""
    return build_time_series_transformer(encoder_decoder_config)


@pytest.fixture(scope="module")
def decoder_only_model(decoder_only_config):
    """Builds a decoder-only model from the config."""
    return build_time_series_transformer(decoder_only_config)


# --- Test Cases ---


def test_encoder_decoder_forward_pass(encoder_decoder_model):
    """Tests the end-to-end forward pass for an encoder-decoder model."""
    config = encoder_decoder_model.config
    batch_size = 2

    encoder_inputs = torch.randn(
        batch_size, config.context_length, config.feature_size
    )
    decoder_inputs = torch.randn(
        batch_size, config.prediction_length, config.feature_size
    )
    targets = torch.randn(batch_size, config.prediction_length, config.feature_size)

    output = encoder_decoder_model(
        encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, targets=targets
    )

    assert isinstance(output, dict) or hasattr(output, "logits")
    assert output.logits.shape == (
        batch_size,
        config.prediction_length,
        config.feature_size,
    )
    assert output.loss is not None
    assert output.loss.ndim == 0


def test_decoder_only_forward_pass(decoder_only_model):
    """Tests the end-to-end forward pass for a decoder-only model."""
    config = decoder_only_model.config
    batch_size = 2

    # In decoder-only, the input is just the decoder_input
    decoder_inputs = torch.randn(
        batch_size, config.context_length, config.feature_size
    )
    targets = torch.randn(batch_size, config.prediction_length, config.feature_size)

    output = decoder_only_model(decoder_inputs=decoder_inputs, targets=targets)

    assert isinstance(output, dict) or hasattr(output, "logits")
    # The output logits should correspond to the target length
    assert output.logits.shape == (
        batch_size,
        config.prediction_length,
        config.feature_size,
    )
    assert output.loss is not None
    assert output.loss.ndim == 0


def test_forward_pass_with_attention_mask(encoder_decoder_model):
    """Tests that attention masks are correctly processed without errors."""
    config = encoder_decoder_model.config
    batch_size = 2

    encoder_inputs = torch.randn(
        batch_size, config.context_length, config.feature_size
    )
    decoder_inputs = torch.randn(
        batch_size, config.prediction_length, config.feature_size
    )
    attention_mask = torch.ones(batch_size, config.context_length, dtype=torch.long)
    # Mask out the last few tokens of the encoder input
    attention_mask[:, -3:] = 0

    output = encoder_decoder_model(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        attention_mask=attention_mask,
    )

    assert output.logits.shape == (
        batch_size,
        config.prediction_length,
        config.feature_size,
    )


def test_forward_pass_error_missing_inputs(
    encoder_decoder_model, decoder_only_model
):
    """Tests that the model raises a ValueError if required inputs are missing."""

    # Test encoder-decoder model
    with pytest.raises(ValueError, match="requires 'encoder_inputs'"):
        encoder_decoder_model(decoder_inputs=torch.randn(2, 5, 3))

    with pytest.raises(ValueError, match="requires 'decoder_inputs'"):
        encoder_decoder_model(encoder_inputs=torch.randn(2, 10, 3))

    # Test decoder-only model
    with pytest.raises(ValueError, match="requires 'decoder_inputs'"):
        decoder_only_model()
