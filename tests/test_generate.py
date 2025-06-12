# tests/test_generate.py

import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_config import TransformerTimeSeriesConfig, TransformerBlockConfig

# --- Fixtures ---

@pytest.fixture(scope="module")
def encoder_decoder_config():
    """Provides a config for a standard encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        num_heads=2,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture={"layout": "encoder-decoder"},
        decoder_start_token_id=0,  # Required for autoregressive generation
        encoder_blocks=[TransformerBlockConfig(block_type="default_encoder")],
        decoder_blocks=[TransformerBlockConfig(block_type="default_decoder")],
        loss_config={"type": "mse"}
    )

@pytest.fixture(scope="module")
def decoder_only_config():
    """Provides a config for a decoder-only model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        num_heads=2,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        architecture={"layout": "decoder"},
        decoder_blocks=[TransformerBlockConfig(block_type="default_decoder")],
        loss_config={"type": "mse"}
    )

@pytest.fixture(scope="module")
def encoder_decoder_model(encoder_decoder_config):
    """Builds an encoder-decoder model from the config."""
    return build_time_series_transformer(encoder_decoder_config)

@pytest.fixture(scope="module")
def decoder_only_model(decoder_only_config):
    """Builds a decoder-only model from the config."""
    return build_time_series_transformer(decoder_only_config)

# --- Autoregressive Generation Tests ---

def test_generate_autoregressive_encoder_decoder(encoder_decoder_model):
    """Tests autoregressive generation for an encoder-decoder model."""
    config = encoder_decoder_model.config
    batch_size = 2
    encoder_inputs = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = encoder_decoder_model.generate(
        input_values=encoder_inputs,
        prediction_length=config.prediction_length
    )

    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

def test_generate_autoregressive_decoder_only(decoder_only_model):
    """Tests autoregressive generation for a decoder-only model."""
    config = decoder_only_model.config
    batch_size = 2
    decoder_inputs = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = decoder_only_model.generate(
        input_values=decoder_inputs,
        prediction_length=config.prediction_length
    )

    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

def test_generate_autoregressive_no_cache(encoder_decoder_model):
    """Tests that autoregressive generation works correctly with use_cache=False."""
    config = encoder_decoder_model.config
    batch_size = 2
    encoder_inputs = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = encoder_decoder_model.generate(
        input_values=encoder_inputs,
        prediction_length=config.prediction_length,
        use_cache=False
    )

    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

# --- Multi-Step Generation Tests ---

def test_generate_multistep_encoder_decoder(encoder_decoder_model):
    """Tests direct multi-step generation for an encoder-decoder model."""
    config = encoder_decoder_model.config
    batch_size = 2
    encoder_inputs = torch.randn(batch_size, config.context_length, config.feature_size)
    
    # We need to access the multistep mixin method directly for this test
    predictions = encoder_decoder_model.generate_multistep(
        input_ids=encoder_inputs,
        prediction_length=config.prediction_length
    )

    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

def test_generate_multistep_decoder_only(decoder_only_model):
    """Tests direct multi-step generation for a decoder-only model."""
    config = decoder_only_model.config
    batch_size = 2
    decoder_inputs = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = decoder_only_model.generate_multistep(
        input_ids=decoder_inputs,
        prediction_length=config.prediction_length
    )

    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)
