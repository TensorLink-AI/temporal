# tests/test_generate.py

import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_config import TransformerTimeSeriesConfig, TransformerBlockConfig, EmbeddingConfig

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
        encoder_blocks=[TransformerBlockConfig(block_type="default_encoder")],
        decoder_blocks=[TransformerBlockConfig(block_type="default_decoder")],
        loss_config={"type": "mse"}
    )

@pytest.fixture(scope="module")
def patched_config(encoder_decoder_config):
    """Provides a config with patch embedding."""
    from copy import deepcopy
    cfg = deepcopy(encoder_decoder_config)
    cfg.value_embedding_config = EmbeddingConfig(type="patch", kwargs={"patch_size": 2})
    return cfg

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
    return build_time_series_transformer(encoder_decoder_config)

@pytest.fixture(scope="module")
def patched_model(patched_config):
    return build_time_series_transformer(patched_config)

@pytest.fixture(scope="module")
def decoder_only_model(decoder_only_config):
    return build_time_series_transformer(decoder_only_config)

# --- Autoregressive Generation Tests ---

def test_generate_autoregressive_encoder_decoder(encoder_decoder_model):
    config = encoder_decoder_model.config
    batch_size = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = encoder_decoder_model.generate(
        context=context,
        prediction_length=config.prediction_length
    )
    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

def test_generate_autoregressive_patched(patched_model):
    """Ensures generation works with the padding logic in the preprocessor."""
    config = patched_model.config
    batch_size = 2
    context_len = config.context_length + 1
    context = torch.randn(batch_size, context_len, config.feature_size)

    predictions = patched_model.generate(
        context=context,
        prediction_length=config.prediction_length
    )
    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

def test_generate_autoregressive_decoder_only(decoder_only_model):
    config = decoder_only_model.config
    batch_size = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = decoder_only_model.generate(
        context=context,
        prediction_length=config.prediction_length
    )
    assert predictions.shape == (batch_size, config.prediction_length, config.feature_size)

# --- Multi-Step Generation Tests ---

def test_generate_multistep(encoder_decoder_model):
    """Tests iterative multi-step generation."""
    config = encoder_decoder_model.config
    batch_size = 2
    num_iterations = 3
    context = torch.randn(batch_size, config.context_length, config.feature_size)
    
    predictions = encoder_decoder_model.generate(
        context=context, # Using consistent argument name
        num_iterations=num_iterations
    )
    
    expected_pred_len = num_iterations * config.prediction_length
    assert predictions.shape == (batch_size, expected_pred_len, config.feature_size)

def test_generate_multistep_patched(patched_model):
    """Tests iterative multi-step generation with patch embeddings."""
    config = patched_model.config
    batch_size = 2
    num_iterations = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)
    
    predictions = patched_model.generate(
        context=context, # Using consistent argument name
        num_iterations=num_iterations
    )
    
    expected_pred_len = num_iterations * config.prediction_length
    assert predictions.shape == (batch_size, expected_pred_len, config.feature_size)
