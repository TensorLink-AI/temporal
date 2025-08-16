import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.embedding_config import EmbeddingConfig, TimeSeriesPatchEmbeddingConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.loss_config import LossConfig

# --- Fixtures ---


@pytest.fixture(scope="module")
def encoder_decoder_config():
    """Provides a config for a standard encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=1,
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
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )

@pytest.fixture(scope="module")
def patched_config(encoder_decoder_config):
    """Provides a config with patch embedding."""
    config_dict = encoder_decoder_config.to_dict()
    # FIX: Instantiate the specific config class directly
    config_dict["value_embedding_config"] = TimeSeriesPatchEmbeddingConfig(
        patch_size=2, feature_size=1
    ).to_dict()
    return TransformerTimeSeriesConfig.from_dict(config_dict)

@pytest.fixture(scope="module")
def decoder_only_config():
    """Provides a config for a decoder-only model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=1,
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
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
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
        encoder_inputs=context, prediction_length=config.prediction_length
    )
    assert predictions.shape == (
        batch_size,
        config.prediction_length,
        config.feature_size,
    )


def test_generate_autoregressive_patched(patched_model):
    """Ensures generation works with the padding logic in the preprocessor."""
    config = patched_model.config
    batch_size = 2
    context_len = config.context_length + 1
    context = torch.randn(batch_size, context_len, config.feature_size)

    predictions = patched_model.generate(
        encoder_inputs=context, prediction_length=config.prediction_length
    )
    assert predictions.shape == (
        batch_size,
        config.prediction_length,
        config.feature_size,
    )


def test_generate_autoregressive_decoder_only(decoder_only_model):
    config = decoder_only_model.config
    batch_size = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = decoder_only_model.generate(
        encoder_inputs=context, prediction_length=config.prediction_length
    )
    assert predictions.shape == (
        batch_size,
        config.prediction_length,
        config.feature_size,
    )


# --- Multi-Step Generation Tests ---


def test_generate_multistep(encoder_decoder_model):
    """Tests iterative multi-step generation."""
    config = encoder_decoder_model.config
    batch_size = 2
    num_iterations = 3
    context = torch.randn(batch_size, config.context_length, config.feature_size)

    predictions = encoder_decoder_model.generate(
        encoder_inputs=context,  # Using consistent argument name
        num_iterations=num_iterations,
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
        encoder_inputs=context,  # Using consistent argument name
        num_iterations=num_iterations,
    )

    expected_pred_len = num_iterations * config.prediction_length
    assert predictions.shape == (batch_size, expected_pred_len, config.feature_size)