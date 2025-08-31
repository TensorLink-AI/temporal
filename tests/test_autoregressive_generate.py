import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.embedding_config import TimeSeriesPatchEmbeddingConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.loss_config import LossConfig

# --- Fixtures ---
@pytest.fixture(scope="module")
def patched_model_config():
    """Provides a config for a patched encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=1,
        prediction_length=6,
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
        value_embedding_config=TimeSeriesPatchEmbeddingConfig(patch_size=2, feature_size=3),
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=3),
    )


@pytest.fixture(scope="module")
def patched_model(patched_model_config):
    """Builds the patched model."""
    torch.manual_seed(0)
    return build_time_series_transformer(patched_model_config)


# --- New Detailed Tests ---
def test_cache_logic_equivalence(patched_model):
    """
    Asserts that generation with and without the KV cache produces numerically
    identical results. This is critical for validating the caching implementation.
    """
    # FIX: Put model in eval mode to disable dropout/layerdrop for a deterministic comparison
    patched_model.eval()
    config = patched_model.config
    batch_size = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)

    torch.manual_seed(42)
    predictions_with_cache = patched_model.generate(
        encoder_inputs=context,
        prediction_length=config.prediction_length,
        use_cache=True,
    )

    torch.manual_seed(42)
    predictions_without_cache = patched_model.generate(
        encoder_inputs=context,
        prediction_length=config.prediction_length,
        use_cache=False,
    )

    assert (
        predictions_with_cache.shape == predictions_without_cache.shape
    ), "Output shapes do not match"
    assert torch.allclose(
        predictions_with_cache, predictions_without_cache, atol=1e-5
    ), "Outputs from cached and non-cached generation are not numerically close."


def test_probabilistic_generation_quantiles(patched_model):
    """
    Tests probabilistic forecasting by checking quantile output shape and ordering.
    """
    patched_model.eval()
    config = patched_model.config
    batch_size = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)
    quantile_levels = [0.1, 0.5, 0.9]
    
    # This test requires a head that can produce quantiles. We'll replace the linear head.
    original_head = patched_model.output_heads
    from temporal.modules.heads.output_heads import GaussianHead
    patched_model.output_heads = torch.nn.ModuleDict({
        "point": GaussianHead(hidden_size=config.d_model, output_size=config.feature_size)
    })

    predictions = patched_model.generate(
        encoder_inputs=context,
        prediction_length=config.prediction_length,
        quantile_levels=quantile_levels,
    )

    expected_shape = (
        batch_size,
        config.prediction_length,
        config.feature_size,
        len(quantile_levels),
    )
    assert (
        predictions.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {predictions.shape}"

    # Restore the original head
    patched_model.output_heads = original_head


def test_generate_raises_error_on_no_input(patched_model):
    """
    Ensures that calling generate() with no input context raises a ValueError.
    """
    with pytest.raises(
        ValueError, match="You must provide either 'encoder_inputs' or 'decoder_inputs'."
    ):
        patched_model.generate(prediction_length=5)


def test_prediction_length_not_multiple_of_patch_size(patched_model):
    """
    Tests generation where prediction_length is not a perfect multiple of patch size.
    The model should still run and produce an output of the requested length.
    """
    patched_model.eval()
    config = patched_model.config
    batch_size = 2
    context = torch.randn(batch_size, config.context_length, config.feature_size)
    prediction_length = 7  # Not a multiple of patch size 2

    predictions = patched_model.generate(
        encoder_inputs=context, prediction_length=prediction_length
    )

    assert predictions.shape == (
        batch_size,
        prediction_length,
        config.feature_size,
    ), "Output shape does not match the requested non-multiple prediction length."