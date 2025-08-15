
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


@pytest.fixture
def generation_config():
    """Provides a standard config for an encoder-decoder model suitable for generation."""
    return TransformerTimeSeriesConfig(
        feature_size=1,
        d_model=16,
        num_encoder_layers=2,
        num_decoder_layers=2,
        context_length=10,
        prediction_length=5,
        architecture={"layout": "encoder-decoder"},
        loss_config={"type": "mse"},
        use_cache=True,  # Explicitly enable KV Caching
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
        encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
        decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
    )


@pytest.fixture
def generation_model(generation_config):
    """Provides a model instance for generation and serialization tests."""
    torch.manual_seed(0)
    return build_time_series_transformer(generation_config)


def test_kv_cache_correctness(generation_model, generation_config):
    """
    Tests that a forward pass with KV caching token-by-token produces the
    same result as a single forward pass with a causal mask.
    This is a critical test for ensuring autoregressive generation is correct.
    """
    model = generation_model
    model.eval()  # Ensure dropout is disabled

    batch_size = 2
    past_values = torch.randn(
        batch_size, generation_config.context_length, generation_config.feature_size
    )
    future_values = torch.randn(
        batch_size, generation_config.prediction_length, generation_config.feature_size
    )

    # 1. Forward pass without cache (full sequence at once)
    with torch.no_grad():
        full_pass_output = model(
            encoder_inputs=past_values, decoder_inputs=future_values
        )
    full_pass_logits = full_pass_output.logits

    # 2. Forward pass with cache (token-by-token)
    iterative_logits = []
    past_key_values = None

    # Encoder forward pass to get encoder_hidden_states
    preprocessor_output = model.preprocessor.process(
        past_values,
        # TODO: add support for past_features and future_features
        past_features=None,
        future_features=None,
    )
    encoder_output = model.encoder(**preprocessor_output)
    encoder_hidden_states = encoder_output

    with torch.no_grad():
        for i in range(generation_config.prediction_length):
            # Use the single next token as input to the decoder
            next_token_input = future_values[:, i : i + 1, :]

            output = model.decoder(
                input_values=next_token_input,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # The output of the decoder is the *next* token's representation
            hidden_state = output
            # Project to logits
            next_logit = model.output_heads["default"](hidden_state)

            iterative_logits.append(next_logit)
            past_key_values = output.past_key_values

    iterative_logits = torch.cat(iterative_logits, dim=1)

    # 3. Compare the results
    assert torch.allclose(full_pass_logits, iterative_logits, atol=1e-5), (
        "Logits from single forward pass and iterative pass with KV cache do not match."
    )


def test_model_serialization(generation_model):
    """
    Tests that the model can be saved and reloaded correctly.
    This validates the Hugging Face-style `save_pretrained` and `from_pretrained` methods.
    """
    model = generation_model
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save the model
        model.save_pretrained(tmpdir)

        # Check that essential files were created
        assert os.path.isfile(os.path.join(tmpdir, "config.json"))
        assert os.path.isfile(os.path.join(tmpdir, "pytorch_model.bin"))

        # 2. Load the model from the saved directory
        # We use the base class to call from_pretrained, as is standard
        reloaded_model = TransformerTemporalModel.from_pretrained(tmpdir)
        reloaded_model.eval()

        # 3. Verify that the reloaded model has the same architecture and weights
        assert reloaded_model.config.to_dict() == model.config.to_dict()

        for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
            assert torch.equal(
                p1, p2
            ), "Model parameters are not identical after reloading."


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

    # Generate a forecast
    with torch.no_grad():
        generated_sequence = model.generate(encoder_inputs=past_values)

    # Check the output shape
    expected_shape = (
        batch_size,
        generation_config.prediction_length,
        generation_config.feature_size,
    )
    assert generated_sequence.shape == expected_shape, (
        f"Generated sequence shape is incorrect. Expected {expected_shape}, got {generated_sequence.shape}"
    )

    # Test with a different batch size
    batch_size = 4
    past_values_b4 = torch.randn(
        batch_size, generation_config.context_length, generation_config.feature_size
    )
    with torch.no_grad():
        generated_sequence_b4 = model.generate(encoder_inputs=past_values_b4)
    expected_shape_b4 = (
        batch_size,
        generation_config.prediction_length,
        generation_config.feature_size,
    )
    assert generated_sequence_b4.shape == expected_shape_b4
