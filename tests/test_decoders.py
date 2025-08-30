import pytest
import torch
from temporal.modules.decoders.decoders import TimeSeriesTransformerDecoder
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import DecoderBlockConfig
from temporal.configs.attention_config import FullAttentionConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.embedding_config import EmbeddingConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.output_head_config import OutputHeadConfig


# --- Fixtures ---


@pytest.fixture
def decoder_config():
    """Provides a base TransformerTimeSeriesConfig for a decoder."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=4,
        prediction_length=5,
        context_length=10,
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                attention_config=FullAttentionConfig(type="full", num_heads=2),
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            ),
        ],
        value_embedding_config=EmbeddingConfig(type="value"),
        positional_embedding_config=EmbeddingConfig(
            type="sinusoidal", kwargs={"max_seq_len": 100}
        ),
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="decoder"
        ),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )


@pytest.fixture
def patched_decoder_config(decoder_config):
    """Provides a config with patch embedding."""
    config_dict = decoder_config.to_dict()
    config_dict["value_embedding_config"] = {"type": "patch", "kwargs": {"patch_size": 2}}
    return TransformerTimeSeriesConfig.from_dict(config_dict)


@pytest.fixture
def module_builder(decoder_config):
    return ModuleBuilder(decoder_config)


@pytest.fixture
def decoder(decoder_config, module_builder):
    return TimeSeriesTransformerDecoder(
        config=decoder_config,
        builder=module_builder,
        block_configs=decoder_config.decoder_blocks,
    )


@pytest.fixture
def patched_decoder(patched_decoder_config):
    builder = ModuleBuilder(patched_decoder_config)
    return TimeSeriesTransformerDecoder(
        config=patched_decoder_config,
        builder=builder,
        block_configs=patched_decoder_config.decoder_blocks,
    )


# --- Test Cases ---


def test_decoder_initialization(decoder, decoder_config):
    assert isinstance(decoder, TimeSeriesTransformerDecoder)
    assert len(decoder.layers) == len(decoder_config.decoder_blocks)


def test_decoder_forward_pass_shapes(decoder):
    batch_size, dec_seq_len, enc_seq_len = 2, 8, 10
    d_model = decoder.config.d_model

    hidden_states = torch.randn(batch_size, dec_seq_len, d_model)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, d_model)

    output = decoder(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=True,
    )

    assert hasattr(output, "last_hidden_state")
    assert output.last_hidden_state.shape == (batch_size, dec_seq_len, d_model)


def test_patched_decoder_forward_pass(patched_decoder):
    batch_size, dec_seq_len = 2, 8
    d_model = patched_decoder.config.d_model
    patch_size = patched_decoder.config.value_embedding_config.kwargs["patch_size"]

    hidden_states = torch.randn(batch_size, dec_seq_len // patch_size, d_model)

    output = patched_decoder(hidden_states=hidden_states, return_dict=True)
    assert output.last_hidden_state.shape == (
        batch_size,
        dec_seq_len // patch_size,
        d_model,
    )


def test_decoder_kv_caching_mechanism(decoder):
    batch_size, history_len, enc_seq_len = 2, 8, 10
    d_model = decoder.config.d_model

    history_embeds = torch.randn(batch_size, history_len, d_model)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, d_model)

    # First pass with history
    output_with_cache = decoder(
        hidden_states=history_embeds,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = output_with_cache.past_key_values

    # Second pass with a single new token
    new_token_embeds = torch.randn(batch_size, 1, d_model)
    output_stepwise = decoder(
        hidden_states=new_token_embeds,
        encoder_hidden_states=encoder_hidden_states,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    # For comparison, run without cache on the full sequence
    full_sequence_embeds = torch.cat([history_embeds, new_token_embeds], dim=1)
    output_full = decoder(
        hidden_states=full_sequence_embeds,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=False,
        return_dict=True,
    )

    # The last hidden state from the full pass should be very close to the
    # hidden state from the stepwise pass.
    assert torch.allclose(
        output_full.last_hidden_state[:, -1, :],
        output_stepwise.last_hidden_state[:, -1, :],
        atol=1e-5,
    )
