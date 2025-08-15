# tests/test_decoders.py

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
    return decoder_config.model_copy(
        update={"value_embedding_config": EmbeddingConfig(type="patch", kwargs={"patch_size": 2})}
    )


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
    # We need a new builder for the patched config
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

    # --- Step 1: Process initial sequence ---
    output_with_cache = decoder(
        hidden_states=history_embeds,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=True,
        return_dict=True,
    )
    assert output_with_cache.past_key_values is not None

    # --- Step 2: Process a single new token ---
    new_token_embed = torch.randn(batch_size, 1, d_model)

    output_next_step = decoder(
        hidden_states=new_token_embed,
        encoder_hidden_states=encoder_hidden_states,
        past_key_values=output_with_cache.past_key_values,
        use_cache=True,
        return_dict=True,
    )

    assert output_next_step.last_hidden_state.shape == (batch_size, 1, d_model)
    new_cache = output_next_step.past_key_values
    assert new_cache[0][0][0].shape[-2] == history_len + 1  # Check sequence length dim


def test_decoder_output_flags(decoder):
    batch_size, dec_seq_len = 2, 8
    d_model = decoder.config.d_model
    hidden_states = torch.randn(batch_size, dec_seq_len, d_model)

    # Test with flags enabled
    output_full = decoder(
        hidden_states=hidden_states,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    assert output_full.attentions is not None
    assert output_full.hidden_states is not None
    assert len(output_full.hidden_states) == len(decoder.layers) + 1

    # Test with flags disabled
    output_simple = decoder(hidden_states=hidden_states, return_dict=True)
    assert output_simple.attentions is None
    assert output_simple.hidden_states is None
