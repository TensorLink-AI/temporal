# tests/test_decoders.py

import pytest
import torch
from temporal.modules.decoders.decoders import TimeSeriesTransformerDecoder
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_config import (
    TransformerTimeSeriesConfig,
    TransformerBlockConfig,
    AttentionConfig,
    FeedForwardConfig,
    EmbeddingConfig,
    NormalizationConfig,
)

# --- Fixtures ---

@pytest.fixture
def decoder_config():
    """Provides a base TransformerTimeSeriesConfig for an encoder-decoder model."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        num_heads=2,
        feature_size=4,
        prediction_length=5,
        context_length=10,
        architecture={
            "layout": "encoder-decoder"
        },
        decoder_blocks=[
            TransformerBlockConfig(
                block_type="default_decoder",
                attention_config=AttentionConfig(attention_type="full", num_heads=2),
                ffn_config=FeedForwardConfig(type="standard", intermediate_size=32),
            ),
            TransformerBlockConfig(
                block_type="default_decoder",
                attention_config=AttentionConfig(attention_type="full", num_heads=2),
                ffn_config=FeedForwardConfig(type="standard", intermediate_size=32),
            ),
        ],
        value_embedding_config=EmbeddingConfig(type="value"),
        positional_embedding_config=EmbeddingConfig(type="sinusoidal", kwargs={"max_seq_len": 100}),
        norm_config=NormalizationConfig(norm_type="layer_norm"),
    )

@pytest.fixture
def module_builder(decoder_config):
    """Provides an instance of ModuleBuilder based on the decoder_config."""
    return ModuleBuilder(decoder_config)

@pytest.fixture
def decoder(decoder_config, module_builder):
    """Provides an instantiated TimeSeriesTransformerDecoder."""
    return TimeSeriesTransformerDecoder(
        config=decoder_config,
        builder=module_builder,
        block_configs=decoder_config.decoder_blocks,
    )

# --- Test Cases ---

def test_decoder_initialization(decoder, decoder_config):
    """
    Tests that the decoder initializes correctly with the right number of layers.
    """
    assert isinstance(decoder, TimeSeriesTransformerDecoder)
    assert len(decoder.layers) == len(decoder_config.decoder_blocks)
    print(f"Decoder initialized with {len(decoder.layers)} layers.")

def test_decoder_forward_pass_shapes(decoder):
    """
    Tests the basic forward pass, ensuring output shapes are correct.
    """
    batch_size, dec_seq_len, enc_seq_len = 2, 8, 10
    d_model = decoder.config.d_model
    feature_size = decoder.config.feature_size

    decoder_inputs = torch.randn(batch_size, dec_seq_len, feature_size)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, d_model)

    output = decoder(
        input_ids=decoder_inputs,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=True
    )

    assert hasattr(output, 'last_hidden_state')
    assert output.last_hidden_state.shape == (batch_size, dec_seq_len, d_model)

def test_decoder_forward_without_encoder_states(decoder):
    """
    Tests that the decoder can run without encoder hidden states (for self-attention only).
    """
    batch_size, dec_seq_len = 2, 8
    d_model = decoder.config.d_model
    feature_size = decoder.config.feature_size
    decoder_inputs = torch.randn(batch_size, dec_seq_len, feature_size)

    output = decoder(input_ids=decoder_inputs, encoder_hidden_states=None, return_dict=True)
    assert output.last_hidden_state.shape == (batch_size, dec_seq_len, d_model)

def test_decoder_kv_caching_mechanism(decoder):
    """
    Tests the Key-Value caching mechanism for efficient autoregressive generation.
    """
    batch_size, history_len, enc_seq_len = 2, 8, 10
    d_model = decoder.config.d_model
    feature_size = decoder.config.feature_size

    history_inputs = torch.randn(batch_size, history_len, feature_size)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, d_model)

    # --- Step 1: Process initial sequence with use_cache=True ---
    output_with_cache = decoder(
        input_ids=history_inputs,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=True,
        return_dict=True
    )

    assert hasattr(output_with_cache, 'past_key_values')
    assert output_with_cache.past_key_values is not None
    # Check structure: list of layers, each with (self_kv, cross_kv)
    assert len(output_with_cache.past_key_values) == len(decoder.layers)
    # Check self-attention cache shape (key tensor)
    assert output_with_cache.past_key_values[0][0][0].shape == (batch_size, decoder.config.num_heads, history_len, d_model // decoder.config.num_heads)
    # Check cross-attention cache shape (key tensor)
    assert output_with_cache.past_key_values[0][1][0].shape == (batch_size, decoder.config.num_heads, enc_seq_len, d_model // decoder.config.num_heads)

    # --- Step 2: Process a single new token using the cache ---
    new_token_input = torch.randn(batch_size, 1, feature_size)
    output_next_step = decoder(
        input_ids=new_token_input,
        encoder_hidden_states=encoder_hidden_states,
        past_key_values=output_with_cache.past_key_values,
        use_cache=True,
        return_dict=True
    )

    # The output hidden state should only be for the new token
    assert output_next_step.last_hidden_state.shape == (batch_size, 1, d_model)

    # The new cache should now have a sequence length of `history_len + 1`
    new_cache = output_next_step.past_key_values
    assert new_cache[0][0][0].shape == (batch_size, decoder.config.num_heads, history_len + 1, d_model // decoder.config.num_heads)
    # Cross-attention cache length should remain unchanged
    assert new_cache[0][1][0].shape == (batch_size, decoder.config.num_heads, enc_seq_len, d_model // decoder.config.num_heads)

def test_decoder_output_flags(decoder):
    """
    Tests the output_attentions and output_hidden_states flags.
    """
    batch_size, dec_seq_len = 2, 8
    feature_size = decoder.config.feature_size
    decoder_inputs = torch.randn(batch_size, dec_seq_len, feature_size)

    # Test with flags enabled
    output_full = decoder(
        input_ids=decoder_inputs,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True
    )
    assert output_full.attentions is not None
    assert isinstance(output_full.attentions, tuple)
    assert output_full.hidden_states is not None
    assert isinstance(output_full.hidden_states, tuple)
    assert len(output_full.hidden_states) == len(decoder.layers) + 1 # num_layers + initial embedding

    # Test with flags disabled (default)
    output_simple = decoder(
        input_ids=decoder_inputs,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )
    assert output_simple.attentions is None
    assert output_simple.hidden_states is None
