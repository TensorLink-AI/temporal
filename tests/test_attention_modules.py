# tests/test_attention_modules.py

import pytest
import torch
from temporal.modules.attentions.base_attention import FullAttention
from temporal.modules.attentions.time_attention import TimeAttention
from temporal.modules.attentions.diff_attention import DifferentialAttention

# --- Fixtures ---

@pytest.fixture
def attention_config():
    """Provides a base configuration dictionary for attention modules."""
    return {
        "embed_dim": 32,
        "num_heads": 4,
        "dropout": 0.1,
    }

@pytest.fixture
def sample_tensors():
    """Provides sample tensors for testing attention modules."""
    batch_size = 2
    seq_len = 10
    embed_dim = 32
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    # Mask out the last two positions for testing
    attention_mask[:, :, :, -2:] = -torch.finfo(torch.float32).min
    return hidden_states, attention_mask

# --- FullAttention Tests ---

def test_full_attention_init(attention_config):
    """Tests the initialization of the FullAttention module."""
    attn = FullAttention(**attention_config)
    assert attn.embed_dim == attention_config["embed_dim"]
    assert attn.num_heads == attention_config["num_heads"]
    assert attn.head_dim == attention_config["embed_dim"] // attention_config["num_heads"]

def test_full_attention_forward_pass(attention_config, sample_tensors):
    """Tests the forward pass of FullAttention, checking output shape."""
    hidden_states, _ = sample_tensors
    attn = FullAttention(**attention_config)
    output, attn_probs, _ = attn(hidden_states)

    assert output.shape == hidden_states.shape
    assert attn_probs is None  # output_attentions is False by default

def test_full_attention_with_mask_and_kv_cache(attention_config, sample_tensors):
    """Tests FullAttention with an attention mask and KV caching."""
    hidden_states, attention_mask = sample_tensors
    attn = FullAttention(is_decoder=True, **attention_config)

    # First pass with full sequence
    output1, _, past_key_value = attn(
        hidden_states,
        attention_mask=attention_mask,
        use_cache=True
    )
    assert past_key_value is not None
    assert past_key_value[0].shape == (hidden_states.shape[0], attn.num_heads, hidden_states.shape[1], attn.head_dim)

    # Second pass with a single new token and the cache
    next_token = torch.randn(hidden_states.shape[0], 1, hidden_states.shape[2])
    output2, _, _ = attn(
        next_token,
        past_key_value=past_key_value,
        use_cache=True
    )
    assert output2.shape == next_token.shape

# --- TimeAttention Tests ---

def test_time_attention_init(attention_config):
    """Tests the initialization of the TimeAttention module."""
    attn = TimeAttention(**attention_config)
    assert hasattr(attn, "rotary_embed")
    assert hasattr(attn, "rel_pos_bias")

def test_time_attention_forward_pass(attention_config, sample_tensors):
    """Tests the forward pass of TimeAttention, checking output shape."""
    hidden_states, _ = sample_tensors
    attn = TimeAttention(**attention_config)
    output, _, _ = attn(hidden_states)
    assert output.shape == hidden_states.shape

# --- DifferentialAttention Tests ---

@pytest.fixture
def diff_attention_config(attention_config):
    """Provides a configuration for DifferentialAttention."""
    config = attention_config.copy()
    config["num_kv_heads"] = 2
    return config

def test_diff_attention_init(diff_attention_config):
    """Tests the initialization of DifferentialAttention."""
    attn = DifferentialAttention(**diff_attention_config)
    assert attn.num_kv_heads == 2
    assert attn.n_rep == diff_attention_config["num_heads"] // diff_attention_config["num_kv_heads"]

def test_diff_attention_forward_pass(diff_attention_config, sample_tensors):
    """Tests the forward pass of DifferentialAttention."""
    hidden_states, _ = sample_tensors
    attn = DifferentialAttention(**diff_attention_config)

    # DifferentialAttention requires a `rel_pos` argument
    seq_len = hidden_states.shape[1]
    head_dim = attn.head_dim
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)
    rel_pos = (cos, sin)

    output, _, _ = attn(hidden_states, rel_pos=rel_pos)
    assert output.shape == hidden_states.shape
