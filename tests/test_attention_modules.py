# tests/test_attention_modules.py

import pytest
import torch
from temporal.modules.attentions.base_attention import FullAttention
from temporal.modules.attentions.destationary import DestationaryAttention
from temporal.modules.attentions.diff_attention import DifferentialAttention

# --- Fixtures ---

@pytest.fixture
def attention_config():
    """Provides a base configuration dictionary for attention modules."""
    return {
        "d_model": 32,
        "n_heads": 4,
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
    assert attn.d_model == attention_config["d_model"]
    assert attn.n_heads == attention_config["n_heads"]
    assert attn.head_dim == attention_config["d_model"] // attention_config["n_heads"]

def test_full_attention_forward_pass(attention_config, sample_tensors):
    """Tests the forward pass of FullAttention, checking output shape."""
    hidden_states, _ = sample_tensors
    attn = FullAttention(**attention_config)
    output, attn_probs = attn(hidden_states)

    assert output.shape == hidden_states.shape
    assert attn_probs is None  # output_attentions is False by default

def test_full_attention_with_mask_and_kv_cache(attention_config, sample_tensors):
    """Tests FullAttention with an attention mask and KV caching."""
    hidden_states, attention_mask = sample_tensors
    # is_decoder equivalent
    attn = FullAttention(is_causal=True, **attention_config)

    # First pass with full sequence
    output1, _, past_key_value = attn(
        hidden_states,
        attention_mask=attention_mask,
        use_cache=True
    )
    assert past_key_value is not None
    assert past_key_value[0].shape == (hidden_states.shape[0], attn.n_heads, hidden_states.shape[1], attn.head_dim)

    # Second pass with a single new token and the cache
    next_token = torch.randn(hidden_states.shape[0], 1, hidden_states.shape[2])
    output2, _, _ = attn(
        next_token,
        past_key_value=past_key_value,
        use_cache=True
    )
    assert output2.shape == next_token.shape

# --- DestationaryAttention Tests ---

def test_destationary_attention_init(attention_config):
    """Tests the initialization of the DestationaryAttention module."""
    attn = DestationaryAttention(**attention_config)
    assert attn.tau.shape == (attn.n_heads, 1, 1)
    assert attn.delta.shape == (attn.n_heads, 1, 1)

def test_destationary_attention_forward_pass(attention_config, sample_tensors):
    """Tests the forward pass of DestationaryAttention."""
    hidden_states, _ = sample_tensors
    attn = DestationaryAttention(**attention_config)
    output, _ = attn(hidden_states)
    assert output.shape == hidden_states.shape

def test_destationary_logic():
    """
    Tests the core de-stationarization logic with predictable inputs.
    """
    n_heads = 2
    seq_len = 4
    attn = DestationaryAttention(d_model=32, n_heads=n_heads)

    # Initialize tau and delta to predictable values (ones)
    torch.nn.init.ones_(attn.tau)
    torch.nn.init.ones_(attn.delta)

    # Create a simple, non-normalized attention score matrix
    # Shape: (batch_size, n_heads, seq_len, seq_len)
    attn_scores = torch.ones(1, n_heads, seq_len, seq_len)

    # Call the destationarize method
    destationarized_weights = attn.destationarize(attn_scores)

    # Manually calculate the expected output
    # tau = 1, so cumulative sum is [1, 2, 3, 4]
    # The weights should be exp(scores) * exp(cumsum(tau))
    # Since scores are 1, exp(scores) is e.
    # Expected weights = [e*e^1, e*e^2, e*e^3, e*e^4] -> [e^2, e^3, e^4, e^5]
    # The values should be identical across the sequence length dimension (dim -1)
    # and then normalized by the sum along that dimension.
    expected_cumsum = torch.tensor([1., 2., 3., 4.]).view(1, 1, -1)
    expected_unnorm = torch.exp(torch.ones(seq_len, seq_len) + expected_cumsum)
    expected_normalized = expected_unnorm / expected_unnorm.sum(dim=-1, keepdim=True)

    assert destationarized_weights.shape == attn_scores.shape
    assert torch.allclose(destationarized_weights[0, 0], expected_normalized, atol=1e-6)


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
    assert attn.n_rep == diff_attention_config["n_heads"] // diff_attention_config["num_kv_heads"]

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
