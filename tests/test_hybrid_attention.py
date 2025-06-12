# tests/test_hybrid_attention.py

import pytest
import torch
from temporal.modules.attentions.hybrid_attention import HybridAttention
from temporal.modules.attentions.base_attention import FullAttention

# --- Fixtures ---

@pytest.fixture
def hybrid_config():
    """Provides a base configuration for the HybridAttention module."""
    return {
        "embed_dim": 32,
        "num_heads": 8, # Must be sum of head_splits
        "head_splits": [4, 4],
        "head_types": ["full", "full"], # Using simple attention for testing
    }

@pytest.fixture
def sample_tensor(hybrid_config):
    """Provides a sample input tensor."""
    batch_size = 2
    seq_len = 10
    return torch.randn(batch_size, seq_len, hybrid_config["embed_dim"])

# --- Test Cases ---

def test_hybrid_attention_init(hybrid_config):
    """Tests the initialization of the HybridAttention module."""
    hybrid_attn = HybridAttention(**hybrid_config)
    assert len(hybrid_attn.head_groups) == len(hybrid_config["head_splits"])
    assert hybrid_attn.group_count == len(hybrid_config["head_splits"])
    # Check that the sub-modules are the correct type
    assert all(isinstance(group, FullAttention) for group in hybrid_attn.head_groups)

def test_hybrid_attention_init_validation_errors(hybrid_config):
    """Tests that initialization raises errors for invalid configurations."""
    
    # Mismatched head_splits sum
    bad_config_sum = hybrid_config.copy()
    bad_config_sum["head_splits"] = [3, 3] # Sums to 6, not 8
    with pytest.raises(ValueError, match="Sum of head_splits"):
        HybridAttention(**bad_config_sum)
        
    # Mismatched lengths of splits and types
    bad_config_len = hybrid_config.copy()
    bad_config_len["head_types"] = ["full"] # Length 1, expected 2
    with pytest.raises(ValueError, match="Length of head_splits must match"):
        HybridAttention(**bad_config_len)

def test_hybrid_attention_forward_pass_concat(hybrid_config, sample_tensor):
    """Tests the forward pass with the default 'concat' aggregator."""
    hybrid_attn = HybridAttention(**hybrid_config)
    output, _, _ = hybrid_attn(sample_tensor)
    
    assert output.shape == sample_tensor.shape

def test_hybrid_attention_forward_pass_custom_aggregator(hybrid_config, sample_tensor):
    """Tests the forward pass with a mock custom aggregator."""
    from temporal.registry.core import register_module
    from unittest.mock import MagicMock

    # Create and register a mock aggregator
    @register_module("head_agg", "mock_agg")
    class MockAggregator(torch.nn.Module):
        def __init__(self, embed_dim, num_groups, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.forward_mock = MagicMock(return_value=torch.randn_like(sample_tensor))

        def forward(self, head_outputs):
            return self.forward_mock(head_outputs)

    config = hybrid_config.copy()
    config["head_agg"] = "mock_agg"
    
    hybrid_attn = HybridAttention(**config)
    
    # Ensure the mock aggregator was instantiated
    assert isinstance(hybrid_attn.head_aggregator, MockAggregator)
    
    output, _, _ = hybrid_attn(sample_tensor)
    
    # Check that the aggregator's forward method was called
    hybrid_attn.head_aggregator.forward_mock.assert_called_once()
    # Check that the output is the one from our mock
    assert torch.allclose(output, hybrid_attn.head_aggregator.forward_mock.return_value)

def test_hybrid_attention_kv_caching(hybrid_config, sample_tensor):
    """Tests the key-value caching mechanism for HybridAttention."""
    hybrid_attn = HybridAttention(is_decoder=True, **hybrid_config)
    
    # First pass
    output1, _, past_key_values = hybrid_attn(sample_tensor, use_cache=True)
    
    assert past_key_values is not None
    assert isinstance(past_key_values, list)
    assert len(past_key_values) == len(hybrid_config["head_splits"])
    
    # Second pass
    next_token = torch.randn(sample_tensor.shape[0], 1, sample_tensor.shape[2])
    output2, _, new_past_key_values = hybrid_attn(
        next_token,
        past_key_value=past_key_values,
        use_cache=True
    )
    
    assert output2.shape == next_token.shape
    assert len(new_past_key_values) == len(past_key_values)
