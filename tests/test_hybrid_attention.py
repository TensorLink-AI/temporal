# tests/test_hybrid_attention.py

import pytest
import torch
from temporal.modules.attentions.hybrid_attention import HybridAttention


def test_hybrid_attention_empty_head_splits_and_types():
    with pytest.raises(AssertionError):
        HybridAttention(dim=128, num_heads=4, head_splits=[], head_types=[])


def test_hybrid_attention_forward_pass():
    # Example usage with dummy data
    batch_size = 2
    seq_len = 10
    dim = 128
    num_heads = 4

    # Define head_splits and head_types (replace with actual values)
    head_splits = [1, 1, 1, 1]
    head_types = ["time", "time", "time", "time"]

    # Create a HybridAttention instance
    attention = HybridAttention(dim=dim, num_heads=num_heads, head_splits=head_splits, head_types=head_types)

    # Create dummy input data
    x = torch.randn(batch_size, seq_len, dim)

    # Perform the forward pass
    try:
        output = attention(x)
        assert output.shape == (batch_size, seq_len, dim)
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")


def test_hybrid_attention_different_head_splits_and_types_lengths():
    with pytest.raises(AssertionError):
        HybridAttention(dim=128, num_heads=4, head_splits=[1, 2], head_types=["time", "time", "time"])


def test_hybrid_attention_invalid_head_type():
     with pytest.raises(ValueError, match="Invalid attention type: invalid"):
        HybridAttention(dim=128, num_heads=4, head_splits=[1, 1, 1, 1], head_types=["invalid", "time", "time", "time"])


def test_hybrid_attention_zero_dimension():
    with pytest.raises(AssertionError):
        HybridAttention(dim=0, num_heads=4, head_splits=[1, 1, 1, 1], head_types=["time", "time", "time", "time"])

def test_hybrid_attention_head_splits_sum_not_equal_num_heads():
    with pytest.raises(AssertionError):
        HybridAttention(dim=128, num_heads=4, head_splits=[1, 1, 1, 2], head_types=["time", "time", "time", "time"])
