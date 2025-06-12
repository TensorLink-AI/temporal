# tests/test_embedders.py

import pytest
import torch
from temporal.modules.embedders.embedding import (
    TimeSeriesValueEmbedding,
    SinusoidalPositionalEmbedding,
    RotaryPositionalEmbedding,
    LearnedAbsolutePositionalEmbedding,
    TimeSeriesPatchEmbedding,
    apply_rotary_pos_emb,
)

# --- Fixtures ---

@pytest.fixture
def embedding_params():
    """Provides common parameters for embedding layers."""
    return {
        "d_model": 32,
        "feature_size": 4,
        "batch_size": 2,
        "seq_len": 20,
        "max_seq_len": 100,
    }

# --- Value Embedding Tests ---

def test_value_embedding_init(embedding_params):
    """Tests the initialization of TimeSeriesValueEmbedding."""
    emb = TimeSeriesValueEmbedding(
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"]
    )
    assert emb.value_projection.in_features == embedding_params["feature_size"]
    assert emb.value_projection.out_features == embedding_params["d_model"]
    assert emb.value_norm is None

def test_value_embedding_forward(embedding_params):
    """Tests the forward pass of TimeSeriesValueEmbedding."""
    emb = TimeSeriesValueEmbedding(
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"]
    )
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["feature_size"]
    )
    output = emb(input_tensor)
    assert output.shape == (
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"]
    )

def test_value_embedding_with_norm(embedding_params):
    """Tests TimeSeriesValueEmbedding with layer normalization enabled."""
    emb = TimeSeriesValueEmbedding(
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"],
        use_value_norm=True
    )
    assert emb.value_norm is not None

# --- Positional Embedding Tests ---

def test_sinusoidal_embedding_forward(embedding_params):
    """Tests the forward pass of SinusoidalPositionalEmbedding."""
    emb = SinusoidalPositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"]
    )
    output = emb(
        batch_size=embedding_params["batch_size"],
        seq_len=embedding_params["seq_len"]
    )
    # Sinusoidal embedding is broadcasted across the batch dimension
    assert output.shape == (1, embedding_params["seq_len"], embedding_params["d_model"])

def test_sinusoidal_embedding_with_offset(embedding_params):
    """Tests SinusoidalPositionalEmbedding with a past_key_values_length offset."""
    emb = SinusoidalPositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"]
    )
    offset = 50
    output = emb(
        batch_size=embedding_params["batch_size"],
        seq_len=embedding_params["seq_len"],
        past_key_values_length=offset
    )
    # Ensure it doesn't just return the start of the embedding table
    first_pos_embedding = emb.weight[0]
    assert not torch.allclose(output[0, 0, :], first_pos_embedding)

def test_sinusoidal_embedding_out_of_bounds(embedding_params):
    """Tests that SinusoidalPositionalEmbedding raises an error for out-of-bounds requests."""
    emb = SinusoidalPositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"]
    )
    with pytest.raises(IndexError):
        emb(batch_size=1, seq_len=1, past_key_values_length=embedding_params["max_seq_len"])

def test_learned_embedding_forward(embedding_params):
    """Tests the forward pass of LearnedAbsolutePositionalEmbedding."""
    emb = LearnedAbsolutePositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"]
    )
    output = emb(
        batch_size=embedding_params["batch_size"],
        seq_len=embedding_params["seq_len"]
    )
    assert output.shape == (1, embedding_params["seq_len"], embedding_params["d_model"])


# --- Rotary Embedding (RoPE) Tests ---

def test_rotary_embedding_init(embedding_params):
    """Tests the initialization of RotaryPositionalEmbedding."""
    # Should succeed with even d_model
    RotaryPositionalEmbedding(d_model=embedding_params["d_model"])
    # Should fail with odd d_model
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(d_model=31)

def test_rotary_embedding_forward(embedding_params):
    """Tests the forward pass of RotaryPositionalEmbedding."""
    emb = RotaryPositionalEmbedding(d_model=embedding_params["d_model"])
    dummy_input = torch.randn(1, 1, embedding_params["d_model"])
    cos, sin = emb(dummy_input, seq_len=embedding_params["seq_len"])
    assert cos.shape == (embedding_params["seq_len"], embedding_params["d_model"])
    assert sin.shape == (embedding_params["seq_len"], embedding_params["d_model"])

def test_apply_rotary_pos_emb(embedding_params):
    """Tests the application of RoPE to query and key tensors."""
    rope = RotaryPositionalEmbedding(d_model=embedding_params["d_model"])
    dummy_input = torch.randn(1, 1, embedding_params["d_model"])
    cos, sin = rope(dummy_input, seq_len=embedding_params["seq_len"])
    
    q = torch.randn(
        embedding_params["batch_size"],
        4, # num_heads
        embedding_params["seq_len"],
        embedding_params["d_model"]
    )
    k = torch.randn_like(q)

    q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_emb.shape == q.shape
    assert k_emb.shape == k.shape
    # Check that the embeddings are different from the original
    assert not torch.allclose(q_emb, q)

# --- Patch Embedding Tests ---

def test_patch_embedding_forward(embedding_params):
    """Tests the forward pass of TimeSeriesPatchEmbedding."""
    patch_size = 5
    emb = TimeSeriesPatchEmbedding(
        patch_size=patch_size,
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"]
    )
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"], # 20
        embedding_params["feature_size"]
    )
    output = emb(input_tensor)
    
    num_patches = embedding_params["seq_len"] // patch_size
    assert output.shape == (
        embedding_params["batch_size"],
        num_patches, # 20 / 5 = 4
        embedding_params["d_model"]
    )

def test_patch_embedding_with_stride_and_padding(embedding_params):
    """Tests patching with a stride that requires padding."""
    patch_size = 5
    stride = 3
    emb = TimeSeriesPatchEmbedding(
        patch_size=patch_size,
        stride=stride,
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"]
    )
    
    seq_len = 19 # A length that will require padding
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        seq_len,
        embedding_params["feature_size"]
    )
    output = emb(input_tensor)
    
    # Expected number of patches: floor((19 + pad - 5) / 3) + 1
    # pad = 3 - ((19-5) % 3) = 3 - (14 % 3) = 3 - 2 = 1. Padded length = 20.
    # num_patches = floor((20-5)/3) + 1 = floor(5) + 1 = 6
    expected_num_patches = 6
    assert output.shape[1] == expected_num_patches
