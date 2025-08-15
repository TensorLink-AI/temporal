# tests/test_embedders.py

import pytest
import torch
from temporal.modules.embedders.embedding import (
    SinusoidalPositionalEmbedding,
    LearnedAbsolutePositionalEmbedding,
    RotaryPositionalEmbedding,
    TimeSeriesPatchEmbedding,
    TimeSeriesValueEmbedding,
)

# --- Fixtures ---


@pytest.fixture(scope="module")
def embedding_params():
    """Provides common parameters for embedding tests."""
    return {
        "batch_size": 2,
        "seq_len": 20,
        "d_model": 32,
        "feature_size": 4,
        "max_seq_len": 100,
    }


# --- Value Embedding Tests ---


def test_value_embedding_forward(embedding_params):
    """Tests the forward pass of TimeSeriesValueEmbedding."""
    emb = TimeSeriesValueEmbedding(
        feature_size=embedding_params["feature_size"], d_model=embedding_params["d_model"]
    )
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["feature_size"],
    )
    output = emb(input_tensor)
    assert output.shape == (
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"],
    )


def test_value_embedding_with_norm(embedding_params):
    """Tests the value embedding with LayerNorm enabled."""
    emb = TimeSeriesValueEmbedding(
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"],
        use_value_norm=True,
    )
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["feature_size"],
    )
    output = emb(input_tensor)
    assert output.shape == (
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"],
    )


# --- Positional Embedding Tests ---


def test_sinusoidal_embedding_forward(embedding_params):
    """Tests the forward pass of SinusoidalPositionalEmbedding."""
    emb = SinusoidalPositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"],
    )
    output = emb(
        batch_size=embedding_params["batch_size"], seq_len=embedding_params["seq_len"]
    )
    # Sinusoidal embedding is broadcasted across the batch dimension
    assert output.shape == (
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"],
    )


def test_sinusoidal_embedding_caching(embedding_params):
    """Tests that the sinusoidal embedding cache is correctly used."""
    emb = SinusoidalPositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"],
    )
    # First call, cache should be populated
    emb(batch_size=embedding_params["batch_size"], seq_len=embedding_params["seq_len"])
    assert emb.cache.shape == (
        1,
        embedding_params["max_seq_len"],
        embedding_params["d_model"],
    )

    # Second call with smaller seq_len, should slice from cache
    output = emb(batch_size=embedding_params["batch_size"], seq_len=10)
    assert output.shape == (embedding_params["batch_size"], 10, embedding_params["d_model"])


def test_learned_embedding_forward(embedding_params):
    """Tests the forward pass of LearnedAbsolutePositionalEmbedding."""
    emb = LearnedAbsolutePositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"],
    )
    output = emb(
        batch_size=embedding_params["batch_size"], seq_len=embedding_params["seq_len"]
    )
    assert output.shape == (
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"],
    )


def test_rotary_embedding_init(embedding_params):
    """Tests the initialization of RotaryPositionalEmbedding."""
    # Should succeed with even d_model
    RotaryPositionalEmbedding(d_model=embedding_params["d_model"])
    # Should fail with odd d_model
    with pytest.raises(AssertionError):
        RotaryPositionalEmbedding(d_model=31)


def test_rotary_embedding_forward(embedding_params):
    """Tests the forward pass of RotaryPositionalEmbedding."""
    emb = RotaryPositionalEmbedding(d_model=embedding_params["d_model"])
    # Rotary expects a sequence length to generate angles up to that point
    cos, sin = emb(seq_len=embedding_params["seq_len"])
    assert cos.shape == (1, 1, embedding_params["seq_len"], embedding_params["d_model"])
    assert sin.shape == (1, 1, embedding_params["seq_len"], embedding_params["d_model"])


# --- Patch Embedding Tests ---


def test_patch_embedding_forward(embedding_params):
    """Tests the forward pass of TimeSeriesPatchEmbedding."""
    patch_size = 5
    emb = TimeSeriesPatchEmbedding(
        patch_size=patch_size,
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"],
    )
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"],  # 20
        embedding_params["feature_size"],
    )
    output = emb(input_tensor)
    assert output.shape == (
        embedding_params["batch_size"],
        embedding_params["seq_len"] // patch_size,
        embedding_params["d_model"],
    )


def test_patch_embedding_with_stride_and_padding(embedding_params):
    """Tests patching with a stride that requires padding."""
    patch_size = 5
    stride = 3
    emb = TimeSeriesPatchEmbedding(
        patch_size=patch_size,
        stride=stride,
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"],
    )

    seq_len = 19  # A length that will require padding
    input_tensor = torch.randn(
        embedding_params["batch_size"], seq_len, embedding_params["feature_size"]
    )
    output = emb(input_tensor)

    # Calculate expected number of patches: ceil((L - P) / S) + 1
    # Padded length L' = 21. num_patches = floor((21-5)/3) + 1 = floor(16/3)+1=5+1=6
    expected_num_patches = (
        (seq_len - patch_size + (patch_size - seq_len % stride)) // stride + 1
    )
    assert output.shape == (
        embedding_params["batch_size"],
        expected_num_patches,
        embedding_params["d_model"],
    )
