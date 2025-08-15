import pytest
import torch
from temporal.modules.embedders.embedding import (
    SinusoidalPositionalEmbedding,
    RotaryPositionalEmbedding,
    TimeSeriesValueEmbedding,
    TimeSeriesPatchEmbedding,
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

# --- Test Cases ---

def test_value_embedding_forward(embedding_params):
    """Tests the forward pass of TimeSeriesValueEmbedding."""
    emb = TimeSeriesValueEmbedding(
        feature_size=embedding_params["feature_size"],
        d_model=embedding_params["d_model"],
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

def test_sinusoidal_embedding_forward(embedding_params):
    """Tests the forward pass of SinusoidalPositionalEmbedding."""
    emb = SinusoidalPositionalEmbedding(
        d_model=embedding_params["d_model"],
        max_seq_len=embedding_params["max_seq_len"],
    )
    # The new implementation adds positional encodings to an input tensor
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"],
    )
    output = emb(input_tensor)
    assert output.shape == input_tensor.shape
    # Check that something was added
    assert not torch.allclose(input_tensor, output)

def test_rotary_embedding_forward(embedding_params):
    """Tests the forward pass of RotaryPositionalEmbedding."""
    emb = RotaryPositionalEmbedding(d_model=embedding_params["d_model"])
    input_tensor = torch.randn(
        embedding_params["batch_size"],
        embedding_params["seq_len"],
        embedding_params["d_model"],
    )
    # FIX: The forward pass now takes a tensor 'x' as input
    output = emb(input_tensor)
    assert output.shape == input_tensor.shape
    # Rotary embeddings modify the input tensor
    assert not torch.allclose(input_tensor, output)

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
    
    # Expected number of patches = seq_len / patch_size = 20 / 5 = 4
    num_patches = embedding_params["seq_len"] // patch_size
    assert output.shape == (
        embedding_params["batch_size"],
        num_patches,
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

    # Expected number of patches with padding and stride:
    # L_out = floor((L_in + 2*padding - patch_size) / stride) + 1
    # Padding is calculated to make L_in divisible by stride.
    # Padded length = 21. num_patches = floor((21-5)/3)+1 = 6
    expected_num_patches = 6
    assert output.shape == (
        embedding_params["batch_size"],
        expected_num_patches,
        embedding_params["d_model"],
    )