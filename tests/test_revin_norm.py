# tests/test_revin_norm.py

import pytest
import torch
from temporal.modules.norm.revin import RevIN as Revin
from temporal.modules.norm.dynamic_revin import DynamicRevIN as DynamicRevin

# --- Fixtures ---


@pytest.fixture
def sample_tensor():
    """Provides a sample input tensor with non-zero mean and non-one std."""
    # Shape: (batch_size, seq_len, feature_size)
    tensor = torch.randn(4, 20, 5) * 8 + 3
    return tensor


# --- Revin Tests ---


def test_revin_init():
    """Tests the initialization of the Revin layer."""
    layer = Revin(num_features=5)
    assert isinstance(layer.affine_weight, torch.nn.Parameter)
    assert isinstance(layer.affine_bias, torch.nn.Parameter)
    assert layer.mean is None
    assert layer.stdev is None


def test_revin_reversibility(sample_tensor):
    """
    Tests that denormalize(normalize(x)) returns x. This is the most
    critical test for this module.
    """
    num_features = sample_tensor.shape[-1]
    layer = Revin(num_features=num_features)

    # 1. Normalize the tensor
    # The `normalize` method in Revin typically takes a (B, T, D) tensor
    # and stores the stats from the last time step.
    normalized_output = layer(sample_tensor, mode="norm")

    # 2. Denormalize the output
    denormalized_output = layer(normalized_output, mode="denorm")

    # 3. Assert that the result is numerically identical to the original
    assert normalized_output.shape == sample_tensor.shape
    assert denormalized_output.shape == sample_tensor.shape
    assert torch.allclose(denormalized_output, sample_tensor, atol=1e-6), (
        "Revin denormalization did not perfectly reverse the normalization."
    )


# --- DynamicRevin Tests ---


def test_dynamic_revin_init():
    """Tests the initialization of the DynamicRevin layer."""
    layer = DynamicRevin(num_features=5)
    assert isinstance(layer.gamma, torch.nn.Parameter)
    assert isinstance(layer.beta, torch.nn.Parameter)


def test_dynamic_revin_reversibility(sample_tensor):
    """
    Tests that denormalize(normalize(x)) returns x for DynamicRevin.
    """
    num_features = sample_tensor.shape[-1]
    layer = DynamicRevin(num_features=num_features)

    # 1. Normalize the tensor
    normalized_output = layer(sample_tensor, mode="norm")

    # 2. Denormalize the output
    denormalized_output = layer(normalized_output, mode="denorm")

    # 3. Assert perfect reversibility
    assert normalized_output.shape == sample_tensor.shape
    assert denormalized_output.shape == sample_tensor.shape
    assert torch.allclose(denormalized_output, sample_tensor, atol=1e-6), (
        "DynamicRevin denormalization did not perfectly reverse the normalization."
    )


def test_dynamic_revin_non_affine(sample_tensor):
    """
    Tests DynamicRevin with affine=False to ensure it still normalizes
    and is reversible.
    """
    num_features = sample_tensor.shape[-1]
    layer = DynamicRevin(num_features=num_features, affine_mode="dynamic")

    # Test for reversibility
    normalized = layer(sample_tensor, mode="norm")
    denormalized = layer(normalized, mode="denorm")

    assert torch.allclose(denormalized, sample_tensor, atol=1e-6)
