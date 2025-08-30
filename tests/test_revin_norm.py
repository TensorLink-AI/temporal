import pytest
import torch
from temporal.modules.norm.revin import RevIN as Revin
from temporal.modules.norm.dynamic_revin import DynamicRevIN as DynamicRevin

# --- Fixtures ---


@pytest.fixture
def sample_tensor():
    """Provides a sample input tensor with non-zero mean and non-one std."""
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
    Tests that denormalize(normalize(x)) returns x.
    """
    num_features = sample_tensor.shape[-1]
    layer = Revin(num_features=num_features)

    normalized_output = layer(sample_tensor, mode="norm")
    denormalized_output = layer(normalized_output, mode="denorm")

    assert normalized_output.shape == sample_tensor.shape
    assert denormalized_output.shape == sample_tensor.shape
    assert torch.allclose(denormalized_output, sample_tensor, atol=1e-6)


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

    normalized_output = layer(sample_tensor, mode="norm")
    denormalized_output = layer(normalized_output, mode="denorm")

    assert normalized_output.shape == sample_tensor.shape
    assert denormalized_output.shape == sample_tensor.shape
    assert torch.allclose(denormalized_output, sample_tensor, atol=1e-4)


def test_dynamic_revin_non_affine(sample_tensor):
    """
    Tests DynamicRevin with affine=False to ensure it still normalizes
    and is reversible.
    """
    num_features = sample_tensor.shape[-1]
    layer = DynamicRevin(num_features=num_features, affine_mode="dynamic")

    normalized = layer(sample_tensor, mode="norm")
    denormalized = layer(normalized, mode="denorm")

    assert torch.allclose(denormalized, sample_tensor, atol=1e-4)