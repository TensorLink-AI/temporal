import pytest
import torch
from temporal.modules.norm.layer_norm import LayerNorm
from temporal.modules.norm.rms_norm import RMSNorm
from temporal.modules.norm.scale_norm import ScaleNorm

# --- Fixtures ---


@pytest.fixture
def sample_tensor():
    """Provides a sample input tensor with non-zero mean and non-one std."""
    tensor = torch.randn(2, 10, 32) * 5 + 3  # B, T, D
    return tensor


# --- LayerNorm Tests ---


def test_layer_norm_init():
    """Tests the initialization of the custom LayerNorm."""
    norm = LayerNorm(normalized_shape=32)
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert isinstance(norm.bias, torch.nn.Parameter)


def test_layer_norm_forward(sample_tensor):
    """Tests the forward pass of LayerNorm and checks output properties."""
    normalized_shape = sample_tensor.shape[-1]
    norm = LayerNorm(normalized_shape=normalized_shape)

    output = norm(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert not torch.allclose(output, sample_tensor)
    assert torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(output.std(), torch.tensor(1.0), atol=1e-2)


# --- RMSNorm Tests ---


def test_rms_norm_init():
    """Tests the initialization of RMSNorm."""
    norm = RMSNorm(normalized_shape=32)
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert not hasattr(norm, "bias")


def test_rms_norm_forward(sample_tensor):
    """Tests the forward pass of RMSNorm."""
    dim = sample_tensor.shape[-1]
    norm = RMSNorm(normalized_shape=dim)

    output = norm(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert not torch.allclose(output, sample_tensor)
    rms = torch.sqrt(torch.mean(output**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


# --- ScaleNorm Tests ---


def test_scale_norm_init():
    """Tests the initialization of ScaleNorm."""
    norm = ScaleNorm(d_model=32)
    assert isinstance(norm.g, torch.nn.Parameter)
    assert not hasattr(norm, "bias")


def test_scale_norm_forward(sample_tensor):
    """Tests the forward pass of ScaleNorm."""
    d_model = sample_tensor.shape[-1]
    norm = ScaleNorm(d_model=d_model)

    output = norm(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert not torch.allclose(output, sample_tensor)
    l2_norm = torch.norm(output, p=2, dim=-1)
    assert torch.allclose(
        l2_norm.mean(), torch.tensor(norm.g.item()), atol=1e-2
    )