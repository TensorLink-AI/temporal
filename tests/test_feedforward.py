# tests/test_feedforward.py

import pytest
import torch
from temporal.modules.feedforward.standard import StandardFeedForward
from temporal.modules.feedforward.moe import MoEFeedForward

# --- Fixtures ---


@pytest.fixture
def ffn_config():
    """Provides a base configuration for feed-forward networks."""
    return {
        "hidden_size": 32,
        "intermediate_size": 64,
        "dropout": 0.1,
    }


@pytest.fixture
def sample_tensor(ffn_config):
    """Provides a sample input tensor."""
    return torch.randn(2, 10, ffn_config["hidden_size"])


# --- StandardFeedForward Tests ---


def test_standard_feedforward_init(ffn_config):
    """Tests the initialization of the StandardFeedForward module."""
    ffn = StandardFeedForward(**ffn_config)
    assert ffn.fc1.in_features == ffn_config["hidden_size"]
    assert ffn.fc1.out_features == ffn_config["intermediate_size"]
    assert ffn.fc2.in_features == ffn_config["intermediate_size"]
    assert ffn.fc2.out_features == ffn_config["hidden_size"]


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
def test_standard_feedforward_activations(ffn_config, sample_tensor, activation):
    """Tests that different activation functions can be used."""
    config = ffn_config.copy()
    config["activation"] = activation
    ffn = StandardFeedForward(**config)
    output, _ = ffn(sample_tensor)
    assert output.shape == sample_tensor.shape


def test_standard_feedforward_invalid_activation(ffn_config):
    """Tests that an invalid activation function raises a ValueError."""
    config = ffn_config.copy()
    config["activation"] = "invalid_activation"
    with pytest.raises(ValueError):
        StandardFeedForward(**config)


# --- MoEFeedForward Tests ---


@pytest.fixture
def moe_config(ffn_config):
    """Provides a configuration for the MoEFeedForward module."""
    config = ffn_config.copy()
    config.update(
        {
            "num_experts": 4,
            "top_k": 2,
        }
    )
    return config


def test_moe_feedforward_init(moe_config):
    """Tests the initialization of the MoEFeedForward module."""
    moe = MoEFeedForward(**moe_config)
    assert len(moe.experts) == moe_config["num_experts"]
    assert moe.top_k == moe_config["top_k"]
    assert moe.gate.out_features == moe_config["num_experts"]


def test_moe_feedforward_forward_pass_eval_mode(moe_config, sample_tensor):
    """Tests the forward pass of MoEFeedForward in evaluation mode."""
    moe = MoEFeedForward(**moe_config)
    moe.eval()  # Ensure it's in eval mode, so aux_loss should be None

    output, aux_loss = moe(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert aux_loss is None


def test_moe_feedforward_forward_pass_train_mode(moe_config, sample_tensor):
    """Tests the forward pass of MoEFeedForward in training mode."""
    moe = MoEFeedForward(**moe_config)
    moe.train()  # Ensure it's in train mode

    output, aux_loss = moe(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert aux_loss is not None
    assert aux_loss.ndim == 0  # Should be a scalar tensor


def test_moe_feedforward_top_k_validation(ffn_config):
    """Tests that MoEFeedForward raises an error if top_k > num_experts."""
    config = ffn_config.copy()
    config.update(
        {
            "num_experts": 4,
            "top_k": 5,  # Invalid top_k
        }
    )
    with pytest.raises(ValueError, match="top_k .* cannot be greater than num_experts"):
        MoEFeedForward(**config)
