# tests/test_heads.py

import pytest
import torch
from temporal.modules.heads.output_heads import (
    LinearOutputHead,
    GaussianHead,
    QuantileRegressionOutputHead,
    DistPredHead,
    MixtureOutputHead,
)

# --- Fixtures ---

@pytest.fixture
def head_params():
    """Provides common parameters for output heads."""
    return {
        "hidden_size": 32,
        "batch_size": 2,
        "seq_len": 10,
    }

@pytest.fixture
def sample_hidden_state(head_params):
    """Provides a sample hidden state tensor."""
    return torch.randn(
        head_params["batch_size"],
        head_params["seq_len"],
        head_params["hidden_size"]
    )

# --- LinearOutputHead Tests ---

def test_linear_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of the LinearOutputHead."""
    output_size = 3
    head = LinearOutputHead(hidden_size=head_params["hidden_size"], output_size=output_size)
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        output_size
    )
    assert head.get_loss_fn() is None

# --- GaussianHead Tests ---

def test_gaussian_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of the GaussianHead."""
    feature_size = 3
    head = GaussianHead(hidden_size=head_params["hidden_size"], output_size=feature_size)
    output = head(sample_hidden_state)
    # mu and log_sigma for each feature
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size * 2
    )
    assert head.get_loss_fn() is None

# --- QuantileRegressionOutputHead Tests ---

def test_quantile_head_init_validation():
    """Tests that QuantileRegressionOutputHead validates its init args."""
    with pytest.raises(ValueError, match="Output size mismatch"):
        QuantileRegressionOutputHead(
            hidden_size=32,
            output_size=10, # Mismatch
            num_quantiles=3,
            feature_size=3
        )

def test_quantile_head_forward_multivariate(head_params, sample_hidden_state):
    """Tests the forward pass of QuantileRegressionOutputHead for a multivariate case."""
    feature_size = 3
    num_quantiles = 5
    head = QuantileRegressionOutputHead(
        hidden_size=head_params["hidden_size"],
        output_size=feature_size * num_quantiles,
        num_quantiles=num_quantiles,
        feature_size=feature_size
    )
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        num_quantiles
    )

# --- DistPredHead Tests ---

def test_distpred_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of the DistPredHead."""
    num_outputs = 7
    feature_size = 1
    head = DistPredHead(
        hidden_size=head_params["hidden_size"],
        output_size=num_outputs * feature_size,
        num_outputs=num_outputs,
        feature_size=feature_size
    )
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        num_outputs
    )

def test_distpred_head_predict_method(head_params):
    """Tests the `predict` method of DistPredHead for autoregression."""
    num_outputs = 7
    feature_size = 3
    head = DistPredHead(
        hidden_size=head_params["hidden_size"],
        output_size=num_outputs * feature_size,
        num_outputs=num_outputs,
        feature_size=feature_size
    )
    # Mock forward output
    forward_output = torch.randn(
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        num_outputs
    )
    # The `predict` method should return the median point forecast
    prediction = head.predict(forward_output)
    assert prediction.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size
    )

# --- MixtureOutputHead Tests ---

def test_mixture_head_init(head_params):
    """Tests the initialization of the MixtureOutputHead."""
    components = ["normal", "student_t"]
    head = MixtureOutputHead(
        hidden_size=head_params["hidden_size"],
        components=components
    )
    # Expected params: normal(mu, sigma) + student_t(df, mu, scale) + mixture_logits
    expected_dim = 2 + 3 + len(components)
    assert head.output_projection.out_features == expected_dim

def test_mixture_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of MixtureOutputHead."""
    components = ["normal", "student_t", "fixed_normal"]
    head = MixtureOutputHead(
        hidden_size=head_params["hidden_size"],
        components=components
    )
    output_dict = head(sample_hidden_state)
    
    assert isinstance(output_dict, dict)
    assert "mixture_logits" in output_dict
    assert "normal_mu" in output_dict
    assert "student_df" in output_dict
    assert output_dict["mixture_logits"].shape[-1] == len(components)
    assert output_dict["normal_mu"].shape == (
        head_params["batch_size"],
        head_params["seq_len"]
    )
