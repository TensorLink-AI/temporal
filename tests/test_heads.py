import pytest
import torch
from temporal.modules.heads.output_heads import (
    LinearOutputHead,
    GaussianHead,
    QuantileRegressionOutputHead,
    DistPredHead,
)
from temporal.modules.losses.loss_functions import QuantileLoss

# --- Fixtures ---


@pytest.fixture(scope="module")
def head_params():
    """Provides common parameters for head tests."""
    return {"batch_size": 2, "seq_len": 10, "hidden_size": 32}


@pytest.fixture
def sample_hidden_state(head_params):
    """Provides a sample hidden state tensor."""
    return torch.randn(
        head_params["batch_size"], head_params["seq_len"], head_params["hidden_size"]
    )


# --- LinearHead Tests ---


def test_linear_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of the LinearOutputHead."""
    feature_size = 3
    head = LinearOutputHead(hidden_size=head_params["hidden_size"], output_size=feature_size)
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
    )


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
        feature_size * 2,
    )


def test_gaussian_head_predict(head_params, sample_hidden_state):
    """Tests the `predict` method (sampling) of the GaussianHead."""
    feature_size = 3
    head = GaussianHead(hidden_size=head_params["hidden_size"], output_size=feature_size)
    mock_forward_output = torch.randn(
        head_params["batch_size"], head_params["seq_len"], feature_size * 2
    )
    head.proj = torch.nn.Identity()
    
    prediction = head.predict(mock_forward_output)
    assert prediction.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
    )


def test_gaussian_head_sample_quantiles(head_params, sample_hidden_state):
    """Tests the `sample_quantiles` method of the GaussianHead."""
    feature_size = 3
    quantile_levels = [0.1, 0.5, 0.9]
    head = GaussianHead(hidden_size=head_params["hidden_size"], output_size=feature_size)
    mock_forward_output = torch.randn(
        head_params["batch_size"], head_params["seq_len"], feature_size * 2
    )
    head.proj = torch.nn.Identity()

    quantiles = head.sample_quantiles(mock_forward_output, quantile_levels)
    expected_shape = (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        len(quantile_levels),
    )
    assert quantiles.shape == expected_shape


# --- QuantileRegressionOutputHead Tests ---


def test_quantile_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of QuantileRegressionOutputHead."""
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
        num_quantiles,
    )


def test_quantile_head_predict_method(head_params, sample_hidden_state):
    """Tests the `predict` method of QuantileRegressionOutputHead."""
    feature_size = 3
    num_quantiles = 5
    head = QuantileRegressionOutputHead(
        hidden_size=head_params["hidden_size"],
        output_size=feature_size * num_quantiles,
        num_quantiles=num_quantiles,
        feature_size=feature_size
    )
    forward_output = torch.randn(
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        num_quantiles
    ).sort(dim=-1).values

    prediction = head.predict(forward_output)
    assert prediction.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
    )


def test_quantile_head_get_loss(head_params):
    """Tests that the correct loss function is returned."""
    head = QuantileRegressionOutputHead(
        hidden_size=head_params["hidden_size"],
        output_size=15,
        num_quantiles=5,
        feature_size=3
    )
    loss_fn = head.get_loss_fn()
    assert loss_fn is None


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
    assert output["paths"].shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        num_outputs,
    )


def test_distpred_head_predict(head_params, sample_hidden_state):
    """Tests the `predict` method of DistPredHead."""
    num_outputs = 7
    feature_size = 1
    head = DistPredHead(
        hidden_size=head_params["hidden_size"],
        output_size=num_outputs * feature_size,
        num_outputs=num_outputs,
        feature_size=feature_size,
    )
    mock_forward_output = head(sample_hidden_state)

    prediction_mean = head.predict(mock_forward_output, method="mean")
    assert prediction_mean.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
    )

    prediction_median = head.predict(mock_forward_output, method="median")
    assert prediction_median.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
    )

    prediction_q = head.predict(mock_forward_output, method=0.1)
    assert prediction_q.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
    )


def test_distpred_head_sample_quantiles(head_params, sample_hidden_state):
    """Tests the `sample_quantiles` method of DistPredHead."""
    num_outputs = 7
    feature_size = 1
    quantile_levels = [0.1, 0.5, 0.9]
    head = DistPredHead(
        hidden_size=head_params["hidden_size"],
        output_size=num_outputs * feature_size,
        num_outputs=num_outputs,
        feature_size=feature_size,
    )
    mock_forward_output = head(sample_hidden_state)

    quantiles = head.sample_quantiles(mock_forward_output, quantile_levels)
    expected_shape = (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        len(quantile_levels),
    )
    assert quantiles.shape == expected_shape