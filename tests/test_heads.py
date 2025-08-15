# tests/test_heads.py

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
    """Tests the forward pass of the LinearHead."""
    output_size = 5
    head = LinearHead(hidden_size=head_params["hidden_size"], output_size=output_size)
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        output_size,
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
        feature_size,
    )


# --- QuantileRegressionOutputHead Tests ---


def test_quantile_head_init(head_params):
    """Tests the initialization of QuantileRegressionOutputHead."""
    feature_size = 3
    num_quantiles = 5
    head = QuantileRegressionOutputHead(
        hidden_size=head_params["hidden_size"],
        output_size=feature_size * num_quantiles,
        num_quantiles=num_quantiles,
        feature_size=feature_size,
    )
    assert head.num_quantiles == num_quantiles
    assert head.feature_size == feature_size
    assert head.proj.out_features == feature_size * num_quantiles


def test_quantile_head_forward(head_params, sample_hidden_state):
    """Tests the forward pass of QuantileRegressionOutputHead."""
    feature_size = 3
    num_quantiles = 5
    head = QuantileRegressionOutputHead(
        hidden_size=head_params["hidden_size"],
        output_size=feature_size * num_quantiles,
        num_quantiles=num_quantiles,
        feature_size=feature_size,
    )
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        num_quantiles,
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
        feature_size=feature_size,
    )
    output = head(sample_hidden_state)
    assert output.shape == (
        head_params["batch_size"],
        head_params["seq_len"],
        feature_size,
        num_outputs,
    )
