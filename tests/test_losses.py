import pytest
import torch
from temporal.modules.losses.losses import (
    NegativeLogLikelihoodLoss,
    TimeSeriesLoss,
    CRPSLoss,
    SpreadPenalty,
)

# --- Fixtures ---


@pytest.fixture
def sample_tensors():
    """Provides simple, predictable tensors for loss calculation."""
    preds = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]).unsqueeze(
        -1
    )  # Shape [2, 3, 1]
    targets = torch.tensor([[2.5, 3.5, 3.5], [5.5, 5.5, 7.5]])  # Shape [2, 3]
    return preds, targets


@pytest.fixture
def quantile_tensors():
    """Provides tensors for testing quantile-based losses."""
    # Shape [Batch, Time, Quantiles]
    preds = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[10, 11, 12], [13, 14, 15]]], dtype=torch.float32
    )
    targets = torch.tensor([[2.5, 4.5], [10.5, 14.5]], dtype=torch.float32)
    return preds, targets


@pytest.fixture
def loss_mask():
    """Provides a sample loss mask."""
    # Mask out the last time step for the first sample in the batch
    return torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])


# --- TimeSeriesLoss Tests ---


@pytest.mark.parametrize(
    "loss_type, expected_loss",
    [
        ("mse", 0.25),  # Avg of (0.5^2, 0.5^2, 0.5^2, 0.5^2, 0.5^2, 0.5^2) = 0.25
        ("mae", 0.5),  # Avg of (0.5, 0.5, 0.5, 0.5, 0.5, 0.5) = 0.5
    ],
)
def test_timeseries_loss_mse_mae(sample_tensors, loss_type, expected_loss):
    """Tests TimeSeriesLoss with MSE and MAE."""
    preds, targets = sample_tensors
    loss_fn = TimeSeriesLoss(loss_type=loss_type)
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_timeseries_loss_rmse(sample_tensors):
    """Tests TimeSeriesLoss with RMSE."""
    preds, targets = sample_tensors
    loss_fn = TimeSeriesLoss(loss_type="rmse")
    loss = loss_fn(preds, targets)
    # sqrt(0.25) = 0.5
    assert torch.isclose(loss, torch.tensor(0.5))


def test_timeseries_mq_loss(quantile_tensors):
    """Tests TimeSeriesLoss with Multi-Quantile loss."""
    preds, targets = quantile_tensors
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = TimeSeriesLoss(loss_type="mq", quantiles=quantiles)
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    assert loss > 0


def test_timeseries_loss_with_mask(sample_tensors, loss_mask):
    """Tests that the loss mask is correctly applied."""
    preds, targets = sample_tensors
    loss_fn = TimeSeriesLoss(loss_type="mse")  # Using MSE for simplicity

    # Calculate unmasked loss first
    unmasked_loss = loss_fn(preds, targets)

    # Calculate masked loss
    masked_loss = loss_fn(preds, targets, loss_mask=loss_mask)

    # Manually calculate expected masked loss
    # Unmasked errors squared: [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]
    # Masked sum: 0.25+0.25+0.25+0.25+0.25 = 1.25. Num elements = 5.
    # Expected mean: 1.25 / 5 = 0.25. Okay, mask doesn't change this specific case.
    # Let's change a value to make it different.
    targets_mod = targets.clone()
    targets_mod[0, 2] = 100  # This value will be masked out
    masked_loss_mod = loss_fn(preds, targets_mod, loss_mask=loss_mask)

    unmasked_loss_mod = loss_fn(preds, targets_mod)

    assert masked_loss_mod < unmasked_loss_mod


# --- SpreadPenalty Tests ---
def test_spread_penalty_calculation():
    """
    Tests the SpreadPenalty loss calculation in isolation.
    """
    loss_fn = SpreadPenalty()
    # Predictions shape: [Batch, Time, Quantiles]
    preds = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[10, 11, 12], [13, 14, 15]]], dtype=torch.float32
    )

    # FIX: The penalty calculation has changed to -log(spread).
    # The spread for all 4 data points is 2. So the expected loss is -log(2).
    penalty = loss_fn(preds)  # Targets are not used
    expected_penalty = -torch.log(torch.tensor(2.0))
    assert torch.isclose(penalty, expected_penalty)

    # Test edge case with zero spread
    preds_zero_spread = torch.tensor([[[2, 2, 2], [5, 5, 5]]], dtype=torch.float32)
    penalty_zero = loss_fn(preds_zero_spread)
    # The new loss will be a large positive number due to log(0), capped by epsilon.
    assert penalty_zero > 10 # Check that it's a large penalty


# --- CRPSLoss Tests ---


def test_crps_loss_forward(quantile_tensors):
    """Tests the forward pass of CRPSLoss."""
    preds, targets = quantile_tensors
    loss_fn = CRPSLoss()
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    assert loss > 0


def test_crps_with_spread_penalty(quantile_tensors):
    """Tests that the spread penalty increases the CRPS loss."""
    preds, targets = quantile_tensors

    # Loss without penalty
    loss_fn_no_penalty = CRPSLoss(spread_lambda=0.0)
    loss_no_penalty = loss_fn_no_penalty(preds, targets)

    # Loss with penalty
    loss_fn_with_penalty = CRPSLoss(spread_lambda=0.1)
    loss_with_penalty = loss_fn_with_penalty(preds, targets)

    assert loss_with_penalty > loss_no_penalty


# --- NegativeLogLikelihoodLoss Tests ---


def test_mixture_nll_loss():
    """Tests the NegativeLogLikelihoodLoss wrapper for mixtures."""
    loss_fn = NegativeLogLikelihoodLoss(distribution_type="mixture")
    batch_size = 4
    seq_len = 10
    # FIX: Changed component name from "normal" to "gaussian" and updated keys.
    preds_dict = {
        "components": ["gaussian"],
        "mixture_logits": torch.ones(batch_size, seq_len, 1),
        "gaussian_mu": torch.randn(batch_size, seq_len),
        "gaussian_sigma": torch.ones(batch_size, seq_len),
    }
    targets = torch.randn(batch_size, seq_len)

    loss = loss_fn(preds_dict, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)