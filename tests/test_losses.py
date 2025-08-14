# tests/test_losses.py

import pytest
import torch
from temporal.modules.losses.losses import TimeSeriesLoss, CRPSLoss, RegisteredMixtureLoss, SpreadPenalty

# --- Fixtures ---

@pytest.fixture
def sample_tensors():
    """Provides simple, predictable tensors for loss calculation."""
    preds = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]).unsqueeze(-1) # Shape [2, 3, 1]
    targets = torch.tensor([[2.5, 3.5, 3.5], [5.5, 5.5, 7.5]]) # Shape [2, 3]
    return preds, targets

@pytest.fixture
def quantile_tensors():
    """Provides tensors for testing quantile-based losses."""
    # Shape [Batch, Time, Quantiles]
    preds = torch.tensor([
        [[1, 2, 3], [4, 5, 6]], # Sample 1
        [[10, 11, 12], [13, 14, 15]]  # Sample 2
    ]).float()
    targets = torch.tensor([[2.5, 4.5], [10.5, 14.5]]).float()
    return preds, targets

@pytest.fixture
def loss_mask():
    """Provides a sample loss mask."""
    # Mask out the last time step for the first sample in the batch
    return torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

# --- TimeSeriesLoss Tests ---

@pytest.mark.parametrize("loss_type, expected_loss", [
    ("mse", 0.25),  # Avg of (0.5^2, 0.5^2, 0.5^2, 0.5^2, 0.5^2, 0.5^2) = 0.25
    ("mae", 0.5),   # Avg of (0.5, 0.5, 0.5, 0.5, 0.5, 0.5) = 0.5
])
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
    loss_fn = TimeSeriesLoss(loss_type="mse") # Using MSE for simplicity
    
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
    targets_mod[0, 2] = 100 # This value will be masked out
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
    # Spread for first sample, first timestep: 3 - 1 = 2
    # Spread for first sample, second timestep: 6 - 4 = 2
    # Spread for second sample, first timestep: 12 - 10 = 2
    # Spread for second sample, second timestep: 15 - 13 = 2
    preds = torch.tensor([
        [[1, 2, 3], [4, 5, 6]],
        [[10, 11, 12], [13, 14, 15]]
    ]).float()
    
    # The penalty is the mean of the squared spreads. 
    # (2^2 + 2^2 + 2^2 + 2^2) / 4 = 4.0
    penalty = loss_fn(preds, None) # Targets are not used
    assert torch.isclose(penalty, torch.tensor(4.0))

    # Test edge case with zero spread
    preds_zero_spread = torch.tensor([[[2, 2, 2], [5, 5, 5]]]).float()
    penalty_zero = loss_fn(preds_zero_spread, None)
    assert torch.isclose(penalty_zero, torch.tensor(0.0))


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

# --- RegisteredMixtureLoss Tests ---

def test_mixture_loss_wrapper(head_params):
    """Tests that the RegisteredMixtureLoss wrapper correctly computes loss."""
    loss_fn = RegisteredMixtureLoss()
    
    # Create a mock prediction dictionary from a MixtureOutputHead
    preds_dict = {
        "components": ["normal"],
        "mixture_logits": torch.ones(head_params["batch_size"], head_params["seq_len"], 1),
        "normal_mu": torch.randn(head_params["batch_size"], head_params["seq_len"]),
        "normal_sigma": torch.ones(head_params["batch_size"], head_params["seq_len"]),
    }
    targets = torch.randn(head_params["batch_size"], head_params["seq_len"])
    
    loss = loss_fn(preds_dict, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
