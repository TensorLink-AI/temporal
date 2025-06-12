# tests/test_module_losses.py

import pytest
import torch
from temporal.modules.losses.loss_functions import (
    MQLoss,
    WeightedQuantileLoss,
    QuantileLoss,
    KernelEnergyLoss,
    SpectralLoss,
    SpreadPenalty,
    MixtureLoss,
)

# --- Fixtures ---

@pytest.fixture
def sample_tensors():
    """Provides simple tensors for testing standard losses."""
    preds = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
    targets = torch.tensor([[2.5, 3.5], [5.5, 5.5]])
    return preds, targets

@pytest.fixture
def quantile_tensors():
    """Provides tensors for quantile-based losses."""
    preds = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]
    ]).float()
    targets = torch.tensor([[2.5, 4.5], [10.5, 14.5]]).float()
    return preds, targets

# --- Test Cases ---

def test_quantile_loss(sample_tensors):
    """Tests the basic QuantileLoss (pinball loss)."""
    preds, targets = sample_tensors
    # Test q=0.5 (should be 0.5 * MAE)
    loss_fn_median = QuantileLoss(quantile=0.5)
    mae = torch.mean(torch.abs(preds - targets))
    assert torch.isclose(loss_fn_median(preds, targets), 0.5 * mae)

    # Test q=0.2
    loss_fn_low = QuantileLoss(quantile=0.2)
    assert loss_fn_low(preds, targets) > 0

def test_mq_loss(quantile_tensors):
    """Tests the Multi-Quantile Loss."""
    preds, targets = quantile_tensors
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = MQLoss(quantiles=quantiles)
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    assert loss > 0

def test_weighted_quantile_loss(quantile_tensors):
    """Tests the WeightedQuantileLoss."""
    preds, targets = quantile_tensors
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = WeightedQuantileLoss(quantiles=quantiles)
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    assert loss > 0

def test_kernel_energy_loss(quantile_tensors):
    """Tests the KernelEnergyLoss."""
    preds, targets = quantile_tensors # Use quantile preds as samples
    loss_fn = KernelEnergyLoss()
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

def test_spectral_loss(sample_tensors):
    """Tests the SpectralLoss."""
    preds, targets = sample_tensors
    loss_fn = SpectralLoss()
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    assert loss > 0

def test_spread_penalty():
    """Tests the SpreadPenalty module."""
    # Sorted predictions [B, T, Q]
    preds = torch.tensor([[[1.0, 2.0, 8.0]]]).float() # Large spread
    preds_narrow = torch.tensor([[[1.0, 1.1, 1.2]]]).float() # Narrow spread
    
    penalty_fn = SpreadPenalty(penalty_type='log')
    
    penalty_high_spread = penalty_fn(preds)
    penalty_low_spread = penalty_fn(preds_narrow)
    
    # log penalty should be lower for higher spread
    assert penalty_high_spread < penalty_low_spread

    # Test inverse penalty
    penalty_fn_inv = SpreadPenalty(penalty_type='inverse')
    assert penalty_fn_inv(preds) < penalty_fn_inv(preds_narrow)

def test_mixture_loss():
    """Tests the MixtureLoss function."""
    loss_fn = MixtureLoss()
    batch_size, seq_len = 2, 10
    
    # Mock prediction from a head with two components: normal and student_t
    preds_dict = {
        "components": ["normal", "student_t"],
        "mixture_logits": torch.randn(batch_size, seq_len, 2),
        "normal_mu": torch.randn(batch_size, seq_len),
        "normal_sigma": torch.rand(batch_size, seq_len),
        "student_df": torch.rand(batch_size, seq_len) * 5,
        "student_mu": torch.randn(batch_size, seq_len),
        "student_scale": torch.rand(batch_size, seq_len),
    }
    targets = torch.randn(batch_size, seq_len)
    
    loss = loss_fn(preds_dict, targets)
    
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
