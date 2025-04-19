# tests/test_losses.py

import pytest
import torch
from temporal.losses.crps_loss_ensemble import crps_loss_ensemble
from temporal.losses.loss_functions import test # Replace with actual loss functions


def test_crps_loss_ensemble():
    # Example test case (replace with actual data and expected values)
    y_true = torch.randn(10)
    y_preds = torch.randn(10, 5)  # Ensemble of 5 predictions
    loss = crps_loss_ensemble(y_true, y_preds)
    assert loss.shape == ()
    # Add more assertions to check the loss value
