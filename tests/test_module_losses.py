# tests/test_module_losses.py

import pytest
import torch
from temporal.modules.losses.losses import MSELoss  # Replace with a real loss module


def test_MSELoss():
    loss_fn = MSELoss()
    x = torch.randn(1, 5, 10)
    y = torch.randn(1, 5, 10)
    loss = loss_fn(x, y)
    assert loss.shape == ()
