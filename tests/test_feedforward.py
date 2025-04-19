# tests/test_feedforward.py

import pytest
import torch
from temporal.modules.feedforward.standard import StandardFeedForward  # Replace with a real feedforward module


def test_StandardFeedForward():
    ff = StandardFeedForward(dim=10, hidden_dim=20)  # Replace with the correct parameters
    x = torch.randn(1, 5, 10)
    output = ff(x)
    assert output.shape == (1, 5, 10)