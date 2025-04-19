# tests/test_norm.py

import pytest
import torch
from temporal.modules.norm.layer_norm import LayerNorm  # Replace with a real norm module


def test_LayerNorm():
    norm = LayerNorm(dim=10)  # Replace with the correct parameters
    x = torch.randn(1, 5, 10)
    output = norm(x)
    assert output.shape == (1, 5, 10)