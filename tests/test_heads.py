# tests/test_heads.py

import pytest
import torch
from temporal.modules.heads.output_heads import LinearOutputHead  # Replace with a real head module


def test_LinearOutputHead():
    head = LinearOutputHead(dim=10, output_dim=5)  # Replace with the correct parameters
    x = torch.randn(1, 5, 10)
    output = head(x)
    assert output.shape == (1, 5, 5)