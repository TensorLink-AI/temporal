# tests/test_blocks.py

import pytest
import torch
from temporal.modules.blocks.block_multiscale import BlockMultiscale  # Replace with a real block module


def test_BlockMultiscale():
    block = BlockMultiscale(dim=10, num_scales=2)  # Replace with the correct parameters
    x = torch.randn(1, 5, 10)
    output = block(x)
    assert output.shape == (1, 5, 10)