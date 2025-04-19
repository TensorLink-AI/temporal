# tests/test_decoders.py

import pytest
import torch
from temporal.modules.decoders.decoders import StandardDecoder  # Replace with a real decoder module


def test_StandardDecoder():
    decoder = StandardDecoder(dim=10, depth=2)  # Replace with the correct parameters
    x = torch.randn(1, 5, 10)
    memory = torch.randn(1, 5, 10)
    output = decoder(x, memory)
    assert output.shape == (1, 5, 10)