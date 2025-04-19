# tests/test_models.py

import pytest
import torch
from temporal.models.basemodel import BaseModel


class MockModel(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def test_basemodel_forward():
    model = MockModel(input_dim=10, output_dim=5)
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 5)
