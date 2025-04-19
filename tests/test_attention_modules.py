# tests/test_attention_modules.py

import pytest
import torch
from temporal.modules.attentions.attention_head_agg import AttentionHeadAgg # replace with a real attention module


def test_AttentionHeadAgg():
    attention = AttentionHeadAgg(dim = 10, num_heads = 2) # replace with the correct parameters
    x = torch.randn(1, 5, 10)
    output = attention(x)
    assert output.shape == (1, 5, 10)
