# tests/test_embedders.py

import pytest
import torch
from temporal.modules.embedders.embedding import Embedding  # Replace with a real embedder module


def test_Embedding():
    embed = Embedding(num_embeddings=10, embedding_dim=5)  # Replace with the correct parameters
    x = torch.randint(0, 10, (1, 5))
    output = embed(x)
    assert output.shape == (1, 5, 5)