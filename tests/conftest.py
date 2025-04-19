# tests/conftest.py

import pytest

# You can add pytest configuration options here if needed.


@pytest.fixture
def sample_transformer_config():
    from temporal.configs.transformer_config import TransformerConfig
    return TransformerConfig()
