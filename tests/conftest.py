# tests/conftest.py

import sys
import os
import pytest

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_transformer_config():
    from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
    return TransformerTimeSeriesConfig()
