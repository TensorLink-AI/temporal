# tests/test_hf_compat.py

import pytest
from temporal.hf_compat.config_wrapper import ConfigWrapper  # Replace with actual class


def test_config_wrapper():
    # Create a mock temporal config
    class MockTemporalConfig:
        def __init__(self):
            self.some_parameter = 10

    temporal_config = MockTemporalConfig()

    # Wrap it
    config_wrapper = ConfigWrapper(temporal_config)

    # Check if attributes are accessible
    assert config_wrapper.some_parameter == 10
