# tests/test_registry.py

import pytest
from temporal.registry.core import Registry
from temporal.registry.generate import generate_registry


def test_registry_registration():
    registry = Registry()

    @registry.register("test_component")
    def test_component():
        return "Test Component"

    assert "test_component" in registry
    assert registry["test_component"]() == "Test Component"


def test_generate_registry():
    registry = generate_registry()
    # Add asserts based on what is expected to be registered by generate_registry
    assert "attention" in registry
