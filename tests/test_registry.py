# tests/test_registry.py

import pytest
import torch.nn as nn
from temporal.registry.core import (
    register_module,
    resolve,
    list_registered,
    MODULE_REGISTRY,
)
from temporal.registry.generate import (
    register_generate,
    resolve_generate,
    list_generate_registered,
    GENERATE_REGISTRY,
)

# A dummy class for testing
class DummyModule(nn.Module):
    pass

# A dummy class for the generate registry
class DummyGenerator:
    pass

# --- Setup and Teardown ---

@pytest.fixture(autouse=True)
def cleanup_registries():
    """Fixture to clean up registries before and after each test."""
    # Store original state
    original_core_registry = {k: v.copy() for k, v in MODULE_REGISTRY.items()}
    original_gen_registry = GENERATE_REGISTRY.copy()
    
    yield # Run the test
    
    # Restore original state
    MODULE_REGISTRY.clear()
    MODULE_REGISTRY.update(original_core_registry)
    GENERATE_REGISTRY.clear()
    GENERATE_REGISTRY.update(original_gen_registry)


# --- Core Registry Tests ---

def test_register_and_resolve_module():
    """Tests basic registration and resolution of a module."""
    # Register the dummy module
    register_module("attention", "dummy_attn")(DummyModule)
    
    # Resolve it
    resolved_class = resolve("attention", "dummy_attn")
    
    assert resolved_class == DummyModule

def test_register_decorator():
    """Tests the decorator syntax for registration."""
    @register_module("loss", "dummy_loss")
    class DummyLoss(nn.Module):
        pass
        
    resolved_class = resolve("loss", "dummy_loss")
    assert resolved_class == DummyLoss

def test_resolve_unregistered_module_raises_keyerror():
    """Tests that resolving an unregistered module raises a KeyError."""
    with pytest.raises(KeyError, match="Module not found for kind='attention', name='non_existent'"):
        resolve("attention", "non_existent")

def test_resolve_unknown_kind_raises_valueerror():
    """Tests that resolving from an unknown kind raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown module kind: 'invalid_kind'"):
        resolve("invalid_kind", "any_name")

def test_list_registered_modules():
    """Tests listing registered modules for a given kind."""
    register_module("feedforward", "ffn1")(DummyModule)
    register_module("feedforward", "ffn2")(DummyModule)
    
    registered_list = list_registered("feedforward")
    assert isinstance(registered_list, list)
    assert "ffn1" in registered_list
    assert "ffn2" in registered_list
    assert len(registered_list) >= 2 # >= to account for modules registered at import time

def test_list_registered_for_empty_kind():
    """Tests listing for a valid kind with no registered modules."""
    # We assume 'head_agg' might be empty in some test setups
    assert list_registered("head_agg") == []

# --- Generate Registry Tests ---

def test_register_and_resolve_generate():
    """Tests basic registration and resolution for the generate registry."""
    register_generate("dummy_gen")(DummyGenerator)
    resolved_class = resolve_generate("dummy_gen")
    assert resolved_class == DummyGenerator

def test_resolve_unregistered_generate_raises_keyerror():
    """Tests that resolving an unregistered generation method raises a KeyError."""
    with pytest.raises(KeyError, match="Generation method not found: 'non_existent'"):
        resolve_generate("non_existent")

def test_list_generate_registered():
    """Tests listing registered generation methods."""
    register_generate("gen1")(DummyGenerator)
    register_generate("gen2")(DummyGenerator)
    
    registered_list = list_generate_registered()
    assert "gen1" in registered_list
    assert "gen2" in registered_list
