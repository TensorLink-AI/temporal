# tests/test_hf_compat.py

import pytest
import torch
import os
from temporal.utils.hf_accessors import save_hf, load_hf
from temporal.utils.hf_adapter import TimeSeriesTransformerModel as HfAdaptedModel
from temporal.configs.transformer_config import TransformerTimeSeriesConfig

# --- Fixtures ---

@pytest.fixture(scope="module")
def hf_config():
    """Provides a simple config for testing HF compatibility."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        num_heads=2,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        loss_config={"type": "mse"}
    )

@pytest.fixture(scope="module")
def adapted_model(hf_config):
    """Provides an instance of the HF-compatible adapted model."""
    return HfAdaptedModel(hf_config)


# --- Test Cases ---

@pytest.mark.parametrize("safe_serialization", [True, False])
def test_save_and_load_hf_local(adapted_model, hf_config, tmp_path, safe_serialization):
    """
    Tests saving and loading the model locally using both .bin and .safetensors.
    """
    save_directory = tmp_path / "test_model"
    
    # 1. Save the model
    save_hf(
        model=adapted_model,
        config=hf_config,
        save_directory=str(save_directory),
        safe=safe_serialization
    )
    
    # 2. Check if files were created
    expected_weight_file = "model.safetensors" if safe_serialization else "pytorch_model.bin"
    assert os.path.exists(save_directory / "config.json")
    assert os.path.exists(save_directory / expected_weight_file)
    
    # 3. Load the model back
    loaded_model = load_hf(
        model_name_or_path=str(save_directory),
        model_cls=HfAdaptedModel,
        config_cls=TransformerTimeSeriesConfig,
        safe=safe_serialization
    )
    
    # 4. Verify the loaded model
    assert isinstance(loaded_model, HfAdaptedModel)
    assert loaded_model.config.d_model == hf_config.d_model
    
    # Compare state dicts to ensure weights are identical
    original_sd = adapted_model.state_dict()
    loaded_sd = loaded_model.state_dict()
    assert original_sd.keys() == loaded_sd.keys()
    for key in original_sd:
        assert torch.allclose(original_sd[key], loaded_sd[key])

def test_load_hf_missing_files(tmp_path):
    """Tests that load_hf raises an error if config or weight files are missing."""
    # Test with missing config
    with pytest.raises(FileNotFoundError, match="Config file 'config.json' not found"):
        load_hf(str(tmp_path), HfAdaptedModel, TransformerTimeSeriesConfig)
        
    # Test with missing weight file
    # Create a dummy config file to proceed past the first check
    (tmp_path / "config.json").touch()
    with pytest.raises(FileNotFoundError, match="Could not find weight file"):
        load_hf(str(tmp_path), HfAdaptedModel, TransformerTimeSeriesConfig)

def test_hf_adapter_init(adapted_model, hf_config):
    """Tests that the Hugging Face adapter model initializes correctly."""
    assert adapted_model.config == hf_config
    assert hasattr(adapted_model, "temporal") # The core model should be an attribute
    assert isinstance(adapted_model.temporal, nn.Module)

def test_hf_adapter_forward_delegation(adapted_model):
    """Tests that the adapter's forward pass delegates correctly."""
    # Mock the underlying model's forward method to track calls
    from unittest.mock import MagicMock
    adapted_model.temporal.forward = MagicMock()
    
    inputs = torch.randn(2, 10, adapted_model.config.feature_size)
    adapted_model.forward(input_values=inputs, attention_mask=None)
    
    # Check that the underlying model's forward method was called once
    adapted_model.temporal.forward.assert_called_once()
