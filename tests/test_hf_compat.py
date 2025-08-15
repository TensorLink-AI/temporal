# tests/test_hf_compat.py
import json
import pytest
import torch
import os
import torch.nn as nn
from temporal.utils.hf_accessors import save_hf, load_hf
from temporal.hf_compat.config_wrapper import TimeSeriesTransformerConfig
from temporal.hf_compat.modeling_wrapper import (
    TimeSeriesTransformerModel as HfAdaptedModel,
)
from temporal.models.builder import build_time_series_transformer


# --- Fixtures ---


@pytest.fixture(scope="module")
def hf_config():
    """Provides a simple config for testing HF compatibility."""
    return TimeSeriesTransformerConfig(
        d_model=16,
        num_heads=2,
        feature_size=3,
        prediction_length=5,
        context_length=10,
    )


@pytest.fixture(scope="module")
def adapted_model(hf_config):
    """Provides an instance of the HF-compatible adapted model."""
    torch.manual_seed(0)
    return HfAdaptedModel(hf_config)


# --- Test Cases ---


@pytest.mark.parametrize("safe_serialization", [True, False])
def test_save_and_load_hf_local(
    adapted_model, hf_config, tmp_path, safe_serialization
):
    """
    Tests saving and loading the model locally using both .bin and .safetensors.
    """
    save_directory = tmp_path / "test_model"

    # 1. Save the model
    adapted_model.save_pretrained(
        save_directory=str(save_directory), safe_serialization=safe_serialization
    )

    # 2. Check if files were created
    expected_weight_file = (
        "model.safetensors" if safe_serialization else "pytorch_model.bin"
    )
    assert os.path.exists(save_directory / "config.json")
    assert os.path.exists(save_directory / expected_weight_file)

    # 3. Load the model back
    loaded_model = HfAdaptedModel.from_pretrained(
        pretrained_model_name_or_path=str(save_directory),
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


def test_load_hf_with_missing_config_attributes(tmp_path):
    """
    Tests backward compatibility by loading a config with missing attributes.
    Pydantic should fill in the missing fields with default values.
    """
    save_directory = tmp_path / "test_model_missing_attrs"
    save_directory.mkdir()

    # 1. Create a minimal config, simulating an older version
    minimal_config = {
        "d_model": 16,
        "num_heads": 2,
        "feature_size": 3,
        "prediction_length": 5,
        "context_length": 10,
        "model_type": "time_series_transformer",  # a required field for HF
    }
    config_path = save_directory / "config.json"
    with open(config_path, "w") as f:
        json.dump(minimal_config, f)

    # 2. Create a dummy model state dict to avoid file-not-found errors
    dummy_model = HfAdaptedModel(TimeSeriesTransformerConfig(**minimal_config))
    dummy_model.save_pretrained(save_directory)

    # 3. Load the model using the current, more complete config class
    loaded_model = HfAdaptedModel.from_pretrained(save_directory)

    # 4. Assert that the model loaded and defaults were set
    assert loaded_model is not None
    # Check a field that was missing and should now have a default value
    assert loaded_model.config.architecture["layout"] == "encoder-decoder"


def test_load_hf_missing_files(tmp_path):
    """Tests that from_pretrained raises an error if essential files are missing."""
    with pytest.raises(OSError, match="does not appear to have a file named config.json"):
        HfAdaptedModel.from_pretrained(str(tmp_path))

    # Create a dummy config to proceed past the first check
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump({"model_type": "time_series_transformer"}, f)

    with pytest.raises(OSError, match="does not appear to have a file named"):
        HfAdaptedModel.from_pretrained(str(tmp_path))


def test_hf_adapter_init(adapted_model, hf_config):
    """Tests that the Hugging Face adapter model initializes correctly."""
    assert adapted_model.config.to_dict() == hf_config.to_dict()
    assert hasattr(adapted_model, "temporal")
    assert isinstance(adapted_model.temporal, nn.Module)


def test_hf_adapter_forward_delegation(adapted_model):
    """Tests that the adapter's forward pass delegates correctly to the core model."""
    from unittest.mock import MagicMock

    # Mock the underlying model's forward method
    adapted_model.temporal.forward = MagicMock(return_value={"loss": torch.tensor(0.5)})

    # Prepare dummy inputs compatible with the HF forward signature
    inputs = torch.randn(
        2, adapted_model.config.context_length, adapted_model.config.feature_size
    )

    # Call the adapter's forward method
    outputs = adapted_model.forward(
        past_values=inputs,
        future_values=torch.randn(
            2, adapted_model.config.prediction_length, adapted_model.config.feature_size
        ),
    )

    # Assert that the underlying model's forward was called
    adapted_model.temporal.forward.assert_called_once()
    assert "loss" in outputs
    assert outputs.loss is not None
