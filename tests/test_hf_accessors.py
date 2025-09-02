import os
import json
import unittest
from unittest.mock import patch
import torch
import torch.nn as nn

from temporal.utils.hf_accessors import save_hf, load_hf
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.loss_config import MSELossConfig


class MockModel(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        self.l = nn.Linear(4, 4)

    def forward(self, x):
        return self.l(x)


class MockConfig(TransformerTimeSeriesConfig):
    def to_dict(self):
        d = super().to_dict()
        d.update(
            {
                "model_type": "mock",
                "architecture": {"type": "transformer_architecture", "layout": "encoder"},
                "d_model": 16,
                "context_length": 4,
                "prediction_length": 2,
            }
        )
        return d


class TestHfAccessors(unittest.TestCase):
    def setUp(self):
        self.save_directory = "test_save"
        self.model = MockModel()
        self.config = MockConfig(
            feature_size=1,
            d_model=16,
            context_length=4,
            prediction_length=2,
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder"),
            output_head_config=OutputHeadConfig(type="linear", output_size=1),
            loss_config=MSELossConfig(type="mse"),
        )

    @patch("temporal.utils.hf_accessors._HAS_SAFETENSORS", True)
    @patch("temporal.utils.hf_accessors._safetensors_save")
    def test_save_hf_safe(self, mock_save):
        save_hf(self.model, self.config, self.save_directory, safe=True)
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "config.json")))
        mock_save.assert_called_once()

    @patch("temporal.utils.hf_accessors._HAS_SAFETENSORS", True)
    @patch("temporal.utils.hf_accessors._safetensors_load")
    def test_load_hf_safe(self, mock_load):
        # Return a valid state dict so load can proceed
        mock_load.return_value = self.model.state_dict()
        os.makedirs(self.save_directory, exist_ok=True)
        with open(os.path.join(self.save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
        with open(os.path.join(self.save_directory, "model.safetensors"), "wb") as f:
            f.write(b"x")
        loaded_model = load_hf(self.save_directory, MockModel, MockConfig, safe=True)
        self.assertIsInstance(loaded_model, MockModel)
