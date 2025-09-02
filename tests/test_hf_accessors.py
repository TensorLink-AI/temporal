import unittest
import os
import shutil
import json
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from temporal.utils.hf_accessors import save_hf, load_hf
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig

class MockModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.linear = nn.Linear(10, 1) # Simplified for clarity

    def state_dict(self, *args, **kwargs):
        # Use the actual state_dict of the underlying module
        return super().state_dict(*args, **kwargs)

class MockConfig(TransformerTimeSeriesConfig):
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "model_type": "mock",
            "architecture": {"type": "transformer_architecture", "layout": "encoder"},
            "d_model": 16,
            "context_length": 4,
            "prediction_length": 2,
        })
        return d


class TestHfAccessors(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.config = MockConfig()
        self.save_directory = "test_save"
        os.makedirs(self.save_directory, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.save_directory):
            shutil.rmtree(self.save_directory)

    def test_save_hf_unsafe(self):
        save_hf(self.model, self.config, self.save_directory, safe=False)
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "pytorch_model.bin")))

    @patch("temporal.utils.hf_accessors._HAS_SAFETENSORS", True)
    @patch("safetensors.torch.save_file")
    def test_save_hf_safe(self, mock_save):
        save_hf(self.model, self.config, self.save_directory, safe=True)
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "config.json")))
        # The safetensors file is created, so assert_called_once should pass
        mock_save.assert_called_once()

    @patch("temporal.utils.hf_accessors._HAS_HUGGINGFACE_HUB", True)
    @patch("huggingface_hub.upload_folder")
    @patch("huggingface_hub.create_repo")
    @patch("huggingface_hub.HfApi.repo_info")
    def test_save_hf_push_to_hub(self, mock_repo_info, mock_create_repo, mock_upload_folder):
        mock_repo_info.side_effect = Exception("Repo not found")
        with self.assertRaises(Exception):
            save_hf(self.model, self.config, self.save_directory, push_to_hub=True, repo_id="test/repo")

    @patch("temporal.utils.hf_accessors._HAS_SAFETENSORS", True)
    @patch("safetensors.torch.load_file")
    def test_load_hf_safe(self, mock_load):
        # Ensure the mock state_dict has the correct keys
        mock_load.return_value = self.model.state_dict()
        # Create dummy files for the load function to find
        os.makedirs(self.save_directory, exist_ok=True)
        with open(os.path.join(self.save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
        with open(os.path.join(self.save_directory, "model.safetensors"), "wb") as f:
            f.write(b"dummy data") # Write some bytes

        loaded_model = load_hf(self.save_directory, MockModel, MockConfig, safe=True)
        self.assertIsInstance(loaded_model, MockModel)

if __name__ == "__main__":
    unittest.main()