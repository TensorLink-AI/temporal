import os
import json
import torch
import torch.nn as nn
import unittest
from unittest.mock import patch, MagicMock
import pytest
from temporal.utils.hf_accessors import save_hf, load_hf

class MockModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(10, 10)

    # FIX: Ensure state_dict returns correctly named keys for the nn.Module
    def state_dict(self, *args, **kwargs):
        return {"linear.weight": self.linear.weight, "linear.bias": self.linear.bias}

class MockConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type", "mock")

    def to_dict(self):
        return {"model_type": "mock"}

class TestHfAccessors(unittest.TestCase):

    def setUp(self):
        self.model = MockModel(MockConfig())
        self.config = MockConfig()
        self.save_directory = "test_save"

    def tearDown(self):
        if os.path.exists(self.save_directory):
            for f in os.listdir(self.save_directory):
                os.remove(os.path.join(self.save_directory, f))
            os.rmdir(self.save_directory)

    @patch("temporal.utils.hf_accessors._HAS_SAFETENSORS", True)
    @patch("safetensors.torch.save_file")
    def test_save_hf_safe(self, mock_save):
        save_hf(self.model, self.config, self.save_directory, safe=True)
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "config.json")))
        mock_save.assert_called_once() # Will now be called with a non-empty dict

    def test_save_hf_unsafe(self):
        save_hf(self.model, self.config, self.save_directory, safe=False)
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "pytorch_model.bin")))

    @patch("temporal.utils.hf_accessors._HAS_HUGGINGFACE_HUB", True)
    @patch("huggingface_hub.upload_folder")
    @patch("huggingface_hub.create_repo")
    @patch("huggingface_hub.HfApi.repo_info")
    def test_save_hf_push_to_hub(self, mock_repo_info, mock_create_repo, mock_upload_folder):
        mock_repo_info.side_effect = Exception("Repo not found")
        with pytest.raises(Exception, match="Repo not found"):
            save_hf(self.model, self.config, self.save_directory, push_to_hub=True, repo_id="test/repo")

    @patch("temporal.utils.hf_accessors._HAS_SAFETENSORS", True)
    @patch("safetensors.torch.load_file")
    def test_load_hf_safe(self, mock_load):
        # The state_dict from the model now has the correct keys
        mock_load.return_value = self.model.state_dict()
        # Temporarily create the config file needed by the load function
        os.makedirs(self.save_directory, exist_ok=True)
        with open(os.path.join(self.save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
            
        loaded_model = load_hf(self.save_directory, MockModel, MockConfig, safe=True)
        self.assertIsInstance(loaded_model, MockModel)

    def test_load_hf_unsafe(self):
        save_hf(self.model, self.config, self.save_directory, safe=False)
        loaded_model = load_hf(self.save_directory, MockModel, MockConfig, safe=False)
        self.assertIsInstance(loaded_model, MockModel)

if __name__ == '__main__':
    unittest.main()
