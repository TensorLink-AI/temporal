import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from temporal.models.mixin.autoregressive_patch import AutoregressivePatchMixin

class MockModel(nn.Module, AutoregressivePatchMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessor = MagicMock()
        self.encoder = None
        self.decoder = MagicMock()
        self.output_patch_reconstructor = MagicMock()
        self.output_heads = MagicMock()

class TestAutoregressivePatchMixin(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.feature_size = 1
        self.config.d_model = 16
        self.config.prediction_length = 5
        self.model = MockModel(self.config)
        # Mock preprocessor to have patch_size
        self.model.preprocessor.patch_size = 2

    def test_generate(self):
        decoder_inputs = torch.randn(2, 10, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 5, 16),
        }
        # FIX: Return an object with attributes, not a dict
        self.model.decoder.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 1, 16),
            past_key_values=None
        )
        self.model.output_patch_reconstructor.return_value = torch.randn(2, 1, 16)
        self.model.output_heads.return_value = torch.randn(2, 5, 1)
        self.model.output_heads.predict.return_value = torch.randn(2, 5, 1)


        predictions = self.model.generate(
            decoder_inputs=decoder_inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

    def test_forecast(self):
        inputs = torch.randn(2, 10, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 5, 16),
        }
        # FIX: Return an object with attributes, not a dict
        self.model.decoder.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 1, 16),
            past_key_values=None
        )
        self.model.output_patch_reconstructor.return_value = torch.randn(2, 1, 16)
        self.model.output_heads.return_value = torch.randn(2, 5, 1)
        self.model.output_heads.predict.return_value = torch.randn(2, 5, 1)


        predictions = self.model.forecast(
            inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

if __name__ == '__main__':
    unittest.main()