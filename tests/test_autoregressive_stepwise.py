import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from temporal.models.mixin.autoregressive_stepwise import AutoregressiveStepwiseMixin

class MockModel(nn.Module, AutoregressiveStepwiseMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessor = MagicMock()
        self.decoder = MagicMock()
        self.output_heads = MagicMock()

class TestAutoregressiveStepwiseMixin(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.feature_size = 1
        self.config.prediction_length = 5
        self.model = MockModel(self.config)

    def test_generate(self):
        decoder_inputs = torch.randn(2, 10, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 1, 1),
            "attention_mask": torch.ones(2, 1),
        }
        self.model.decoder.return_value = {
            "last_hidden_state": torch.randn(2, 1, 1),
        }
        self.model.output_heads.return_value = torch.randn(2, 1, 1)

        predictions = self.model.generate(
            decoder_inputs=decoder_inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

    def test_forecast(self):
        inputs = torch.randn(2, 10, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 1, 1),
            "attention_mask": torch.ones(2, 1),
        }
        self.model.decoder.return_value = {
            "last_hidden_state": torch.randn(2, 1, 1),
        }
        self.model.output_heads.return_value = torch.randn(2, 1, 1)

        predictions = self.model.forecast(
            inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

if __name__ == '__main__':
    unittest.main()
