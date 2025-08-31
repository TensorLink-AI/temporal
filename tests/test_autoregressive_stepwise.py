import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from temporal.models.mixin.autoregressive_stepwise import AutoregressiveStepwiseMixin

# FIX: Use the user-suggested FakeHead for robust mocking
class FakeHead:
    def __call__(self, hidden_states):
        # Mimic forward returning a params dict or tensor
        return {"params": torch.randn(hidden_states.size(0), hidden_states.size(1), 1)}
    def predict(self, params, method="mean"):
        # Use the tensor from the params dict
        tensor = params["params"] if isinstance(params, dict) else params
        return torch.randn(tensor.size(0), tensor.size(1), 1)
    def sample(self, params, **kwargs):
        tensor = params["params"] if isinstance(params, dict) else params
        return torch.randn(tensor.size(0), tensor.size(1), 1)

class MockModel(nn.Module, AutoregressiveStepwiseMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessor = MagicMock()
        self.decoder = MagicMock()
        self._primary = FakeHead()
        # FIX: The output_heads mock is a callable that returns a dict of heads
        self.output_heads = MagicMock(return_value={"point": self._primary})

    def _get_primary_head(self):
        return self.output_heads()["point"]

class TestAutoregressiveStepwiseMixin(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.feature_size = 1
        self.config.prediction_length = 5
        self.model = MockModel(self.config)
        self.model.preprocessor.denormalize.return_value = torch.randn(2, 5, 1)

    def test_generate(self):
        decoder_inputs = torch.randn(2, 10, 1)
        # FIX: Use realistic tensor shapes
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 10, 1),
            "attention_mask": torch.ones(2, 10),
        }
        self.model.decoder.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 10, 1),
            past_key_values=None
        )
        
        predictions = self.model.generate(
            decoder_inputs=decoder_inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

    def test_forecast(self):
        inputs = torch.randn(2, 10, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 10, 1),
            "attention_mask": torch.ones(2, 10),
        }
        self.model.decoder.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 10, 1),
            past_key_values=None
        )

        predictions = self.model.forecast(
            inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

if __name__ == '__main__':
    unittest.main()