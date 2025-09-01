import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from temporal.models.mixin.autoregressive_stepwise import AutoregressiveStepwiseMixin

# --- Mock Wrapper ---
class MockHeadWrapper(torch.nn.Module):
    def __init__(self, module_dict):
        super().__init__()
        self._modules_dict = module_dict

    def forward(self, *args, **kwargs):
        # Delegate to the 'point' head, which is the default
        return self._modules_dict['point'](*args, **kwargs)

# FIX: A robust, self-contained FakeHead for mocking
class FakeHead(nn.Module):
    def forward(self, hidden_states):
        # Mimic the forward pass, returning a parameter dictionary
        return {"params": torch.randn(hidden_states.size(0), hidden_states.size(1), 1)}

    def predict(self, params, method="mean"):
        # Prediction is based on the tensor from the forward pass
        tensor = self._extract_tensor(params)
        return torch.randn(tensor.size(0), tensor.size(1), 1)

    def sample(self, params, **kwargs):
        # Sampling is based on the tensor from the forward pass
        tensor = self._extract_tensor(params)
        return torch.randn(tensor.size(0), tensor.size(1), 1)
    
    def _extract_tensor(self, params):
        if not isinstance(params, dict):
            return params
        
        if "point" in params and isinstance(params["point"], dict) and "params" in params["point"]:
            return params["point"]["params"]
        
        if "params" in params:
            return params["params"]
            
        raise ValueError("Could not find a 'params' tensor in the mocked head outputs.")


class MockModel(nn.Module, AutoregressiveStepwiseMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessor = MagicMock()
        self.decoder = MagicMock()
        self.output_heads = MockHeadWrapper(nn.ModuleDict({"point": FakeHead()}))

    def _get_primary_head(self):
        return self.output_heads._modules_dict["point"]

class TestAutoregressiveStepwiseMixin(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.feature_size = 1
        self.config.prediction_length = 5
        self.model = MockModel(self.config)
        self.model.preprocessor.denormalize.return_value = torch.randn(2, 5, 1)

    def test_generate(self):
        decoder_inputs = torch.randn(2, 10, 1)
        # Use realistic tensor shapes
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 10, 16), # Realistic hidden dim
            "attention_mask": torch.ones(2, 10),
        }
        self.model.decoder.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 10, 16),
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
            "hidden_states": torch.randn(2, 10, 16),
            "attention_mask": torch.ones(2, 10),
        }
        self.model.decoder.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 10, 16),
            past_key_values=None
        )

        predictions = self.model.forecast(
            inputs,
            prediction_length=5,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

if __name__ == '__main__':
    unittest.main()
