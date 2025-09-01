import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from temporal.models.mixin.multistep import MultiStepMixin

# Custom mock nn.Module to satisfy type checks
class MockNNModule(nn.Module):
    def __init__(self, mock_obj):
        super().__init__()
        self.mock = mock_obj

    def forward(self, *args, **kwargs):
        return self.mock(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate attribute access to the internal mock
        return getattr(self.mock, name)


class MockModel(nn.Module, MultiStepMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessor = MagicMock()
        # Use the wrapper for the decoder and the head
        self.decoder = MockNNModule(MagicMock())
        self.output_heads = nn.ModuleDict({"point": MockNNModule(MagicMock())})

    def _get_primary_head(self):
        return self.output_heads["point"]


class TestMultiStepMixin(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.feature_size = 1
        self.config.context_length = 10
        self.config.prediction_length = 5
        self.model = MockModel(self.config)

    def test_forecast_single_pass_chunked(self):
        inputs = torch.randn(2, 20, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 10, 1),
            "attention_mask": torch.ones(2, 10),
        }
        self.model.decoder.mock.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 10, 1)
        )
        primary_head = self.model._get_primary_head()
        primary_head.mock.return_value = {"params": torch.randn(2, 5, 1)}
        primary_head.predict.return_value = torch.randn(2, 5, 1)

        predictions = self.model.forecast_single_pass_chunked(
            inputs,
            prediction_length=5,
            chunk_length=5,
            context_length=10,
        )
        self.assertEqual(predictions.shape, (2, 5, 1))

    def test_forecast_single_pass_chunked_with_quantiles(self):
        inputs = torch.randn(2, 20, 1)
        self.model.preprocessor.process.return_value = {
            "hidden_states": torch.randn(2, 10, 1),
            "attention_mask": torch.ones(2, 10),
        }
        self.model.decoder.mock.return_value = SimpleNamespace(
            last_hidden_state=torch.randn(2, 10, 1)
        )
        primary_head = self.model._get_primary_head()
        primary_head.mock.return_value = {"params": torch.randn(2, 5, 1)}
        primary_head.sample_quantiles.return_value = torch.randn(2, 5, 1, 3)

        predictions = self.model.forecast_single_pass_chunked(
            inputs,
            prediction_length=5,
            chunk_length=5,
            context_length=10,
            quantiles=[0.25, 0.5, 0.75],
        )
        self.assertEqual(predictions.shape, (2, 5, 1, 3))


if __name__ == "__main__":
    unittest.main()
