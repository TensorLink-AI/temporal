import torch
import unittest
from unittest.mock import MagicMock, patch
from temporal.models.mixin.autoregressive import AutoregressiveDispatchMixin

class MockPatchModel(torch.nn.Module, AutoregressiveDispatchMixin):
    def __init__(self):
        super().__init__()
        self.preprocessor = MagicMock()
        self.preprocessor.value_embedding.patch_size = 5

class MockStepwiseModel(torch.nn.Module, AutoregressiveDispatchMixin):
    def __init__(self):
        super().__init__()
        self.preprocessor = MagicMock()
        # Remove patch_size to simulate a stepwise model
        del self.preprocessor.value_embedding.patch_size

class TestAutoregressiveDispatchMixin(unittest.TestCase):

    @patch('temporal.models.mixin.autoregressive_patch.AutoregressivePatchMixin.forecast')
    def test_forecast_patch_based(self, mock_forecast):
        model = MockPatchModel()
        model.forecast(torch.randn(2, 10, 1), 5)
        mock_forecast.assert_called_once()

    @patch('temporal.models.mixin.autoregressive_stepwise.AutoregressiveStepwiseMixin.forecast')
    def test_forecast_stepwise(self, mock_forecast):
        model = MockStepwiseModel()
        model.forecast(torch.randn(2, 10, 1), 5)
        mock_forecast.assert_called_once()

    @patch('temporal.models.mixin.autoregressive_patch.AutoregressivePatchMixin.generate')
    def test_generate_patch_based(self, mock_generate):
        model = MockPatchModel()
        model.generate(decoder_inputs=torch.randn(2, 10, 1))
        mock_generate.assert_called_once()

    @patch('temporal.models.mixin.autoregressive_stepwise.AutoregressiveStepwiseMixin.generate')
    def test_generate_stepwise(self, mock_generate):
        model = MockStepwiseModel()
        model.generate(decoder_inputs=torch.randn(2, 10, 1))
        mock_generate.assert_called_once()

if __name__ == '__main__':
    unittest.main()
