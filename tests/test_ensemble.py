import torch
import torch.nn as nn
import unittest
from temporal.utils.ensemble import EnsembleSampler

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(self.dropout(x))

    def generate(self, **kwargs):
        return self.forward(torch.randn(1, 10))

class TestEnsembleSampler(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.sampler = EnsembleSampler(self.model)

    def test_dropout_enabled(self):
        self.sampler.dropout_enabled = True
        self.sampler._set_dropout_mode(True)
        self.assertTrue(self.model.dropout.training)

    def test_dropout_disabled(self):
        self.sampler.dropout_enabled = False
        self.sampler._set_dropout_mode(False)
        self.assertFalse(self.model.dropout.training)

    def test_generate(self):
        predictions = self.sampler.generate(ensemble_size=5)
        self.assertEqual(predictions.shape, (1, 5, 1, 10))

    def test_context_manager(self):
        with torch.no_grad():
            try:
                self.sampler.generate(ensemble_size=2)
            except Exception:
                pass
        self.assertFalse(self.model.dropout.training)

if __name__ == '__main__':
    unittest.main()
