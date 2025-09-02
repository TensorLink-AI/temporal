import torch
import unittest
from temporal.modules.norm.revin import RevIN, RevIN2d

class TestRevIN(unittest.TestCase):

    def setUp(self):
        self.num_features = 4
        self.seq_len = 10
        self.batch_size = 2
        self.revin = RevIN(self.num_features)
        self.revin2d = RevIN2d(self.num_features)

    def test_revin_norm_denorm(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        x_norm = self.revin(x, 'norm')
        x_denorm = self.revin(x_norm, 'denorm')
        self.assertTrue(torch.allclose(x, x_denorm, atol=1e-5))

    def test_revin_transform_inverse_transform(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        self.revin(x, 'norm')
        x_transform = self.revin.transform(x)
        x_inverse_transform = self.revin.inverse_transform(x_transform)
        self.assertTrue(torch.allclose(x, x_inverse_transform, atol=1e-5))

    def test_revin_subtract_last(self):
        revin_sl = RevIN(self.num_features, subtract_last=True)
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        x_norm = revin_sl(x, 'norm')
        x_denorm = revin_sl(x_norm, 'denorm')
        self.assertTrue(torch.allclose(x, x_denorm, atol=1e-5))

    def test_revin2d_norm_denorm(self):
        x = torch.randn(self.batch_size, self.num_features, self.seq_len, self.seq_len)
        x_norm = self.revin2d(x, 'norm')
        x_denorm = self.revin2d(x_norm, 'denorm')
        self.assertTrue(torch.allclose(x, x_denorm, atol=1e-5))

    def test_revin2d_transform_inverse_transform(self):
        x = torch.randn(self.batch_size, self.num_features, self.seq_len, self.seq_len)
        self.revin2d(x, 'norm')
        x_transform = self.revin2d.transform(x)
        x_inverse_transform = self.revin2d.inverse_transform(x_transform)
        self.assertTrue(torch.allclose(x, x_inverse_transform, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
