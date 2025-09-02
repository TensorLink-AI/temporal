import torch
import unittest
from temporal.modules.norm.dynamic_revin import DynamicRevIN

class TestDynamicRevIN(unittest.TestCase):

    def setUp(self):
        self.num_features = 4
        self.seq_len = 10
        self.batch_size = 2
        self.revin_fixed = DynamicRevIN(self.num_features, affine_mode='fixed')
        self.revin_dynamic = DynamicRevIN(
            self.num_features,
            affine_mode={'type': 'dynamic', 'mapper': 'linear'}
        )

    def test_fixed_norm_denorm(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        x_norm = self.revin_fixed(x, 'norm')
        x_denorm = self.revin_fixed(x_norm, 'denorm')
        self.assertTrue(torch.allclose(x, x_denorm, atol=1e-5))

    def test_dynamic_norm_denorm(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        x_norm = self.revin_dynamic(x, 'norm')
        x_denorm = self.revin_dynamic(x_norm, 'denorm')
        self.assertTrue(torch.allclose(x, x_denorm, atol=1e-5))

    def test_fixed_transform_inverse_transform(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        self.revin_fixed(x, 'norm')
        x_transform = self.revin_fixed.transform(x)
        x_inverse_transform = self.revin_fixed.inverse_transform(x_transform)
        self.assertTrue(torch.allclose(x, x_inverse_transform, atol=1e-5))

    def test_dynamic_transform_inverse_transform(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        self.revin_dynamic(x, 'norm')
        x_transform = self.revin_dynamic.transform(x)
        x_inverse_transform = self.revin_dynamic.inverse_transform(x_transform)
        self.assertTrue(torch.allclose(x, x_inverse_transform, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
