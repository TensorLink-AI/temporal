import torch
import unittest
from temporal.modules.attentions.destationary import Projector

class TestDestationary(unittest.TestCase):

    def setUp(self):
        self.enc_in = 16
        self.seq_len = 10
        self.hidden_dims = [32, 64]
        self.hidden_layers = 2
        self.output_dim = 8
        self.batch_size = 2

    def test_projector(self):
        projector = Projector(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            hidden_dims=self.hidden_dims,
            hidden_layers=self.hidden_layers,
            output_dim=self.output_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.enc_in)
        stats = torch.randn(self.batch_size, 1, self.enc_in)
        y = projector(x, stats)
        self.assertEqual(y.shape, (self.batch_size, self.output_dim))

if __name__ == '__main__':
    unittest.main()
