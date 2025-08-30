import torch
import unittest
from unittest.mock import MagicMock
from temporal.modules.encoders.transformer_encoder_layer import TimeSeriesTransformerEncoderLayer

class TestTransformerEncoderLayer(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.attention_config = MagicMock()
        self.config.ffn_config = MagicMock()
        self.config.normalization_config = MagicMock()
        self.builder = MagicMock()
        self.encoder_layer = TimeSeriesTransformerEncoderLayer(self.config, self.builder)

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder_layer(hidden_states)
        self.assertEqual(output.hidden_states.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        self.encoder_layer.self_attn.return_value = (torch.randn(2, 10, 16), torch.randn(2, 4, 10, 10))
        output = self.encoder_layer(hidden_states, output_attentions=True)
        self.assertIsNotNone(output.attention_weights)
        self.assertEqual(output.attention_weights.shape, (2, 4, 10, 10))

if __name__ == '__main__':
    unittest.main()
