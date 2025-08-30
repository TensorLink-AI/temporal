import torch
import unittest
from unittest.mock import MagicMock
from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder

class TestEncoders(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.encoder_layerdrop = 0.1
        self.builder = MagicMock()
        self.block_configs = [MagicMock()] * 2
        self.encoder = TimeSeriesTransformerEncoder(self.config, self.builder, self.block_configs)

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder(hidden_states)
        self.assertEqual(output.last_hidden_state.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        self.encoder.layers[0].return_value.attention_weights = torch.randn(2, 4, 10, 10)
        output = self.encoder(hidden_states, output_attentions=True)
        self.assertIsNotNone(output.attentions)
        self.assertEqual(len(output.attentions), 2)

    def test_forward_with_output_hidden_states(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder(hidden_states, output_hidden_states=True)
        self.assertIsNotNone(output.hidden_states)
        self.assertEqual(len(output.hidden_states), 3)

if __name__ == '__main__':
    unittest.main()
