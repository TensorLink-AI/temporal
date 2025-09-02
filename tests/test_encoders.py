import unittest
import torch
import torch.nn as nn

from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.models.outputs import EncoderLayerOutput


class MockEncoderLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # Return the dataclass your encoder layer expects
        return EncoderLayerOutput(hidden_states=hidden_states, attention_weights=None)


class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.encoder = TimeSeriesTransformerEncoder(
            layers=nn.ModuleList([MockEncoderLayer(), MockEncoderLayer()]),
            layerdrop=0.0,
        )

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(hidden_states, attention_mask)
        # Your encoder returns an HF-style BaseModelOutput
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertEqual(output.last_hidden_state.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(hidden_states, attention_mask, output_attentions=True)
        # HF-style attention attribute
        self.assertTrue(hasattr(output, "attentions"))
        self.assertIsNotNone(output.attentions)


if __name__ == "__main__":
    unittest.main()
