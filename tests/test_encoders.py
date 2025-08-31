import torch
import unittest
from unittest.mock import MagicMock
from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig

# FIX: Create a simple stub nn.Module for type checking
class StubModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # Return a dict to mimic the expected output of a layer
        return {"hidden_states": args[0]}

class TestEncoders(unittest.TestCase):

    def setUp(self):
        self.config = TransformerTimeSeriesConfig(architecture=TransformerArchitectureConfig())
        self.builder = MagicMock()
        # FIX: The builder must return a valid nn.Module instance
        self.builder.build_block.return_value = StubModule()
        
        # We can still use MagicMock for block_configs as they are just configuration holders
        self.block_configs = [MagicMock()] * 2
        self.encoder = TimeSeriesTransformerEncoder(self.config, self.builder, self.block_configs)

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder(hidden_states)
        self.assertEqual(output.last_hidden_state.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        # Mock the return value of the stub to include attention weights
        self.builder.build_block.return_value.forward = MagicMock(
            return_value={"hidden_states": hidden_states, "attention_weights": torch.randn(2, 4, 10, 10)}
        )
        # Re-initialize encoder with the updated mock
        self.encoder = TimeSeriesTransformerEncoder(self.config, self.builder, self.block_configs)
        
        output = self.encoder(hidden_states, output_attentions=True)
        self.assertIsNotNone(output.attentions)
        self.assertEqual(len(output.attentions), 2)

    def test_forward_with_output_hidden_states(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder(hidden_states, output_hidden_states=_content)
        self.assertIsNotNone(output.hidden_states)
        self.assertEqual(len(output.hidden_states), 3)

if __name__ == '__main__':
    unittest.main()