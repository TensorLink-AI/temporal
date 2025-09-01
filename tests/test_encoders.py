import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.configs.transformer_block_config import EncoderBlockConfig

# A mock layer that returns a dict, as expected by the encoder
class MockEncoderLayer(nn.Module):
    def __init__(self, return_attentions=False):
        super().__init__()
        self.return_attentions = return_attentions

    def forward(self, hidden_states, **kwargs):
        output = {"hidden_states": hidden_states}
        if self.return_attentions:
            output["attention_weights"] = torch.randn(2, 4, 10, 10)
        return output

class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.config = TransformerTimeSeriesConfig(
            d_model=16,
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")] * 2, # Add block_configs
            hidden_dropout_prob=0.1,
            layer_norm_config=NormalizationConfig(type="layer"),
        )
        self.layers = nn.ModuleList([MockEncoderLayer(), MockEncoderLayer()])
        # Correctly instantiate the encoder with the config and layers
        self.encoder = TimeSeriesTransformerEncoder(
            config=self.config,
            block_configs=self.config.encoder_blocks,
            layers=self.layers
        )
        
        self.encoder.layer_norm = MagicMock(return_value=torch.randn(2, 10, 16))


    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder(hidden_states)
        self.assertEqual(output.last_hidden_state.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        self.layers = nn.ModuleList([
            MockEncoderLayer(return_attentions=True),
            MockEncoderLayer(return_attentions=True),
        ])
        # Re-initialize the encoder with the new layers
        self.encoder = TimeSeriesTransformerEncoder(
            config=self.config,
            block_configs=self.config.encoder_blocks,
            layers=self.layers
        )
        self.encoder.layer_norm = MagicMock(return_value=torch.randn(2, 10, 16))

        output = self.encoder(hidden_states, output_attentions=True)
        self.assertIsNotNone(output.attentions)
        self.assertEqual(len(output.attentions), 2)

    def test_forward_with_output_hidden_states(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder(hidden_states, output_hidden_states=True)
        self.assertIsNotNone(output.hidden_states)
        self.assertEqual(len(output.hidden_states), 3) # Initial + 2 layers

if __name__ == '__main__':
    unittest.main()
