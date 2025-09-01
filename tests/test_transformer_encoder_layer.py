import torch
import unittest
from unittest.mock import MagicMock
from temporal.modules.encoders.transformer_encoder_layer import TimeSeriesTransformerEncoderLayer
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.normalization_config import NormalizationConfig

class TestTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.attention_config = MagicMock(spec=AttentionConfig)
        self.config.ffn_config = StandardFeedForwardConfig(
            type="standard", intermediate_size=32
        )
        # This config is for the layer itself, to be resolved by the builder
        self.config.normalization_config = MagicMock(spec=NormalizationConfig)
        
        self.builder = MagicMock()
        
        # Mock the builder's resolution method to return a valid, callable nn.Module
        # for both normalization layers required by the encoder layer.
        self.builder.build_normalization.side_effect = [
            MagicMock(spec=torch.nn.Module)(), 
            MagicMock(spec=torch.nn.Module)()
        ]
        
        self.encoder_layer = TimeSeriesTransformerEncoderLayer(self.config, self.builder)

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder_layer(hidden_states)
        self.assertEqual(output['hidden_states'].shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        
        # Mock the attention module to return weights in the expected tuple format
        self.encoder_layer.self_attn.return_value = (
            torch.randn(2, 10, 16),      # hidden_states
            torch.randn(2, 4, 10, 10),   # attention_weights
        )
        
        output = self.encoder_layer(hidden_states, output_attentions=True)
        
        self.assertIn('attention_weights', output)
        self.assertIsNotNone(output['attention_weights'])
        self.assertEqual(output['attention_weights'].shape, (2, 4, 10, 10))


if __name__ == "__main__":
    unittest.main()
