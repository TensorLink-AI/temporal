import unittest
import torch
from unittest.mock import MagicMock
from temporal.modules.encoders.transformer_encoder_layer import (
    TimeSeriesTransformerEncoderLayer,
)
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.models.module_builder_helper import ModuleBuilder

class TestTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.attention_config = MagicMock(spec=AttentionConfig)
        self.config.ffn_config = StandardFeedForwardConfig(
            type="standard", intermediate_size=32
        )
        self.config.normalization_config = MagicMock(spec=NormalizationConfig)
        self.config.hidden_dropout_prob = 0.1  # Set dropout probability
        self.builder = MagicMock(spec=ModuleBuilder)
        self.builder.build_normalization.return_value = NormalizationConfig(type="layer")
        
        # Mock the builder to return mock modules
        self.builder.build_attention.return_value = MagicMock(return_value=(torch.randn(2, 10, 16), None))
        self.builder.build_feedforward.return_value = MagicMock(return_value=torch.randn(2, 10, 16))
        self.builder.build_normalization.return_value = MagicMock(return_value=torch.randn(2, 10, 16))

        self.encoder_layer = TimeSeriesTransformerEncoderLayer(self.config, self.builder)

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder_layer(hidden_states)
        self.assertEqual(output[0].shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        output = self.encoder_layer(hidden_states, output_attentions=True)
        self.assertIsNotNone(output[1])

if __name__ == "__main__":
    unittest.main()