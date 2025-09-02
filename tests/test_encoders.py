import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.transformer_block_config import EncoderBlockConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.models.module_builder_helper import ModuleBuilder

class MockEncoderLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # Return a tuple to match the expected output format of a real layer
        return (hidden_states,)

class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.config = TransformerTimeSeriesConfig(
            d_model=16,
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")] * 2,
            hidden_dropout_prob=0.1,
            layer_norm_config=NormalizationConfig(type="layer"),
        )
        self.builder = MagicMock(spec=ModuleBuilder)
        self.builder._build.return_value = MockEncoderLayer()

        # The constructor expects the builder. Layers are now built internally.
        self.encoder = TimeSeriesTransformerEncoder(
            config=self.config,
            builder=self.builder,
            block_configs=self.config.encoder_blocks,
        )

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(hidden_states, attention_mask)
        self.assertEqual(output.last_hidden_state.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(
            hidden_states, attention_mask, output_attentions=True
        )
        self.assertIsNotNone(output.attentions)

    def test_forward_with_output_hidden_states(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(
            hidden_states, attention_mask, output_hidden_states=True
        )
        self.assertIsNotNone(output.hidden_states)

if __name__ == "__main__":
    unittest.main()