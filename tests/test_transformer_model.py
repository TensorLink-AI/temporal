import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from temporal.models.transformer_model import TransformerTemporalModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.models.module_builder_helper import ModuleBuilder

class MockLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # A minimal forward pass that returns a tuple as expected by the model
        return (hidden_states,)

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.builder = MagicMock(spec=ModuleBuilder)
        self.builder._build.return_value = MockLayer()
        self.builder.build_loss.return_value = nn.MSELoss() # Provide a default loss

    def test_forward_encoder_decoder(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            d_model=16,
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 1)
        decoder_inputs = torch.randn(2, 5, 1)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        self.assertIn("point", output)

    def test_forward_encoder_only(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder"
            ),
            d_model=16,
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 1)
        output = model(encoder_inputs=encoder_inputs)
        self.assertIn("point", output)

    def test_forward_decoder_only(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="decoder"
            ),
            d_model=16,
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        decoder_inputs = torch.randn(2, 5, 1)
        output = model(decoder_inputs=decoder_inputs)
        self.assertIn("point", output)

    def test_forward_with_loss(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            d_model=16,
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 1)
        decoder_inputs = torch.randn(2, 5, 1)
        targets = torch.randn(2, 5, 1)
        output = model(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            targets=targets,
        )
        self.assertIn("loss", output)

if __name__ == "__main__":
    unittest.main()