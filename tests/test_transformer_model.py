import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, create_autospec
from temporal.models.transformer_model import TransformerTemporalModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.transformer_block_config import EncoderBlockConfig, DecoderBlockConfig

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.builder = MagicMock()
        self.builder.build_preprocessor.return_value = nn.Module()
        self.builder.build_encoder.return_value = nn.Module()
        self.builder.build_decoder.return_value = nn.Module()
        self.builder.build_output_heads.return_value = nn.ModuleDict()
        self.builder.build_loss.return_value = nn.Module()

    def test_forward_encoder_decoder(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)

        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_encoder_only(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder"
            ),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)

        encoder_inputs = torch.randn(2, 10, 4)
        output = model(encoder_inputs=encoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_decoder_only(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="decoder"
            ),
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)

        decoder_inputs = torch.randn(2, 5, 4)
        output = model(decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_with_loss(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)

        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        targets = torch.randn(2, 5, 4)
        output = model(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            targets=targets,
        )
        self.assertIsNotNone(output.loss)


if __name__ == "__main__":
    unittest.main()
