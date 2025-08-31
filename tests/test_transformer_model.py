import torch
import unittest
from unittest.mock import MagicMock
from temporal.models.transformer_model import TransformerTemporalModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.transformer_block_config import EncoderBlockConfig, DecoderBlockConfig

class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        # A generic builder for all tests
        self.builder = MagicMock()
        self.builder.build_preprocessor.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_encoder.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_decoder.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_output_heads.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_loss.return_value = MagicMock(spec=torch.nn.Module)

    def test_forward_encoder_decoder(self):
        # FIX: Create a specific, valid config for this test
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder"),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1 # Provide a valid float
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        model.temporal = MagicMock(return_value=MagicMock(logits=torch.randn(2, 5, 1)))

        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_encoder_only(self):
        # FIX: Create a specific, valid config for this test
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-only"),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            hidden_dropout_prob=0.1
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        model.temporal = MagicMock(return_value=MagicMock(logits=torch.randn(2, 10, 1)))

        encoder_inputs = torch.randn(2, 10, 4)
        output = model(encoder_inputs=encoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_decoder_only(self):
        # FIX: Create a specific, valid config for this test
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="decoder-only"),
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        model.temporal = MagicMock(return_value=MagicMock(logits=torch.randn(2, 5, 1)))
        
        decoder_inputs = torch.randn(2, 5, 4)
        output = model(decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_with_loss(self):
        # FIX: Create a specific, valid config for this test
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder"),
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        model.temporal = MagicMock(return_value=MagicMock(logits=torch.randn(2, 5, 1), loss=torch.tensor(0.5)))
        
        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        targets = torch.randn(2, 5, 4)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, targets=targets)
        self.assertIsNotNone(output.loss)

if __name__ == '__main__':
    unittest.main()