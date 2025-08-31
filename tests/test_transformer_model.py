import torch
import unittest
from unittest.mock import MagicMock
from temporal.models.transformer_model import TransformerTemporalModel
# FIX: Import real config objects for robust testing
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig

class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        # FIX: Use a real config object to provide necessary attributes like dropout probability
        self.config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder")
        )
        self.builder = MagicMock()
        # Mock the builder's methods to return simple nn.Module instances
        self.builder.build_preprocessor.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_encoder.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_decoder.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_output_heads.return_value = MagicMock(spec=torch.nn.Module)
        self.builder.build_loss.return_value = MagicMock(spec=torch.nn.Module)
        
        self.model = TransformerTemporalModel(self.config, builder=self.builder)

    def test_forward_encoder_decoder(self):
        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        # Mock the forward call of the internal temporal model
        self.model.temporal.forward.return_value = MagicMock(logits=torch.randn(2, 5, 1))
        output = self.model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_encoder_only(self):
        self.config.architecture.layout = "encoder-only"
        self.model = TransformerTemporalModel(self.config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 4)
        self.model.temporal.forward.return_value = MagicMock(logits=torch.randn(2, 10, 1))
        output = self.model(encoder_inputs=encoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_decoder_only(self):
        self.config.architecture.layout = "decoder-only"
        self.model = TransformerTemporalModel(self.config, builder=self.builder)
        decoder_inputs = torch.randn(2, 5, 4)
        self.model.temporal.forward.return_value = MagicMock(logits=torch.randn(2, 5, 1))
        output = self.model(decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_with_loss(self):
        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        targets = torch.randn(2, 5, 4)
        self.model.temporal.forward.return_value = MagicMock(logits=torch.randn(2, 5, 1), loss=torch.tensor(0.5))
        output = self.model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, targets=targets)
        self.assertIsNotNone(output.loss)

if __name__ == '__main__':
    unittest.main()