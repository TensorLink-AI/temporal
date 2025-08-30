import torch
import unittest
from unittest.mock import MagicMock
from temporal.models.transformer_model import TransformerTemporalModel

class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.architecture.layout = "encoder-decoder"
        self.config.encoder_blocks = [MagicMock()]
        self.config.decoder_blocks = [MagicMock()]
        self.config.loss_config.type = "mse"
        self.builder = MagicMock()
        self.model = TransformerTemporalModel(self.config, builder=self.builder)

    def test_forward_encoder_decoder(self):
        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        output = self.model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_encoder_only(self):
        self.config.architecture.layout = "encoder-only"
        self.model = TransformerTemporalModel(self.config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 4)
        output = self.model(encoder_inputs=encoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_decoder_only(self):
        self.config.architecture.layout = "decoder-only"
        self.model = TransformerTemporalModel(self.config, builder=self.builder)
        decoder_inputs = torch.randn(2, 5, 4)
        output = self.model(decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

    def test_forward_with_loss(self):
        encoder_inputs = torch.randn(2, 10, 4)
        decoder_inputs = torch.randn(2, 5, 4)
        targets = torch.randn(2, 5, 4)
        output = self.model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, targets=targets)
        self.assertIsNotNone(output.loss)

if __name__ == '__main__':
    unittest.main()
