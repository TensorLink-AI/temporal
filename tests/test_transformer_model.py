import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from temporal.models.transformer_model import TransformerTemporalModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.transformer_block_config import EncoderBlockConfig, DecoderBlockConfig

# Define a simple mock layer that is a subclass of nn.Module
class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Give it a parameter to be discoverable by things like model.parameters()
        self.param = nn.Parameter(torch.empty(1))

    def forward(self, hidden_states, **kwargs):
        # Return a dictionary to mimic the actual block's output structure
        return {"hidden_states": hidden_states}

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.builder = MagicMock()
        # Configure the builder to return instances of our valid nn.Module subclass
        mock_module_instance = MockLayer()
        self.builder.build_preprocessor.return_value = mock_module_instance
        self.builder.build_encoder.return_value = mock_module_instance
        self.builder.build_decoder.return_value = mock_module_instance
        self.builder.build_output_heads.return_value = nn.ModuleDict({"point": mock_module_instance})
        self.builder.build_loss.return_value = MagicMock()

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
        
        encoder_inputs = torch.randn(2, 10, 16)
        decoder_inputs = torch.randn(2, 5, 16)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

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

        encoder_inputs = torch.randn(2, 10, 16)
        output = model(encoder_inputs=encoder_inputs)
        self.assertIsNotNone(output.logits)

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

        decoder_inputs = torch.randn(2, 5, 16)
        output = model(decoder_inputs=decoder_inputs)
        self.assertIsNotNone(output.logits)

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
        model.loss_fn = MagicMock(return_value=torch.tensor(0.5))

        encoder_inputs = torch.randn(2, 10, 16)
        decoder_inputs = torch.randn(2, 5, 16)
        targets = torch.randn(2, 5, 16)
        output = model(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            targets=targets,
        )
        self.assertIsNotNone(output.loss)

if __name__ == "__main__":
    unittest.main()
