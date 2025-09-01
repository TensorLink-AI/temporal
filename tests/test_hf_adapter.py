import torch
import unittest
from unittest.mock import MagicMock
from temporal.utils.hf_adapter import TimeSeriesTransformerModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.loss_config import LossConfig

class TestTimeSeriesTransformerModel(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock(spec=TransformerTimeSeriesConfig)
        self.config.prediction_length = 10
        self.config.architecture = TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder")
        self.config._attn_implementation = "eager"
        # Add nested mock configs
        self.config.output_head_config = MagicMock(spec=OutputHeadConfig)
        self.config.loss_config = MagicMock(spec=LossConfig)

        self.model = TimeSeriesTransformerModel(self.config)
        self.model.temporal = MagicMock()

    def test_forward_pass(self):
        input_values = torch.randn(2, 4, 8)
        attention_mask = torch.ones(2, 4)
        self.model.forward(input_values=input_values, attention_mask=attention_mask, decoder_inputs=None, targets=None)
        self.model.temporal.forward.assert_called_once_with(
            encoder_inputs=input_values,
            attention_mask=attention_mask,
            decoder_inputs=None,
            targets=None
        )

    def test_generate_pass(self):
        input_values = torch.randn(2, 4, 8)
        self.model.generate(input_values=input_values, prediction_length=5)
        self.model.temporal.generate.assert_called_once_with(
            encoder_inputs=input_values,
            prediction_length=5
        )

    def test_generate_pass_with_config_prediction_length(self):
        input_values = torch.randn(2, 4, 8)
        self.model.generate(input_values=input_values)
        self.model.temporal.generate.assert_called_once_with(
            encoder_inputs=input_values,
            prediction_length=self.config.prediction_length
        )

if __name__ == '__main__':
    unittest.main()
