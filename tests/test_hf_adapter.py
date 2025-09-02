import unittest
import torch
from temporal.utils.hf_adapter import TimeSeriesTransformerModel
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.loss_config import MSELossConfig

class TestTimeSeriesTransformerModel(unittest.TestCase):
    def setUp(self):
        # Using a real config object prevents most mocking issues.
        # We create it from a dictionary to make it mutable.
        config_dict = TransformerTimeSeriesConfig(
            feature_size=1,
            d_model=16,
            context_length=10,
            prediction_length=5,
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder"),
            output_head_config=OutputHeadConfig(type="linear", output_size=1),
            loss_config=MSELossConfig(type="mse"),
        ).to_dict()
        
        # The Hugging Face model expects this attribute to be set.
        config_dict['_attn_implementation'] = "eager"
        
        # Re-create the config from the mutable dictionary
        self.config = TransformerTimeSeriesConfig.from_dict(config_dict)
        
        self.model = TimeSeriesTransformerModel(self.config)
        self.model.eval()

    def test_forward_pass(self):
        encoder_inputs = torch.randn(2, self.config.context_length, self.config.feature_size)
        decoder_inputs = torch.randn(2, self.config.prediction_length, self.config.feature_size)
        outputs = self.model(
            encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs
        )
        self.assertIn("logits", outputs)
        self.assertEqual(
            outputs.logits.shape,
            (2, self.config.prediction_length, self.config.output_head_config.output_size),
        )

    def test_generate_pass(self):
        encoder_inputs = torch.randn(2, self.config.context_length, self.config.feature_size)
        prediction_length = 5
        generated_outputs = self.model.generate(
            encoder_inputs=encoder_inputs, max_length=prediction_length
        )
        self.assertEqual(
            generated_outputs.shape,
            (2, prediction_length, self.config.output_head_config.output_size),
        )

    def test_generate_pass_with_config_prediction_length(self):
        encoder_inputs = torch.randn(2, self.config.context_length, self.config.feature_size)
        generated_outputs = self.model.generate(encoder_inputs=encoder_inputs)
        self.assertEqual(
            generated_outputs.shape,
            (
                2,
                self.config.prediction_length,
                self.config.output_head_config.output_size,
            ),
        )

if __name__ == "__main__":
    unittest.main()
