import unittest
from unittest.mock import patch
from temporal.hf_compat.config_wrapper import HFCompatibleTimeSeriesConfig
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig

class TestHFCompatibleTimeSeriesConfig(unittest.TestCase):

    @patch('temporal.configs.transformer_model_config.TransformerTimeSeriesConfig')
    def test_from_custom(self, MockTransformerTimeSeriesConfig):
        mock_instance = MockTransformerTimeSeriesConfig.return_value
        mock_instance.to_dict.return_value = {
            "architecture": {"type": "transformer_architecture", "layout": "encoder-decoder"}
        }
        hf_config = HFCompatibleTimeSeriesConfig.from_custom(mock_instance)
        self.assertEqual(hf_config.model_type, "transformer_time_series")
        self.assertEqual(hf_config.architecture["layout"], "encoder-decoder")

    def test_to_custom(self):
        hf_config = HFCompatibleTimeSeriesConfig(
            model_type="transformer_time_series",
            architecture={"type": "transformer_architecture", "layout": "encoder-decoder"}
        )
        custom_config = hf_config.to_custom()
        self.assertIsInstance(custom_config, TransformerTimeSeriesConfig)
        self.assertEqual(custom_config.architecture.layout, "encoder-decoder")

if __name__ == '__main__':
    unittest.main()
