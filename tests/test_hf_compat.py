import unittest
from temporal.hf_compat.config_wrapper import HFCompatibleTimeSeriesConfig
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig

class TestHFCompatibleTimeSeriesConfig(unittest.TestCase):

    def test_from_custom(self):
        custom_config = TransformerTimeSeriesConfig()
        hf_config = HFCompatibleTimeSeriesConfig.from_custom(custom_config)
        self.assertEqual(hf_config.model_type, "transformer_time_series")

    def test_to_custom(self):
        hf_config = HFCompatibleTimeSeriesConfig()
        custom_config = hf_config.to_custom()
        self.assertIsInstance(custom_config, TransformerTimeSeriesConfig)

if __name__ == '__main__':
    unittest.main()
