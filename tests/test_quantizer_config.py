import unittest
from temporal.configs.quantizer_config import QuantizerConfig, quantizer_config_from_dict

class TestQuantizerConfig(unittest.TestCase):

    def test_quantizer_config_creation(self):
        config = QuantizerConfig(
            type="mean_std_bins",
            vocab_size=1024,
            num_features=4,
        )
        self.assertEqual(config.type, "mean_std_bins")
        self.assertEqual(config.vocab_size, 1024)
        self.assertEqual(config.num_features, 4)

    def test_quantizer_config_validation(self):
        with self.assertRaises(ValueError):
            QuantizerConfig(vocab_size=0)
        with self.assertRaises(ValueError):
            QuantizerConfig(num_features=0)

    def test_quantizer_config_from_dict(self):
        config_dict = {
            "type": "mean_std_bins",
            "vocab_size": 1024,
            "num_features": 4,
        }
        config = quantizer_config_from_dict(config_dict)
        self.assertIsInstance(config, QuantizerConfig)
        self.assertEqual(config.vocab_size, 1024)

if __name__ == '__main__':
    unittest.main()
