import unittest
from temporal.configs.feedforward_config import (
    StandardFeedForwardConfig,
    MoEFeedForwardConfig,
    feedforward_config_from_dict,
)

class TestFeedForwardConfig(unittest.TestCase):

    def test_standard_feedforward_config(self):
        config = StandardFeedForwardConfig(intermediate_size=128)
        self.assertEqual(config.intermediate_size, 128)
        with self.assertRaises(ValueError):
            StandardFeedForwardConfig(intermediate_size=0)

    def test_moe_feedforward_config(self):
        config = MoEFeedForwardConfig(num_experts=4, top_k=2)
        self.assertEqual(config.num_experts, 4)
        self.assertEqual(config.top_k, 2)
        with self.assertRaises(ValueError):
            MoEFeedForwardConfig(num_experts=0)
        with self.assertRaises(ValueError):
            MoEFeedForwardConfig(top_k=0)
        with self.assertRaises(ValueError):
            MoEFeedForwardConfig(top_k=5, num_experts=4)

    def test_feedforward_config_from_dict(self):
        config_dict = {"type": "standard", "intermediate_size": 128}
        config = feedforward_config_from_dict(config_dict)
        self.assertIsInstance(config, StandardFeedForwardConfig)

        config_dict = {"type": "moe", "num_experts": 4, "top_k": 2}
        config = feedforward_config_from_dict(config_dict)
        self.assertIsInstance(config, MoEFeedForwardConfig)

if __name__ == '__main__':
    unittest.main()
