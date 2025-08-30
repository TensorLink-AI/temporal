import unittest
from temporal.configs.attention_config import (
    FullAttentionConfig,
    PatternedAttentionConfig,
    FlashAttentionConfig,
    LSEAttentionConfig,
    DiffWistAttentionConfig,
    HybridAttentionConfig,
    attention_config_from_dict,
)

class TestAttentionConfigs(unittest.TestCase):

    def test_full_attention_config(self):
        config = FullAttentionConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_patterned_attention_config(self):
        config = PatternedAttentionConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_flash_attention_config(self):
        config = FlashAttentionConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_lse_attention_config(self):
        config = LSEAttentionConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_diff_wist_attention_config(self):
        config = DiffWistAttentionConfig(num_heads=4, depth=1)
        self.assertEqual(config.num_heads, 4)
        with self.assertRaises(ValueError):
            DiffWistAttentionConfig(num_heads=4, depth=0)

    def test_hybrid_attention_config(self):
        config = HybridAttentionConfig(num_heads=4, head_splits=[2, 2], head_types=["full", "flash"])
        self.assertEqual(config.num_heads, 4)
        with self.assertRaises(ValueError):
            HybridAttentionConfig(num_heads=4, head_splits=[2, 1], head_types=["full", "flash"])

    def test_attention_config_from_dict(self):
        config_dict = {"type": "full", "num_heads": 4}
        config = attention_config_from_dict(config_dict)
        self.assertIsInstance(config, FullAttentionConfig)

        config_dict = {"type": "hybrid", "num_heads": 4, "head_splits": [2, 2], "head_types": ["full", "flash"]}
        config = attention_config_from_dict(config_dict)
        self.assertIsInstance(config, HybridAttentionConfig)

if __name__ == '__main__':
    unittest.main()
