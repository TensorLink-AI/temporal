import unittest
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig

class TestTransformerModelConfig(unittest.TestCase):

    def test_transformer_time_series_config_creation(self):
        config = TransformerTimeSeriesConfig(
            d_model=32,
            feature_size=4,
        )
        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.feature_size, 4)

    def test_transformer_time_series_config_validation(self):
        with self.assertRaises(ValueError):
            TransformerTimeSeriesConfig(d_model=0)
        with self.assertRaises(ValueError):
            TransformerTimeSeriesConfig(hidden_dropout_prob=1.1)
        with self.assertRaises(ValueError):
            TransformerTimeSeriesConfig(max_position_embeddings=0)

    def test_transformer_time_series_config_from_dict(self):
        config_dict = {
            "d_model": 32,
            "feature_size": 4,
            "architecture": {"type": "encoder-decoder"},
            "value_embedding_config": {"type": "value", "feature_size": 4},
            "positional_embedding_config": {"type": "sinusoidal"},
            "encoder_blocks": [{"type": "encoder"}],
            "decoder_blocks": [{"type": "decoder"}],
            "output_head_config": {"type": "linear", "output_size": 4},
            "layer_norm_config": {"type": "layer"},
            "loss_config": {"type": "mse"},
        }
        config = TransformerTimeSeriesConfig.from_dict(config_dict)
        self.assertIsInstance(config, TransformerTimeSeriesConfig)
        self.assertEqual(config.d_model, 32)

if __name__ == '__main__':
    unittest.main()
