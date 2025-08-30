import unittest
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.architecture_config import TransformerArchitectureConfig

class TestTransformerModelConfig(unittest.TestCase):

    def test_transformer_time_series_config_creation(self):
        config = TransformerTimeSeriesConfig(
            d_model=32,
            feature_size=4,
            architecture=TransformerArchitectureConfig(layout="encoder-decoder")
        )
        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.feature_size, 4)
        self.assertEqual(config.architecture.layout, "encoder-decoder")

    def test_transformer_time_series_config_validation(self):
        with self.assertRaises(ValueError):
            TransformerTimeSeriesConfig(
                d_model=0,
                architecture=TransformerArchitectureConfig(layout="encoder-decoder")
            )
        with self.assertRaises(ValueError):
            TransformerTimeSeriesConfig(
                hidden_dropout_prob=1.1,
                architecture=TransformerArchitectureConfig(layout="encoder-decoder")
            )
        with self.assertRaises(ValueError):
            TransformerTimeSeriesConfig(
                max_position_embeddings=0,
                architecture=TransformerArchitectureConfig(layout="encoder-decoder")
            )

    def test_transformer_time_series_config_from_dict(self):
        config_dict = {
            "d_model": 32,
            "feature_size": 4,
            "architecture": {"type": "transformer_architecture", "layout": "encoder-decoder"},
            "value_embedding_config": {"type": "value", "input_dim": 4, "d_model": 32},
            "positional_embedding_config": {"type": "sinusoidal", "d_model": 32},
            "encoder_blocks": [{"type": "default_encoder"}],
            "decoder_blocks": [{"type": "default_decoder"}],
            "output_head_config": {"type": "linear", "output_size": 4, "hidden_size": 32},
            "layer_norm_config": {"type": "layer_norm"},
            "loss_config": {"type": "mse"},
        }
        config = TransformerTimeSeriesConfig.from_dict(config_dict)
        self.assertIsInstance(config, TransformerTimeSeriesConfig)
        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.architecture.layout, "encoder-decoder")

if __name__ == '__main__':
    unittest.main()
