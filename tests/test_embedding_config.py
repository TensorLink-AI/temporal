import unittest
from temporal.configs.embedding_config import (
    TimeSeriesValueEmbeddingConfig,
    FlexibleValueEmbeddingConfig,
    SinusoidalPositionalEmbeddingConfig,
    TimeSeriesPatchEmbeddingConfig,
    TimeSeriesGlobalEmbeddingConfig,
    RotaryPositionalEmbeddingConfig,
    LearnedAbsolutePositionalEmbeddingConfig,
    ShawRelativePositionalBiasConfig,
    FourierFeatureEmbeddingConfig,
    Time2VecEmbeddingConfig,
    ALiBiPositionalBiasConfig,
    BucketedRelativeBiasConfig,
    ConvolutionalPositionalEmbeddingConfig,
    TimeDeltaEmbeddingConfig,
    StackedPositionalEmbeddingConfig,
    NoneEmbeddingConfig,
    S4PositionalEmbeddingConfig,
    WaveletPositionalEmbeddingConfig,
    embedding_config_from_dict,
)

class TestEmbeddingConfigs(unittest.TestCase):

    def test_time_series_value_embedding_config(self):
        config = TimeSeriesValueEmbeddingConfig(feature_size=4)
        self.assertEqual(config.feature_size, 4)

    def test_flexible_value_embedding_config(self):
        config = FlexibleValueEmbeddingConfig(input_dims=4)
        self.assertEqual(config.input_dims, 4)

    def test_sinusoidal_positional_embedding_config(self):
        config = SinusoidalPositionalEmbeddingConfig(max_seq_len=1024)
        self.assertEqual(config.max_seq_len, 1024)

    def test_time_series_patch_embedding_config(self):
        config = TimeSeriesPatchEmbeddingConfig(patch_size=5, feature_size=4)
        self.assertEqual(config.patch_size, 5)

    def test_time_series_global_embedding_config(self):
        config = TimeSeriesGlobalEmbeddingConfig(seq_len=10, feature_size=4)
        self.assertEqual(config.seq_len, 10)

    def test_rotary_positional_embedding_config(self):
        config = RotaryPositionalEmbeddingConfig()
        self.assertEqual(config.base, 10000)

    def test_learned_absolute_positional_embedding_config(self):
        config = LearnedAbsolutePositionalEmbeddingConfig()
        self.assertEqual(config.max_seq_len, 2048)

    def test_shaw_relative_positional_bias_config(self):
        config = ShawRelativePositionalBiasConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_fourier_feature_embedding_config(self):
        config = FourierFeatureEmbeddingConfig()
        self.assertEqual(config.num_features, 16)

    def test_time2vec_embedding_config(self):
        config = Time2VecEmbeddingConfig()
        self.assertTrue(config.use_cos)

    def test_alibi_positional_bias_config(self):
        config = ALiBiPositionalBiasConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_bucketed_relative_bias_config(self):
        config = BucketedRelativeBiasConfig(num_heads=4)
        self.assertEqual(config.num_heads, 4)

    def test_convolutional_positional_embedding_config(self):
        config = ConvolutionalPositionalEmbeddingConfig()
        self.assertEqual(config.kernel_size, 3)

    def test_time_delta_embedding_config(self):
        config = TimeDeltaEmbeddingConfig()
        self.assertEqual(config.hidden_dim, 64)

    def test_stacked_positional_embedding_config(self):
        config = StackedPositionalEmbeddingConfig(embedding_configs=[TimeSeriesValueEmbeddingConfig(feature_size=4)])
        self.assertEqual(len(config.embedding_configs), 1)

    def test_none_embedding_config(self):
        config = NoneEmbeddingConfig()
        self.assertEqual(config.type, "none")

    def test_s4_positional_embedding_config(self):
        config = S4PositionalEmbeddingConfig()
        self.assertEqual(config.kernel_size, 512)

    def test_wavelet_positional_embedding_config(self):
        config = WaveletPositionalEmbeddingConfig()
        self.assertEqual(config.wavelet, "db4")

    def test_embedding_config_from_dict(self):
        config_dict = {"type": "value", "feature_size": 4}
        config = embedding_config_from_dict(config_dict)
        self.assertIsInstance(config, TimeSeriesValueEmbeddingConfig)

if __name__ == '__main__':
    unittest.main()
