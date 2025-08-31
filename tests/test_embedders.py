import torch
import unittest
from temporal.modules.embedders.embedding import (
    TimeSeriesValueEmbedding,
    FlexibleValueEmbedding,
    SinusoidalPositionalEmbedding,
    TimeSeriesPatchEmbedding,
    RotaryPositionalEmbedding,
    TimeSeriesGlobalEmbedding,
    LearnedAbsolutePositionalEmbedding,
    ShawRelativePositionalBias,
    FourierFeatureEmbedding,
    Time2VecEmbedding,
    ALiBiPositionalBias,
    BucketedRelativeBias,
    ConvolutionalPositionalEmbedding,
    TimeDeltaEmbedding,
    NoneEmbedding,
    S4PositionalEmbedding,
    WaveletPositionalEmbedding,
)

class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.d_model = 16
        self.feature_size = 4
        self.seq_len = 10
        self.batch_size = 2
        self.max_seq_len = 20

    def test_time_series_value_embedding(self):
        embedding = TimeSeriesValueEmbedding(self.feature_size, self.d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_flexible_value_embedding(self):
        embedding = FlexibleValueEmbedding(d_model=self.d_model, input_dims=self.feature_size)
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_sinusoidal_positional_embedding(self):
        embedding = SinusoidalPositionalEmbedding(self.d_model, self.max_seq_len)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # FIX: Call with the tensor x, not batch_size and seq_len separately
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_time_series_patch_embedding(self):
        embedding = TimeSeriesPatchEmbedding(patch_size=5, feature_size=self.feature_size, d_model=self.d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, 2, self.d_model))

    def test_rotary_positional_embedding(self):
        embedding = RotaryPositionalEmbedding(self.d_model, self.max_seq_len)
        x = torch.randn(self.batch_size, 4, self.seq_len, self.d_model)
        cos, sin = embedding(x)
        self.assertEqual(cos.shape, (self.seq_len, self.d_model))
        self.assertEqual(sin.shape, (self.seq_len, self.d_model))

    def test_time_series_global_embedding(self):
        embedding = TimeSeriesGlobalEmbedding(self.seq_len, self.feature_size, self.d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, 1, self.d_model))

    def test_learned_absolute_positional_embedding(self):
        embedding = LearnedAbsolutePositionalEmbedding(self.d_model, self.max_seq_len)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_shaw_relative_positional_bias(self):
        embedding = ShawRelativePositionalBias(num_heads=4)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (1, 4, self.seq_len, self.seq_len))

    def test_fourier_feature_embedding(self):
        embedding = FourierFeatureEmbedding(self.d_model)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_time2vec_embedding(self):
        embedding = Time2VecEmbedding(self.d_model)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_alibi_positional_bias(self):
        embedding = ALiBiPositionalBias(num_heads=4)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (1, 4, self.seq_len, self.seq_len))

    def test_bucketed_relative_bias(self):
        embedding = BucketedRelativeBias(num_heads=4)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (1, 4, self.seq_len, self.seq_len))

    def test_convolutional_positional_embedding(self):
        embedding = ConvolutionalPositionalEmbedding(self.d_model)
        # FIX: The forward pass now takes a tensor x as input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y = embedding(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_time_delta_embedding(self):
        embedding = TimeDeltaEmbedding(self.d_model)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_none_embedding(self):
        embedding = NoneEmbedding(self.d_model)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertTrue(torch.all(y == 0))

    def test_s4_positional_embedding(self):
        embedding = S4PositionalEmbedding(self.d_model)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_wavelet_positional_embedding(self):
        embedding = WaveletPositionalEmbedding(self.d_model)
        y = embedding(self.batch_size, self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

if __name__ == '__main__':
    unittest.main()