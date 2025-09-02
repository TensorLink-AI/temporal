import unittest
import torch
from temporal.modules.embedders.embedding import (
    TimeSeriesValueEmbedding,
    SinusoidalPositionalEmbedding,
    RotaryPositionalEmbedding,
    TimeSeriesPatchEmbedding,
    LearnedAbsolutePositionalEmbedding,
    ConvolutionalPositionalEmbedding,
)

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.feature_size = 4
        self.d_model = 16
        self.max_seq_len = 20
        self.patch_size = 2
        self.stride = 2

        self.input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.feature_size
        )

    def test_time_series_value_embedding(self):
        embedding = TimeSeriesValueEmbedding(self.feature_size, self.d_model)
        output = embedding(self.input_tensor)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.d_model)
        )

    def test_sinusoidal_positional_embedding_additive(self):
        embedding = SinusoidalPositionalEmbedding(self.d_model, self.max_seq_len)
        input_for_pe = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = embedding(input_for_pe)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.d_model)
        )

    def test_sinusoidal_positional_embedding_pe_only(self):
        embedding = SinusoidalPositionalEmbedding(self.d_model, self.max_seq_len)
        output = embedding(batch_size=self.batch_size, seq_len=self.seq_len)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.d_model)
        )

    def test_rotary_positional_embedding(self):
        embedding = RotaryPositionalEmbedding(self.d_model, self.max_seq_len)
        cos, sin = embedding(x=torch.randn(self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(cos.shape, (self.seq_len, self.d_model))
        self.assertEqual(sin.shape, (self.seq_len, self.d_model))

    def test_time_series_patch_embedding(self):
        embedding = TimeSeriesPatchEmbedding(
            self.patch_size, self.feature_size, self.d_model, self.stride
        )
        output = embedding(self.input_tensor)
        num_patches = (self.seq_len - self.patch_size) // self.stride + 1
        self.assertEqual(output.shape, (self.batch_size, num_patches, self.d_model))

    def test_learned_absolute_positional_embedding(self):
        embedding = LearnedAbsolutePositionalEmbedding(self.d_model, self.max_seq_len)
        input_for_pe = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = embedding(input_for_pe)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.d_model)
        )

    def test_convolutional_positional_embedding(self):
        embedding = ConvolutionalPositionalEmbedding(self.d_model)
        # The forward method expects batch_size and seq_len, not a tensor x.
        y = embedding(batch_size=self.batch_size, seq_len=self.seq_len)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

if __name__ == "__main__":
    unittest.main()