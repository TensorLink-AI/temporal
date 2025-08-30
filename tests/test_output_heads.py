import torch
import unittest
from temporal.modules.heads.output_heads import (
    LinearOutputHead,
    GaussianHead,
    QuantileRegressionOutputHead,
    DistPredHead,
    MixtureOutputHead,
    StudentTHead,
)

class TestOutputHeads(unittest.TestCase):

    def setUp(self):
        self.hidden_size = 16
        self.output_size = 4
        self.batch_size = 2
        self.seq_len = 10

    def test_linear_output_head(self):
        head = LinearOutputHead(self.hidden_size, self.output_size)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        y = head(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.output_size))
        y_sample = head.sample(y)
        self.assertEqual(y_sample.shape, (self.batch_size, self.seq_len, self.output_size))

    def test_gaussian_head(self):
        head = GaussianHead(self.hidden_size, self.output_size)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        y = head(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.output_size * 2))
        y_pred = head.predict(y)
        self.assertEqual(y_pred.shape, (self.batch_size, self.seq_len, self.output_size))
        y_sample = head.sample(y)
        self.assertEqual(y_sample.shape, (self.batch_size, self.seq_len, self.output_size))
        y_quantiles = head.sample_quantiles(y, [0.25, 0.5, 0.75])
        self.assertEqual(y_quantiles.shape, (self.batch_size, self.seq_len, self.output_size, 3))

    def test_quantile_regression_output_head(self):
        num_quantiles = 5
        head = QuantileRegressionOutputHead(
            self.hidden_size,
            self.output_size * num_quantiles,
            num_quantiles,
            self.output_size,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        y = head(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.output_size, num_quantiles))
        y_pred = head.predict(y)
        self.assertEqual(y_pred.shape, (self.batch_size, self.seq_len, self.output_size))
        y_sample = head.sample(y)
        self.assertEqual(y_sample.shape, (self.batch_size, self.seq_len, self.output_size))

    def test_dist_pred_head(self):
        num_outputs = 3
        head = DistPredHead(
            self.hidden_size,
            self.output_size * num_outputs,
            num_outputs=num_outputs,
            feature_size=self.output_size,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        y = head(x)
        self.assertEqual(y["paths"].shape, (self.batch_size, self.seq_len, self.output_size, num_outputs))
        y_pred = head.predict(y)
        self.assertEqual(y_pred.shape, (self.batch_size, self.seq_len, self.output_size))
        y_quantiles = head.sample_quantiles(y, [0.25, 0.5, 0.75])
        self.assertEqual(y_quantiles.shape, (self.batch_size, self.seq_len, self.output_size, 3))

    def test_mixture_output_head(self):
        components = ["normal", "student_t"]
        head = MixtureOutputHead(self.hidden_size, components)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        y = head(x)
        self.assertIn("mixture_logits", y)
        self.assertEqual(y["mixture_logits"].shape, (self.batch_size, self.seq_len, len(components)))
        y_pred = head.predict(y)
        self.assertEqual(y_pred.shape, (self.batch_size, self.seq_len, 1))
        y_sample = head.sample(y)
        self.assertEqual(y_sample.shape, (self.batch_size, 1, 1))
        y_quantiles = head.sample_quantiles(y, [0.25, 0.5, 0.75])
        self.assertEqual(y_quantiles.shape, (self.batch_size, self.seq_len, 1, 3))

    def test_student_t_head(self):
        head = StudentTHead(self.hidden_size, self.output_size)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        y = head(x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.output_size * 3))
        y_pred = head.predict(y)
        self.assertEqual(y_pred.shape, (self.batch_size, self.seq_len, self.output_size))
        y_sample = head.sample(y)
        self.assertEqual(y_sample.shape, (self.batch_size, self.seq_len, self.output_size))
        y_quantiles = head.sample_quantiles(y, [0.25, 0.5, 0.75])
        self.assertEqual(y_quantiles.shape, (self.batch_size, self.seq_len, self.output_size, 3))

if __name__ == "__main__":
    unittest.main()
