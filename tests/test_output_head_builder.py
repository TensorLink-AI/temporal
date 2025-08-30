import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from temporal.models.output_head_builder import OutputHeadBuilder

class TestOutputHeadBuilder(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.feature_size = 4
        self.builder = MagicMock()

    def test_build_linear_head(self):
        self.config.output_head_config.type = "linear"
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_gaussian_head(self):
        self.config.output_head_config.type = "gaussian"
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_quantile_regression_head(self):
        self.config.output_head_config.type = "quantile_regression"
        self.config.output_head_config.num_quantiles = 5
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_distpred_head(self):
        self.config.output_head_config.type = "distpred"
        self.config.output_head_config.num_outputs = 3
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_mixture_head(self):
        self.config.output_head_config.type = "mixture"
        self.config.output_head_config.components = ["normal", "student_t"]
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

if __name__ == '__main__':
    unittest.main()
