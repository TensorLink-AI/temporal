import torch.nn as nn
import unittest
from unittest.mock import MagicMock, PropertyMock
from temporal.models.output_head_builder import OutputHeadBuilder
from temporal.configs.output_head_config import (
    OutputHeadConfig,
    GaussianOutputHeadConfig,
    QuantileRegressionOutputHeadConfig,
    DistPredOutputHeadConfig,
    MixtureOutputHeadConfig,
)
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig

class TestOutputHeadBuilder(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.feature_size = 4
        self.config.value_embedding_config = MagicMock(spec=TimeSeriesValueEmbeddingConfig, kwargs={})
        self.builder = MagicMock()

    def test_build_linear_head(self):
        self.config.output_head_config = OutputHeadConfig(type="linear", output_size=1)
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_gaussian_head(self):
        self.config.output_head_config = GaussianOutputHeadConfig(type="gaussian", output_size=1)
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_quantile_regression_head(self):
        self.config.output_head_config = QuantileRegressionOutputHeadConfig(
            type="quantile_regression", num_quantiles=5, feature_size=self.config.feature_size, output_size=20
        )
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_distpred_head(self):
        self.config.output_head_config = DistPredOutputHeadConfig(
            type="distpred", output_size=1, num_outputs=3, feature_size=self.config.feature_size
        )
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_mixture_head(self):
        self.builder.build_output_head.side_effect = [MagicMock(), MagicMock()]
        self.config.output_head_config = MixtureOutputHeadConfig(
            type="mixture",
            output_size=1,
            components=["gaussian", "gaussian"],
        )
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

if __name__ == "__main__":
    unittest.main()
