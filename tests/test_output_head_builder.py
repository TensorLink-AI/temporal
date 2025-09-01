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
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig # Corrected Import

class TestOutputHeadBuilder(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.feature_size = 4
        # Add a mock for value_embedding_config with a kwargs attribute
        self.config.value_embedding_config = MagicMock(spec=TimeSeriesValueEmbeddingConfig, kwargs={}) # Corrected spec
        self.builder = MagicMock()

    def test_build_linear_head(self):
        # Use OutputHeadConfig with type="linear" as it's the default and there's no LinearOutputHeadConfig dataclass
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
            type="quantile_regression", num_quantiles=5, output_size=5
        )
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_distpred_head(self):
        self.config.output_head_config = DistPredOutputHeadConfig(
            type="distpred", output_size=1, num_outputs=3, dist_family="Gaussian"
        )
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)

    def test_build_mixture_head(self):
        # Mock the sub-head builder calls
        self.builder.build_output_head.side_effect = [MagicMock(), MagicMock()]
        self.config.output_head_config = MixtureOutputHeadConfig(
            type="mixture",
            output_size=1,
            components=[
                {"type": "gaussian", "output_size": 1},
                {"type": "gaussian", "output_size": 1},
            ],
        )
        builder = OutputHeadBuilder(self.config, self.builder)
        head = builder.build()
        self.assertIsInstance(head, nn.Module)


if __name__ == "__main__":
    unittest.main()
