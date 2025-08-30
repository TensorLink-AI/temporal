import torch.nn as nn
import unittest
from unittest.mock import MagicMock, PropertyMock
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.attention_config import FullAttentionConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig, SinusoidalPositionalEmbeddingConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.configs.loss_config import TimeSeriesLossConfig

class TestModuleBuilderHelper(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.feature_size = 4
        self.builder = ModuleBuilder(self.config)

    def test_build_attention(self):
        # FIX: Use a real config object so builder can pull attributes
        attn_config = FullAttentionConfig(type="full", num_heads=4)
        attention = self.builder.build_attention(attn_config)
        self.assertIsInstance(attention, nn.Module)

    def test_build_feedforward(self):
        # FIX: Use a real config object
        ffn_config = StandardFeedForwardConfig(type="standard", intermediate_size=32)
        feedforward = self.builder.build_feedforward(ffn_config)
        self.assertIsInstance(feedforward, nn.Module)

    def test_build_value_embedding(self):
        # FIX: Use a real config object
        emb_config = TimeSeriesValueEmbeddingConfig(type="value", feature_size=4, d_model=16)
        embedding = self.builder.build_value_embedding(emb_config)
        self.assertIsInstance(embedding, nn.Module)

    def test_build_positional_embedding(self):
        # FIX: Use a real config object
        pos_config = SinusoidalPositionalEmbeddingConfig(type="sinusoidal", d_model=16)
        embedding = self.builder.build_positional_embedding(pos_config)
        self.assertIsInstance(embedding, nn.Module)

    def test_build_normalization(self):
        # FIX: Use a real config and a registered name ("layer")
        norm_config = NormalizationConfig(type="layer")
        normalization = self.builder.build_normalization(norm_config)
        self.assertIsInstance(normalization, nn.Module)

    def test_build_loss(self):
        # FIX: Use a real config and a registered name ("timeseries_generic")
        loss_config = TimeSeriesLossConfig(type="timeseries_generic", loss_type="mae")
        loss = self.builder.build_loss(loss_config)
        self.assertIsInstance(loss, nn.Module)

if __name__ == '__main__':
    unittest.main()