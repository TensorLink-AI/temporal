import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from temporal.models.module_builder_helper import ModuleBuilder

class TestModuleBuilderHelper(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.feature_size = 4
        self.builder = ModuleBuilder(self.config)

    def test_build_attention(self):
        self.config.attention_config.type = "full"
        attention = self.builder.build_attention(self.config.attention_config)
        self.assertIsInstance(attention, nn.Module)

    def test_build_feedforward(self):
        self.config.feedforward_config.type = "standard"
        feedforward = self.builder.build_feedforward(self.config.feedforward_config)
        self.assertIsInstance(feedforward, nn.Module)

    def test_build_value_embedding(self):
        self.config.value_embedding_config.type = "value"
        embedding = self.builder.build_value_embedding(self.config.value_embedding_config)
        self.assertIsInstance(embedding, nn.Module)

    def test_build_positional_embedding(self):
        self.config.positional_embedding_config.type = "sinusoidal"
        embedding = self.builder.build_positional_embedding(self.config.positional_embedding_config)
        self.assertIsInstance(embedding, nn.Module)

    def test_build_normalization(self):
        self.config.normalization_config.type = "layer_norm"
        normalization = self.builder.build_normalization(self.config.normalization_config)
        self.assertIsInstance(normalization, nn.Module)

    def test_build_loss(self):
        self.config.loss_config.type = "mae"
        loss = self.builder.build_loss(self.config.loss_config)
        self.assertIsInstance(loss, nn.Module)

if __name__ == '__main__':
    unittest.main()
