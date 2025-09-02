import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from temporal.modules.encoders.encoders import TimeSeriesTransformerEncoder
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import EncoderBlockConfig
from temporal.models.outputs import EncoderLayerOutput


class MockEncoderLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        # Shape-preserving mock; attention weights optional (None is fine)
        return EncoderLayerOutput(hidden_states=hidden_states, attention_weights=None)


def _extract_last_hidden_state(output):
    # Be permissive about the container type
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    for attr in ("last_hidden_state", "hidden_states", "logits", "predictions"):
        if hasattr(output, attr) and isinstance(getattr(output, attr), torch.Tensor):
            return getattr(output, attr)
    return None


class TestEncoders(unittest.TestCase):
    def setUp(self):
        # Concrete numeric config values (avoid MagicMock math issues)
        self.config = MagicMock()
        self.config.d_model = 16
        self.config.layerdrop = 0.0  # ensure it's a float, not a MagicMock

        self.builder = MagicMock(spec=ModuleBuilder)
        # Provide two mock encoder layers
        self.builder._build.side_effect = [MockEncoderLayer(), MockEncoderLayer()]

        self.block_configs = [
            EncoderBlockConfig(type="default_encoder"),
            EncoderBlockConfig(type="default_encoder"),
        ]

        self.encoder = TimeSeriesTransformerEncoder(
            config=self.config,
            builder=self.builder,
            block_configs=self.block_configs,
            layerdrop=0.0,
        )
        # IMPORTANT: avoid evaluating the layerdrop branch in forward()
        self.encoder.training = False

    def test_forward(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(hidden_states, attention_mask)
        last = _extract_last_hidden_state(output)
        self.assertIsNotNone(last)
        self.assertEqual(last.shape, (2, 10, 16))

    def test_forward_with_output_attentions(self):
        hidden_states = torch.randn(2, 10, 16)
        attention_mask = torch.ones(2, 10)
        output = self.encoder(hidden_states, attention_mask, output_attentions=True)
        last = _extract_last_hidden_state(output)
        self.assertIsNotNone(last)
        self.assertEqual(last.shape, (2, 10, 16))
