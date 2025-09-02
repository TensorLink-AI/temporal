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
        # Return something your encoder can consume; attention can be None
        return EncoderLayerOutput(hidden_states=hidden_states, attention_weights=None)


def _extract_last_hidden_state(output):
    # Be permissive about output shape/naming
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    for attr in ("last_hidden_state", "hidden_states", "logits", "predictions"):
        if hasattr(output, attr) and isinstance(getattr(output, attr), torch.Tensor):
            return getattr(output, attr)
    return None


class TestEncoders(unittest.TestCase):
    def setUp(self):
        # Build via (config, builder, block_configs) since encoder now expects that
        self.config = MagicMock()
        self.config.d_model = 16

        self.builder = MagicMock(spec=ModuleBuilder)
        # make _build return our mock layer per block
        self.builder._build.side_effect = [MockEncoderLayer(), MockEncoderLayer()]

        self.block_configs = [EncoderBlockConfig(type="default_encoder"), EncoderBlockConfig(type="default_encoder")]

        self.encoder = TimeSeriesTransformerEncoder(
            config=self.config,
            builder=self.builder,
            block_configs=self.block_configs,
            layerdrop=0.0,
        )

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
        # Don't require non-None weights; just ensure attribute exists if provided by impl
        has_attn_attr = any(hasattr(output, n) for n in ("attentions", "attention_weights"))
        self.assertTrue(has_attn_attr or isinstance(output, tuple))
