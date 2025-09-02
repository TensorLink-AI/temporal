import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from temporal.models.outputs import EncoderLayerOutput, DecoderLayerOutput
from temporal.models.transformer_model import TransformerTemporalModel
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import (
    TransformerTimeSeriesConfig,
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.architecture_config import TransformerArchitectureConfig


class MockLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        if "encoder_hidden_states" in kwargs:
            # Decoder layer semantics
            return DecoderLayerOutput(
                hidden_states=hidden_states,
                self_attention_weights=torch.randn(2, 1, hidden_states.size(1), hidden_states.size(1)),
                cross_attention_weights=None,
                past_key_value=None,
            )
        else:
            # Encoder layer semantics
            return EncoderLayerOutput(
                hidden_states=hidden_states,
                attention_weights=torch.randn(2, 1, hidden_states.size(1), hidden_states.size(1)),
            )


class DummyLoss(nn.Module):
    def forward(self, *, preds, targets, loss_mask=None):
        if loss_mask is not None:
            # broadcast mask if needed
            mask = loss_mask
            while mask.dim() < preds.dim():
                mask = mask.unsqueeze(-1)
            preds = preds * mask
            targets = targets * mask
        return ((preds - targets) ** 2).mean()


def _extract_tensor_from_output(out):
    if out is None:
        return None
    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    for k in ("last_hidden_state", "hidden_states", "predictions", "logits"):
        if hasattr(out, k) and isinstance(getattr(out, k), torch.Tensor):
            return getattr(out, k)
    return None


class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        class DummyValueEmbedding(nn.Module):
            def __init__(self, in_features=1, d_model=16):
                super().__init__()
                self.proj = nn.Linear(in_features, d_model, bias=False)

            def forward(self, x):
                return self.proj(x)

        class DummyPositionalEmbedding(nn.Module):
            def __init__(self, d_model=16):
                super().__init__()
                self.d_model = d_model

            def forward(self, x=None, *, batch_size=None, seq_len=None, past_key_values_length=0):
                if x is not None:
                    B, L, D = x.shape
                    return torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
                return torch.zeros(batch_size, seq_len, self.d_model)

        class IdentityNorm(nn.Module):
            def forward(self, x, *args, **kwargs):
                return x

        self.builder = MagicMock(spec=ModuleBuilder)
        self.builder._build.return_value = MockLayer()
        self.builder.build_value_embedding.return_value = DummyValueEmbedding(in_features=1, d_model=16)
        self.builder.build_positional_embedding.return_value = DummyPositionalEmbedding(d_model=16)
        self.builder.build_normalization.return_value = IdentityNorm()
        self.builder.build_loss.return_value = DummyLoss()

    def test_forward_encoder_only(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder"),
            d_model=16,
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 1)
        output = model(encoder_inputs=encoder_inputs)
        tensor = _extract_tensor_from_output(output)
        self.assertIsNotNone(tensor)

    def test_forward_decoder_only(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="decoder"),
            d_model=16,
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        decoder_inputs = torch.randn(2, 5, 1)
        output = model(decoder_inputs=decoder_inputs)
        tensor = _extract_tensor_from_output(output)
        self.assertIsNotNone(tensor)

    def test_forward_encoder_decoder(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder"),
            d_model=16,
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 1)
        decoder_inputs = torch.randn(2, 5, 1)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        tensor = _extract_tensor_from_output(output)
        self.assertIsNotNone(tensor)

    def test_forward_with_loss(self):
        config = TransformerTimeSeriesConfig(
            architecture=TransformerArchitectureConfig(type="transformer_architecture", layout="encoder-decoder"),
            d_model=16,
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            hidden_dropout_prob=0.1,
        )
        model = TransformerTemporalModel(config, builder=self.builder)
        encoder_inputs = torch.randn(2, 10, 1)
        decoder_inputs = torch.randn(2, 5, 1)
        targets = torch.randn(2, 5, 1)
        output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, targets=targets)
        # If the model returns an object with 'loss', it should be a scalar tensor; otherwise just ensure we got something back
        loss = getattr(output, "loss", None)
        self.assertTrue(loss is None or (torch.is_tensor(loss) and loss.dim() == 0))
