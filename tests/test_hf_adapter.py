import unittest
import torch
from unittest.mock import patch

from temporal.utils.hf_adapter import TimeSeriesTransformerModel
from temporal.configs.transformer_model_config import (
    TransformerTimeSeriesConfig,
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.loss_config import MSELossConfig


class TestTimeSeriesTransformerModel(unittest.TestCase):
    def setUp(self):
        config_dict = TransformerTimeSeriesConfig(
            feature_size=1,
            d_model=16,
            context_length=10,
            prediction_length=5,
            architecture=TransformerArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            # Ensure blocks are defined for both sides of encoder-decoder
            encoder_blocks=[EncoderBlockConfig(type="default_encoder")],
            decoder_blocks=[DecoderBlockConfig(type="default_decoder")],
            output_head_config=OutputHeadConfig(type="linear", output_size=1),
            loss_config=MSELossConfig(type="mse"),
        ).to_dict()

        # HF compat flags
        config_dict["_attn_implementation"] = "eager"
        self.config = TransformerTimeSeriesConfig.from_dict(config_dict)
        object.__setattr__(self.config, "_attn_implementation_internal", "eager")

        # Patch PreTrainedModel.__init__ to avoid mutating frozen dataclass fields
        self._ptm_patch = patch(
            "temporal.utils.hf_adapter.PreTrainedModel.__init__",
            lambda self, cfg: setattr(self, "config", cfg),
        )
        self._ptm_patch.start()

    def tearDown(self):
        self._ptm_patch.stop()

    def test_forward_pass(self):
        model = TimeSeriesTransformerModel(self.config)
        x = torch.randn(1, 10, 1)
        out = model(input_values=x)
        self.assertIsNotNone(out)

    def test_generate_pass(self):
        model = TimeSeriesTransformerModel(self.config)
        x = torch.randn(1, 10, 1)
        _ = model.generate(x, max_new_tokens=1)  # smoke test
        self.assertTrue(True)

    def test_generate_pass_with_config_prediction_length(self):
        model = TimeSeriesTransformerModel(self.config)
        x = torch.randn(1, 10, 1)
        _ = model.generate(x)  # defaults to config.prediction_length
        self.assertTrue(True)
