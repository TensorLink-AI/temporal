# temporal/utils/hf_adapter.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.models.builder import build_time_series_transformer


class TimeSeriesTransformerModel(PreTrainedModel):
    """
    Lightweight Hugging Face-compatible wrapper that delegates to the internal
    Temporal model while providing HF-ish argument aliases.

    Key behaviors:
      - Accepts `input_values` (HF audio-style) and maps it to the Temporal
        model's `encoder_inputs`.
      - In `generate`, accepts a positional first tensor and routes it to
        `encoder_inputs` (or `decoder_inputs` if the layout is decoder-only).
      - Maps `max_new_tokens` -> `prediction_length`.
    """
    config_class = TransformerTimeSeriesConfig

    def __init__(self, config: TransformerTimeSeriesConfig):
        # Let PreTrainedModel handle its usual setup
        super().__init__(config)
        # Build the actual Temporal model once we have a valid config
        self.temporal = build_time_series_transformer(config)

    # -------------------------
    # Helpers
    # -------------------------
    def _layout(self) -> str:
        arch = getattr(self.config, "architecture", None)
        return getattr(arch, "layout", "encoder-decoder")

    @staticmethod
    def _pop_alias(kwargs: Dict[str, Any], src: str, dst: str) -> None:
        if src in kwargs and dst not in kwargs:
            kwargs[dst] = kwargs.pop(src)

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        *args,
        **kwargs,
    ):
        """
        HF-style forward that accepts `input_values` and routes it to
        the Temporal model's `encoder_inputs`.
        Also tolerates a positional first tensor and assigns it to the most
        sensible input based on the architecture layout.
        """
        # If a positional tensor was passed, map it to encoder/decoder inputs.
        if args:
            first = args[0]
            if (
                isinstance(first, torch.Tensor)
                and "encoder_inputs" not in kwargs
                and "decoder_inputs" not in kwargs
                and "input_values" not in kwargs
            ):
                if self._layout() in ("encoder", "encoder-decoder"):
                    kwargs["encoder_inputs"] = first
                else:
                    kwargs["decoder_inputs"] = first
                args = args[1:]

        # Map HF alias -> Temporal name
        self._pop_alias(kwargs, "input_values", "encoder_inputs")

        # Pass everything through to the Temporal model
        return self.temporal(*args, **kwargs)

    # -------------------------
    # Generate
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        *args,
        **kwargs,
    ):
        """
        HF-style generate that:
          - Accepts a positional first tensor and routes it to encoder_inputs
            (or decoder_inputs for decoder-only).
          - Accepts `input_values` and routes to `encoder_inputs`.
          - Maps `max_new_tokens` -> `prediction_length`.
        """
        # Positional first argument: assume it's the input tensor
        if args:
            first = args[0]
            if (
                isinstance(first, torch.Tensor)
                and "encoder_inputs" not in kwargs
                and "decoder_inputs" not in kwargs
                and "input_values" not in kwargs
            ):
                if self._layout() in ("encoder", "encoder-decoder"):
                    kwargs["encoder_inputs"] = first
                else:
                    kwargs["decoder_inputs"] = first
                args = args[1:]

        # Alias mappings
        self._pop_alias(kwargs, "input_values", "encoder_inputs")
        if "max_new_tokens" in kwargs and "prediction_length" not in kwargs:
            kwargs["prediction_length"] = kwargs.pop("max_new_tokens")

        return self.temporal.generate(*args, **kwargs)
