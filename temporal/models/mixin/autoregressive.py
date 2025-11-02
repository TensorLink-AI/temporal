# temporal/models/mixin/autoregressive_dispatch.py
import torch
from typing import Any, Optional, List

from .autoregressive_unified import AutoregressiveUnifiedMixin


class AutoregressiveDispatchMixin(AutoregressiveUnifiedMixin):
    """
    Thin adapter over your AutoregressiveUnifiedMixin.

    - user calls: model.forecast(x, ..., mode="patch_blockwise")
    - we:
        * rename quantiles -> quantile_levels
        * map inputs -> encoder_inputs / decoder_inputs
        * forward to unified.generate(...)
    """

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        *,
        mode: Optional[str] = None,
        **kwargs: Any,
    ):
        # 1) normalize quantiles name
        quantiles: Optional[List[float]] = kwargs.pop("quantiles", None)
        if quantiles is not None and "quantile_levels" not in kwargs:
            kwargs["quantile_levels"] = quantiles

        # 2) normalize block len aliases
        block_len = (
            kwargs.pop("block_len", None)
            or kwargs.pop("block_size", None)
            or kwargs.pop("chunk_len", None)
        )
        if block_len is not None:
            kwargs["block_len"] = block_len

        # 3) normalize denorm
        if "denormalize" not in kwargs and "denorm" in kwargs:
            kwargs["denormalize"] = bool(kwargs.pop("denorm"))

        return_bundle = kwargs.pop("return_bundle", False)
        return_raw = kwargs.pop("return_raw", False)

        # 4) map inputs -> encoder_inputs / decoder_inputs
        if hasattr(self, "encoder") and self.encoder is not None:
            kwargs["encoder_inputs"] = inputs
        else:
            kwargs["decoder_inputs"] = inputs

        # 5) call your unified mixin (IMPORTANT: no `inputs=` here)
        return super().generate(
            prediction_length=prediction_length,
            mode=mode,
            return_bundle=return_bundle,
            return_raw=return_raw,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any):
        """
        HF-style generate:
            model.generate(x, prediction_length=...)
        or:
            model.generate(encoder_inputs=x, ...)
        We fix the names, then call unified.generate(...)
        """
        # positional tensor → "inputs"
        if len(args) > 0 and torch.is_tensor(args[0]) and "encoder_inputs" not in kwargs and "decoder_inputs" not in kwargs:
            # decide enc vs dec
            if hasattr(self, "encoder") and self.encoder is not None:
                kwargs["encoder_inputs"] = args[0]
            else:
                kwargs["decoder_inputs"] = args[0]

        # quantiles → quantile_levels
        quantiles = kwargs.pop("quantiles", None)
        if quantiles is not None and "quantile_levels" not in kwargs:
            kwargs["quantile_levels"] = quantiles

        # block len aliases
        block_len = (
            kwargs.pop("block_len", None)
            or kwargs.pop("block_size", None)
            or kwargs.pop("chunk_len", None)
        )
        if block_len is not None:
            kwargs["block_len"] = block_len

        # denorm alias
        if "denormalize" not in kwargs and "denorm" in kwargs:
            kwargs["denormalize"] = bool(kwargs.pop("denorm"))

        # make sure we have prediction_length
        if "prediction_length" not in kwargs:
            pl = getattr(self.config, "prediction_length", None)
            if pl is None:
                raise ValueError("generate(...) needs prediction_length")
            kwargs["prediction_length"] = pl

        return super().generate(**kwargs)
