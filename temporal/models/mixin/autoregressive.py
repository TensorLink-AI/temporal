# temporal/models/mixin/autoregressive_dispatch.py

import torch
from typing import Any, Optional, List

from .autoregressive_unified import AutoregressiveUnifiedMixin


class AutoregressiveDispatchMixin(AutoregressiveUnifiedMixin):
    """
    Thin shim.
    Old calls like:

        model.forecast(x, prediction_length=256, quantiles=[0.25,0.5,0.75], mode="patch_blockwise")

    are normalized and passed straight to AutoregressiveUnifiedMixin.generate(...)
    / .forecast(...).
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
        # old name → new name
        quantiles: Optional[List[float]] = kwargs.pop("quantiles", None)
        quantile_levels: Optional[List[float]] = kwargs.pop("quantile_levels", None) or quantiles

        # people do this all the time
        block_len = (
            kwargs.pop("block_len", None)
            or kwargs.pop("block_size", None)
            or kwargs.pop("chunk_len", None)
        )

        # denorm aliases
        denormalize = kwargs.pop("denormalize", kwargs.pop("denorm", False))

        return_bundle = kwargs.pop("return_bundle", False)
        return_raw = kwargs.pop("return_raw", False)

        # just call the unified one — yours expects `inputs=...`
        return super().generate(
            inputs=inputs,
            prediction_length=prediction_length,
            mode=mode,
            quantile_levels=quantile_levels,
            block_len=block_len,
            denormalize=denormalize,
            return_bundle=return_bundle,
            return_raw=return_raw,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any):
        """
        HF-style callers sometimes do:
            model.generate(x, prediction_length=...)
        or
            model.generate(inputs=x, ...)
        Normalize that and send to unified.
        """
        # positional tensor → inputs
        if len(args) > 0 and torch.is_tensor(args[0]) and "inputs" not in kwargs:
            kwargs["inputs"] = args[0]

        # old quantile name
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
        denorm = kwargs.pop("denorm", None)
        if denorm is not None and "denormalize" not in kwargs:
            kwargs["denormalize"] = bool(denorm)

        return super().generate(**kwargs)
