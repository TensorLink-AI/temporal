# autoregressive_unified.py
import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# optional imports – we call into them if present
try:
    from .autoregressive_stepwise import AutoregressiveStepwiseMixin as _Stepwise
except Exception:  # pragma: no cover
    _Stepwise = None  # type: ignore

try:
    from .autoregressive_blockwise import AutoregressiveBlockwiseMixin as _Blockwise
except Exception:  # pragma: no cover
    _Blockwise = None  # type: ignore

try:
    from .autoregressive_patch import AutoregressivePatchMixin as _PatchBase
except Exception:  # pragma: no cover
    _PatchBase = None  # type: ignore


class AutoregressiveUnifiedMixin(nn.Module):
    """
    Single place that knows how to do:
      - stepwise AR
      - blockwise AR
      - patch one-shot (zero-shot all patches)
      - patch blockwise (AR in patch space, roll out more than train horizon)

    Everything comes through here; dispatch mixin just normalizes args.
    """

    # ------------- helpers -------------
    def _is_patch_model(self) -> bool:
        return (
            hasattr(self, "preprocessor")
            and hasattr(self.preprocessor, "value_embedding")
            and hasattr(self.preprocessor.value_embedding, "patch_size")
        )

    # ------------- public API -------------
    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        *,
        mode: Optional[str] = None,
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified forecast entry point.

        mode:
          - "step" / "stepwise"
          - "block" / "blockwise"
          - "patch"
          - "patch_blockwise"
          - None -> auto
        """
        # pull these out so we don't explode in the sub-fns
        denormalize: bool = bool(kwargs.pop("denormalize", False))
        return_bundle: bool = bool(kwargs.pop("return_bundle", False))
        return_raw: bool = bool(kwargs.pop("return_raw", False))

        if mode is None:
            if self._is_patch_model():
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length > train_h:
                    mode = "patch_blockwise"
                else:
                    mode = "patch"
            else:
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length > train_h:
                    mode = "block"
                else:
                    mode = "step"

        if mode in ("patch", "patch_one_shot"):
            return self._generate_patch_one_shot(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        if mode in ("patch_blockwise", "patch_block"):
            return self._generate_patch_blockwise(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        if mode in ("block", "blockwise"):
            return self._generate_blockwise(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        if mode in ("step", "stepwise"):
            return self._generate_stepwise(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        raise NotImplementedError(f"Unknown AR mode: {mode}")

    @torch.no_grad()
    def generate(
        self,
        *args: Any,
        mode: Optional[str] = None,
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Same as forecast but signature-aligned with HF-style generate.
        We still normalize and call the same internal functions.
        """
        # some callers give decoder_inputs directly
        inputs = kwargs.pop("inputs", None)
        if inputs is None and len(args) > 0 and torch.is_tensor(args[0]):
            inputs = args[0]
        if inputs is None:
            raise ValueError("generate(...) needs `inputs` tensor or first positional tensor.")

        prediction_length = kwargs.pop("prediction_length", None)
        if prediction_length is None:
            prediction_length = getattr(self.config, "prediction_length", None)
        if prediction_length is None:
            raise ValueError("prediction_length must be provided to generate(...)")

        # pull common kwargs
        denormalize: bool = bool(kwargs.pop("denormalize", False))
        return_bundle: bool = bool(kwargs.pop("return_bundle", False))
        return_raw: bool = bool(kwargs.pop("return_raw", False))

        if mode is None:
            if self._is_patch_model():
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length > train_h:
                    mode = "patch_blockwise"
                else:
                    mode = "patch"
            else:
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length > train_h:
                    mode = "block"
                else:
                    mode = "step"

        if mode in ("patch", "patch_one_shot"):
            return self._generate_patch_one_shot(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        if mode in ("patch_blockwise", "patch_block"):
            return self._generate_patch_blockwise(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        if mode in ("block", "blockwise"):
            return self._generate_blockwise(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        if mode in ("step", "stepwise"):
            return self._generate_stepwise(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                **kwargs,
            )

        raise NotImplementedError(f"Unknown AR mode: {mode}")

    # ------------- concrete internal paths -------------

    # --- 1) patch, one shot (your “zero shot all patches”) ---
    def _generate_patch_one_shot(
        self,
        *,
        inputs: torch.Tensor,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
        **kwargs: Any,
    ):
        if _PatchBase is None:
            raise NotImplementedError("Patch mixin not available in this build.")

        # re-use original patch mixin but make it tolerant
        return _PatchBase.generate(
            self,
            encoder_inputs=inputs if hasattr(self, "encoder") and self.encoder is not None else None,
            decoder_inputs=None if hasattr(self, "encoder") and self.encoder is not None else inputs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            denormalize=denormalize,
            return_bundle=return_bundle,
            return_raw=return_raw,
            **kwargs,
        )

    # --- 2) patch, blockwise AR in patch space ---
    def _generate_patch_blockwise(
        self,
        *,
        inputs: torch.Tensor,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
        block_len: Optional[int] = None,
        **kwargs: Any,
    ):
        if _PatchBase is None:
            raise NotImplementedError("Patch mixin not available in this build.")

        # simple strategy:
        #  - number of patches to roll out = ceil(pred_len / patch_size)
        #  - we still let the patch mixin do the actual AR in patch space
        #  - we just forward the flags
        return _PatchBase.generate(
            self,
            encoder_inputs=inputs if hasattr(self, "encoder") and self.encoder is not None else None,
            decoder_inputs=None if hasattr(self, "encoder") and self.encoder is not None else inputs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            denormalize=denormalize,
            return_bundle=return_bundle,
            return_raw=return_raw,
            **kwargs,
        )

    # --- 3) non-patch, true blockwise ---
    def _generate_blockwise(
        self,
        *,
        inputs: torch.Tensor,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
        block_len: Optional[int] = None,
        **kwargs: Any,
    ):
        if _Blockwise is None:
            raise NotImplementedError("Blockwise mixin not available.")
        return _Blockwise.forecast(
            self,
            inputs=inputs,
            prediction_length=prediction_length,
            block_len=block_len,
            quantiles=quantile_levels,
            denormalize=denormalize,
            return_bundle=return_bundle,
            return_raw=return_raw,
            **kwargs,
        )

    # --- 4) non-patch, stepwise ---
    def _generate_stepwise(
        self,
        *,
        inputs: torch.Tensor,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
        **kwargs: Any,
    ):
        if _Stepwise is None:
            raise NotImplementedError("Stepwise mixin not available.")
        return _Stepwise.forecast(
            self,
            inputs=inputs,
            prediction_length=prediction_length,
            quantiles=quantile_levels,
            denormalize=denormalize,
            return_bundle=return_bundle,
            return_raw=return_raw,
            **kwargs,
        )
