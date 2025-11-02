import logging
from typing import Optional, List, Union, Dict, Any

import torch
import torch.nn as nn

# unified mixin – the one that has _generate_patch_one_shot / _generate_patch_blockwise / _generate_blockwise
from .autoregressive_unified import AutoregressiveUnifiedMixin

# optional legacy fallbacks
try:
    from .autoregressive_stepwise import AutoregressiveStepwiseMixin
except Exception:  # pragma: no cover
    AutoregressiveStepwiseMixin = None  # type: ignore

try:
    from .autoregressive_blockwise import AutoregressiveBlockwiseMixin
except Exception:  # pragma: no cover
    AutoregressiveBlockwiseMixin = None  # type: ignore

logger = logging.getLogger(__name__)


class AutoregressiveDispatchMixin(AutoregressiveUnifiedMixin):
    """
    Dispatch layer that normalizes user-facing args and then calls
    the unified autoregressive mixin.

    Supports:
      - model.forecast(..., quantiles=[...])
      - model.forecast(..., blockwise=True)
      - model.forecast(..., mode="patch_blockwise")
      - model.generate(..., mode="block")
      - auto-detect patch vs non-patch
    """

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _is_patch_model(self) -> bool:
        return (
            hasattr(self, "preprocessor")
            and hasattr(self.preprocessor, "value_embedding")
            and hasattr(self.preprocessor.value_embedding, "patch_size")
        )

    @staticmethod
    def _pop_quantiles(kwargs: Dict[str, Any]) -> Optional[List[float]]:
        """
        Many call sites still use `quantiles=...`. The unified mixin
        uses `quantile_levels=...`. Normalize here.
        """
        q = kwargs.pop("quantiles", None)
        if q is None:
            return None
        # ensure list[float]
        return [float(x) for x in q]

    # ------------------------------------------------------------
    # public forecast
    # ------------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        *,
        mode: Optional[str] = None,
        blockwise: bool = False,
        quantiles: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Normalizes:
          - quantiles -> quantile_levels
          - blockwise -> mode
        then calls the unified implementation.
        """

        # 1) collect/normalize quantiles
        # (explicit arg wins over kw)
        kw_quantiles = self._pop_quantiles(kwargs)
        if quantiles is None:
            quantiles = kw_quantiles
        elif kw_quantiles is not None:
            # user passed in both styles – prefer explicit arg
            pass

        # 2) normalize blockwise -> mode
        if mode is None and blockwise:
            if self._is_patch_model():
                mode = "patch_blockwise"
            else:
                mode = "block"

        # 3) auto mode if still None
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

        logger.info(f"[AutoregressiveDispatchMixin] forecast(): mode={mode}, patch={self._is_patch_model()}")

        # 4) try unified path first
        try:
            return super().forecast(
                inputs=inputs,
                prediction_length=prediction_length,
                mode=mode,
                quantile_levels=quantiles,
                **kwargs,
            )
        except NotImplementedError:
            logger.warning(
                f"[AutoregressiveDispatchMixin] unified forecast() could not handle mode='{mode}', "
                f"falling back to legacy mixins (if available)."
            )

        # 5) legacy fallbacks
        if mode in ("block", "blockwise") and AutoregressiveBlockwiseMixin is not None:
            return AutoregressiveBlockwiseMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                **kwargs,
            )

        if mode in ("step", "stepwise") and AutoregressiveStepwiseMixin is not None:
            return AutoregressiveStepwiseMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                **kwargs,
            )

        raise NotImplementedError(f"forecast mode='{mode}' not supported on this model.")

    # ------------------------------------------------------------
    # public generate
    # ------------------------------------------------------------
    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Same idea as forecast(): normalize args, then call unified.
        """
        mode = kwargs.pop("mode", None)
        blockwise = kwargs.pop("blockwise", False)

        # normalize quantiles -> quantile_levels for generate too
        quantiles = self._pop_quantiles(kwargs)
        if quantiles is not None:
            kwargs["quantile_levels"] = quantiles

        if mode is None and blockwise:
            if self._is_patch_model():
                mode = "patch_blockwise"
            else:
                mode = "block"

        if mode is not None:
            kwargs["mode"] = mode

        logger.info(f"[AutoregressiveDispatchMixin] generate(): mode={mode}, patch={self._is_patch_model()}")

        # unified first
        try:
            return super().generate(*args, **kwargs)
        except NotImplementedError:
            logger.warning(
                f"[AutoregressiveDispatchMixin] unified generate() could not handle mode='{mode}', "
                f"falling back to legacy mixins (if available)."
            )

        # legacy
        if mode in ("block", "blockwise") and AutoregressiveBlockwiseMixin is not None:
            return AutoregressiveBlockwiseMixin.generate(self, *args, **kwargs)

        if mode in ("step", "stepwise") and AutoregressiveStepwiseMixin is not None:
            return AutoregressiveStepwiseMixin.generate(self, *args, **kwargs)

        raise NotImplementedError(f"generate mode='{mode}' not supported on this model.")
