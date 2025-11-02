import logging
from typing import Optional, List, Union, Dict, Any

import torch
import torch.nn as nn

# unified version (the one we just designed)
from .autoregressive_unified import AutoregressiveUnifiedMixin

# optional legacy mixins – only used as fallback if present
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
    Thin, backward-compatible dispatcher.

    - New style:
        model.forecast(x, prediction_length=256, mode="patch_blockwise")
        model.generate(..., mode="patch")
    - Old style:
        model.forecast(x, prediction_length=96, blockwise=True)
        model.generate(..., blockwise=True)

    The *real* logic lives in AutoregressiveUnifiedMixin.
    This class just:
      1) normalizes arguments (mode vs blockwise)
      2) auto-detects patch models
      3) tries legacy mixins if the unified one doesn't handle it
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _is_patch_model(self) -> bool:
        return (
            hasattr(self, "preprocessor")
            and hasattr(self.preprocessor, "value_embedding")
            and hasattr(self.preprocessor.value_embedding, "patch_size")
        )

    # ------------------------------------------------------------------
    # public forecast
    # ------------------------------------------------------------------
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
        Entry point most users call.

        Args
        ----
        inputs: [B, T, F]
        prediction_length: int
        mode: one of
            - "patch"             (zero-shot patches)
            - "patch_blockwise"   (AR over native time but using patch model)
            - "block" / "blockwise" (non-patch block AR)
            - "step"              (classic 1-step AR, if present)
            - None -> auto
        blockwise: bool
            old API shortcut; converts to mode="patch_blockwise" for patch models
            or mode="block" for non-patch models.
        """
        # 1) normalize: old arg -> new mode
        if blockwise and mode is None:
            if self._is_patch_model():
                mode = "patch_blockwise"
            else:
                mode = "block"

        # 2) if still no mode -> auto
        if mode is None:
            if self._is_patch_model():
                # decide 1-shot vs blockwise by horizon
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length > train_h:
                    mode = "patch_blockwise"
                else:
                    mode = "patch"
            else:
                # non-patch: try block if user asks long horizon, else step
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length > train_h:
                    mode = "block"
                else:
                    mode = "step"

        logger.info(f"[AutoregressiveDispatchMixin] forecast(): mode={mode}")

        # 3) dispatch to unified mixin first (the new path)
        try:
            return super().forecast(
                inputs=inputs,
                prediction_length=prediction_length,
                mode=mode,
                quantiles=quantiles,
                **kwargs,
            )
        except NotImplementedError:
            logger.warning(
                f"[AutoregressiveDispatchMixin] unified mixin does not handle mode='{mode}', "
                f"trying legacy mixins (if available)."
            )

        # 4) legacy fallbacks — to NOT break old models
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

        # if we get here – we truly don't support it
        raise NotImplementedError(f"forecast mode='{mode}' not supported on this model.")

    # ------------------------------------------------------------------
    # public generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Expert-level entry, mirrors HF-ish API, but we normalize the old
        `blockwise=True` kwarg to a proper unified `mode`.
        """
        mode = kwargs.pop("mode", None)
        blockwise = kwargs.pop("blockwise", False)

        # prefer explicit mode
        if mode is None and blockwise:
            if self._is_patch_model():
                mode = "patch_blockwise"
            else:
                mode = "block"

        if mode is not None:
            kwargs["mode"] = mode

        logger.info(f"[AutoregressiveDispatchMixin] generate(): mode={mode}")

        # try unified first
        try:
            return super().generate(*args, **kwargs)
        except NotImplementedError:
            logger.warning(
                f"[AutoregressiveDispatchMixin] unified generate() does not handle mode='{mode}', "
                f"trying legacy mixins (if available)."
            )

        # legacy fallback path
        if mode in ("block", "blockwise") and AutoregressiveBlockwiseMixin is not None:
            return AutoregressiveBlockwiseMixin.generate(self, *args, **kwargs)

        if mode in ("step", "stepwise") and AutoregressiveStepwiseMixin is not None:
            return AutoregressiveStepwiseMixin.generate(self, *args, **kwargs)

        raise NotImplementedError(f"generate mode='{mode}' not supported on this model.")
