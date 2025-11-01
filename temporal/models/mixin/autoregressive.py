import torch
import torch.nn as nn
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# new unified mixin
from .autoregressive_unified import AutoregressiveUnifiedMixin

# keep stepwise around for legacy / debugging
from .autoregressive_stepwise import AutoregressiveStepwiseMixin
# (we no longer need to import the old patch/blockwise ones here)


class AutoregressiveDispatchMixin:
    """
    Dispatch mixin that routes to the unified autoregressive mixin.

    Rules:
    - if stepwise=True  -> use legacy stepwise (1-step AR)
    - elif blockwise=True -> unified, mode="blockwise"
    - elif model looks patched -> unified, mode="patch_blockwise" (your training-style)
    - else -> unified, mode="blockwise"

    This keeps the old call sites working:
        model.generate(..., blockwise=True)
    while letting newer code do:
        model.generate(..., mode="patch_ar")
        model.generate(..., mode="patch_blockwise")
        model.generate(..., mode="blockwise")
    """

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def _is_patch_based(self) -> bool:
        """
        Checks if the model is patch-based by inspecting its preprocessor/value_embedding.
        """
        if not hasattr(self, "preprocessor"):
            return False
        ve = getattr(self.preprocessor, "value_embedding", None)
        if ve is None:
            return False
        return hasattr(ve, "patch_size")  # your current heuristic

    # ------------------------------------------------------------------
    # forecast -> unified
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        *,
        blockwise: bool = False,
        stepwise: bool = False,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        User-friendly forecast that chooses the right unified mode.

        Params
        ------
        blockwise: force non-patch blockwise (like old AutoregressiveBlockwiseMixin)
        stepwise:  force legacy stepwise (1-step AR; useful for debugging)
        mode:      directly pass a unified mode: "blockwise" | "patch_ar" | "patch_blockwise"
        """
        # 1) explicit legacy stepwise takes priority
        if stepwise:
            logger.info("Stepwise forecasting requested explicitly. Dispatching to AutoregressiveStepwiseMixin.")
            return AutoregressiveStepwiseMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                **kwargs,
            )

        # 2) if caller gave an explicit unified mode, just forward it
        if mode is not None:
            logger.info(f"Unified forecast with explicit mode={mode}.")
            return AutoregressiveUnifiedMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                mode=mode,
                **kwargs,
            )

        # 3) preserve old API: blockwise=True
        if blockwise:
            logger.info("Blockwise forecasting requested. Using unified mode='blockwise'.")
            return AutoregressiveUnifiedMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                mode="blockwise",
                **kwargs,
            )

        # 4) auto-detect patch models -> this is your default for patched models
        if self._is_patch_based():
            logger.info("Patch-based model detected. Using unified mode='patch_blockwise'.")
            return AutoregressiveUnifiedMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                mode="patch_blockwise",
                **kwargs,
            )

        # 5) fallback: normal non-patch blockwise
        logger.info("Non-patch model detected. Using unified mode='blockwise'.")
        return AutoregressiveUnifiedMixin.forecast(
            self,
            inputs=inputs,
            prediction_length=prediction_length,
            quantiles=quantiles,
            mode="blockwise",
            **kwargs,
        )

    # ------------------------------------------------------------------
    # generate -> unified
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        *args,
        blockwise: bool = False,
        stepwise: bool = False,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Expert-level generate that still keeps old flags working.

        You can now do:
            model.generate(decoder_inputs=..., prediction_length=256)   # auto
            model.generate(..., blockwise=True)                        # force non-patch blockwise
            model.generate(..., mode="patch_blockwise")                # train-style patched blocks
            model.generate(..., mode="patch_ar")                       # latent AR (old patch mixin)
            model.generate(..., stepwise=True)                         # legacy stepwise
        """
        # 1) explicit legacy stepwise
        if stepwise:
            logger.info("Stepwise generation requested explicitly. Dispatching to AutoregressiveStepwiseMixin.")
            return AutoregressiveStepwiseMixin.generate(self, *args, **kwargs)

        # 2) explicit unified mode
        if mode is not None:
            logger.info(f"Unified generation with explicit mode={mode}.")
            return AutoregressiveUnifiedMixin.generate(self, *args, mode=mode, **kwargs)

        # 3) old API: blockwise=True
        if blockwise:
            logger.info("Blockwise generation requested. Using unified mode='blockwise'.")
            return AutoregressiveUnifiedMixin.generate(self, *args, mode="blockwise", **kwargs)

        # 4) auto detect patch → patch_blockwise
        if self._is_patch_based():
            logger.info("Patch-based model detected. Using unified mode='patch_blockwise'.")
            return AutoregressiveUnifiedMixin.generate(self, *args, mode="patch_blockwise", **kwargs)

        # 5) fallback: blockwise
        logger.info("Non-patch model detected. Using unified mode='blockwise'.")
        return AutoregressiveUnifiedMixin.generate(self, *args, mode="blockwise", **kwargs)
