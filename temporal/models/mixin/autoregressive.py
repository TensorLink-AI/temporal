import torch
import torch.nn as nn
import logging
from typing import Any, Dict, List, Optional, Union

# Make sure the paths to your other mixins are correct
from .autoregressive_patch import AutoregressivePatchMixin
from .autoregressive_stepwise import AutoregressiveStepwiseMixin
from .autoregressive_blockwise import AutoregressiveBlockwiseMixin

logger = logging.getLogger(__name__)


class AutoregressiveDispatchMixin:
    """
    A mixin that dynamically dispatches calls to the correct implementation
    (patch-based or stepwise) by inspecting the model's components at runtime.

    This should be placed *first* in the inheritance list of a model.
    """

    def _is_patch_based(self) -> bool:
        """
        Checks if the model is patch-based by inspecting its value_embedding module.
        """
        if not hasattr(self, "preprocessor") or not hasattr(
            self.preprocessor, "value_embedding"
        ):
            return False
        # If the embedding module has a 'patch_size', we treat it as a patch model.
        return hasattr(self.preprocessor.value_embedding, "patch_size")

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        blockwise: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        A user-friendly, dispatching forecast method.
        """
        if blockwise:
            logger.info(
                "Block-based forecasting requested. Dispatching to AutoregressiveBlockwiseMixin.forecast."
            )
            return AutoregressiveBlockwiseMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                **kwargs,
            )
        elif self._is_patch_based():
            logger.info(
                "Patch-based model detected. Dispatching to AutoregressivePatchMixin.forecast."
            )
            # CORRECT WAY: Call the parent method directly by name
            return AutoregressivePatchMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                **kwargs,
            )
        else:
            logger.info(
                "Stepwise model detected. Dispatching to AutoregressiveStepwiseMixin.forecast."
            )
            # CORRECT WAY: Call the parent method directly by name
            return AutoregressiveStepwiseMixin.forecast(
                self,
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
                **kwargs,
            )

    @torch.no_grad()
    def generate(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Dispatches the expert-level `generate` call to the correct implementation.
        """
        blockwise = kwargs.pop("blockwise", False)
        if blockwise:
            logger.info(
                "Block-based generation requested. Dispatching to AutoregressiveBlockwiseMixin.generate."
            )
            return AutoregressiveBlockwiseMixin.generate(self, *args, **kwargs)

        elif self._is_patch_based():
            logger.info(
                "Patch-based model detected. Dispatching to AutoregressivePatchMixin.generate."
            )
            # CORRECT WAY: Call the parent method directly by name
            return AutoregressivePatchMixin.generate(self, *args, **kwargs)
        else:
            logger.info(
                "Stepwise model detected. Dispatching to AutoregressiveStepwiseMixin.generate."
            )
            # CORRECT WAY: Call the parent method directly by name
            return AutoregressiveStepwiseMixin.generate(self, *args, **kwargs)
