import torch
from typing import Any, Dict, List, Optional, Union

# assume these live somewhere in your codebase
from temporal.models.mixin.autoregressive_patch import AutoregressivePatchMixin
from temporal.models.mixin.autoregressive_stepwise import AutoregressiveStepwiseMixin


import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union

class AutoregressiveDispatchMixin:
    """
    Dispatch between patch‐based and stepwise generation based on config.
    """

    def generate(
        self,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        If the model's value_embedding_config.type == "patch", delegate to
        AutoregressivePatchMixin.generate; otherwise to AutoregressiveStepwiseMixin.generate.
        """
        # assume `self.config.value_embedding_config.type` exists
        is_patch = getattr(self.config.value_embedding_config, "type", None) == "patch"

        if is_patch:
            # call the patch‐based generator
            return super(AutoregressiveDispatchMixin, self).generate(*args, **kwargs)
        else:
            # call the stepwise generator
            return super(AutoregressiveStepwiseMixin, self).generate(*args, **kwargs)
    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        A user-friendly, dispatching forecast method.

        It checks the model's configuration and calls the appropriate forecast
        implementation from either the Patch or Stepwise mixin.
        """
        # Determine the generation type from the model's config
        is_patch = getattr(self.config.value_embedding_config, "type", None) == "patch"

        if is_patch:
            logger.info("Patch-based model detected. Dispatching to AutoregressivePatchMixin.forecast.")
            # Explicitly call the forecast method from the Patch mixin
            return AutoregressivePatchMixin.forecast(
                self, inputs=inputs, prediction_length=prediction_length, **kwargs
            )
        else:
            logger.info("Stepwise model detected. Dispatching to AutoregressiveStepwiseMixin.forecast.")
            # Explicitly call the forecast method from the Stepwise mixin
            return AutoregressiveStepwiseMixin.forecast(
                self, inputs=inputs, prediction_length=prediction_length, **kwargs
            )