import logging
from typing import Any, Dict, List, Optional, Union

import torch

# Import all three autoregressive implementations
from .autoregressive_latent import AutoregressiveLatentMixin
from .autoregressive_patch import AutoregressivePatchMixin
from .autoregressive_stepwise import AutoregressiveStepwiseMixin

logger = logging.getLogger(__name__)


class AutoregressiveMixin(
    AutoregressiveLatentMixin, AutoregressivePatchMixin, AutoregressiveStepwiseMixin
):
    """
    A mixin that dynamically dispatches autoregressive calls to the optimal
    implementation (stepwise, latent, or patch-based) by inspecting the
    model's components and configuration at runtime.

    This should be placed *first* in the inheritance list of a model that needs
    to support multiple AR strategies.
    """

    def _is_patch_based(self) -> bool:
        """
        Checks if the model is patch-based by inspecting its preprocessor.
        """
        if not hasattr(self, "preprocessor"):
            return False
        # A model is patch-based if its preprocessor has a patch_size > 1
        return getattr(self.preprocessor, "patch_size", 1) > 1

    @torch.no_grad()
    def generate(
        self, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Dispatches the expert-level `generate` call to the correct implementation
        based on the model's architecture and the `generation_strategy` config.
        """
        # Determine strategy: 1. from kwargs, 2. from config, 3. default to 'stepwise'
        strategy = kwargs.pop(
            "generation_strategy", getattr(self.config, "generation_strategy", "stepwise")
        )
        is_patch_model = self._is_patch_based()

        # --- Validation and Dispatch Logic ---

        # 1. Handle patch-based models: they MUST use the 'patch' strategy.
        if is_patch_model:
            if strategy != "patch":
                logger.warning(
                    f"Model is patch-based, but `generation_strategy` was set to '{strategy}'. "
                    f"Overriding to use the required 'patch' strategy."
                )
            logger.debug("Patch-based model detected. Dispatching to AutoregressivePatchMixin.generate.")
            return AutoregressivePatchMixin.generate(self, *args, **kwargs)

        # 2. Handle non-patch models (stepwise vs. latent)
        if strategy == "latent":
            logger.debug(
                "Non-patch model with 'latent' strategy. Dispatching to AutoregressiveLatentMixin.generate."
            )
            return AutoregressiveLatentMixin.generate(self, *args, **kwargs)

        elif strategy == "stepwise":
            logger.debug(
                "Non-patch model with 'stepwise' strategy. Dispatching to AutoregressiveStepwiseMixin.generate."
            )
            return AutoregressiveStepwiseMixin.generate(self, *args, **kwargs)

        else:
            raise ValueError(
                f"Invalid generation_strategy '{strategy}' for a non-patch-based model. "
                f"Valid options are 'latent' or 'stepwise'."
            )

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        *,
        generation_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        A user-friendly, dispatching forecast method.

        This method serves as a simple wrapper around the more complex `generate`
        method, automatically routing the call to the best AR implementation.

        Parameters
        ----------
        inputs : Tensor
            The input sequence for forecasting.
        prediction_length : int
            The number of future time steps to predict.
        quantiles : List[float], optional
            A list of quantiles to generate for the forecast, by default None.
        generation_strategy : str, optional
            Overrides the default generation strategy in the model's config.
            For non-patch models, options are "latent" or "stepwise".
            For patch-based models, this is ignored and always defaults to "patch".
            By default None.
        """
        # Pass the user's chosen strategy to the generate method.
        if generation_strategy is not None:
            kwargs["generation_strategy"] = generation_strategy

        # Determine if the model is encoder-decoder or decoder-only to correctly
        # name the input tensor for the `generate` method.
        if hasattr(self, "encoder") and self.encoder is not None:
            kwargs["encoder_inputs"] = inputs
        else:
            kwargs["decoder_inputs"] = inputs
            
        return self.generate(
            prediction_length=prediction_length, 
            quantile_levels=quantiles, 
            **kwargs
        )

