import torch
import torch.nn as nn
from typing import Any

class EnsembleSampler:
    """A utility for generating Monte Carlo dropout-based ensembles from a model.

    This class provides a convenient context for generating an ensemble of
    predictions by running a model multiple times with dropout enabled during
    inference. It ensures that the model's dropout layers are returned to their
    original evaluation mode afterward, even if an error occurs during generation.

    Attributes:
        model (nn.Module): The model from which to generate ensembles.
        dropout_enabled (bool): A flag to control whether dropout should be
            activated. If False, the model will be run in its default mode.
    """

    def __init__(self, model: nn.Module, dropout_enabled: bool = True):
        """Initializes the EnsembleSampler.

        Args:
            model (nn.Module): The PyTorch model to use for generation.
            dropout_enabled (bool): If True, dropout will be enabled during the
                ensemble generation process.
        """
        self.model = model
        self.dropout_enabled = dropout_enabled

    def _set_dropout_mode(self, enable: bool):
        """Recursively sets the training mode of all nn.Dropout modules in the model.

        Args:
            enable (bool): If True, dropout layers are set to training mode
                (i.e., they will drop units). If False, they are set to
                evaluation mode.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(mode=enable)

    def generate(self, ensemble_size: int, **generate_kwargs: Any) -> torch.Tensor:
        """Generates an ensemble of forecasts by running the model multiple times.

        If dropout is enabled, this method will temporarily switch the model's
        dropout layers to training mode for the duration of the generation process.

        Args:
            ensemble_size (int): The number of samples to generate for the ensemble.
            **generate_kwargs (Any): Keyword arguments to be passed directly to the
                `model.generate()` method.

        Returns:
            torch.Tensor: A tensor containing the ensemble of predictions, stacked
            along a new dimension. The typical shape is `[B, N, T, Q]`, where `N`
            is the `ensemble_size`.
        """
        if self.dropout_enabled:
            self._set_dropout_mode(True)

        all_predictions = []
        try:
            for _ in range(ensemble_size):
                with torch.no_grad(): # Ensure generation is done without gradient tracking
                    prediction = self.model.generate(**generate_kwargs)
                all_predictions.append(prediction)
        finally:
            # Ensure dropout is always returned to its default (eval) mode.
            if self.dropout_enabled:
                self._set_dropout_mode(False)

        # Stack the predictions along a new 'ensemble' dimension.
        return torch.stack(all_predictions, dim=1)
