import torch
import torch.nn as nn


class EnsembleSampler:
    """
    Utility for generating Monte Carlo dropout-based ensembles from a model.
    Ensures dropout is temporarily activated only during generation.
    """

    def __init__(self, model: nn.Module, dropout_enabled: bool = True):
        self.model = model
        self.dropout_enabled = dropout_enabled

    def _set_dropout_mode(self, enable: bool):
        """Temporarily toggle dropout layers to train/eval mode."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train(mode=enable)

    def generate(self, ensemble_size: int, **generate_kwargs) -> torch.Tensor:
        """
        Run the model N times with dropout enabled to produce an ensemble of forecasts.

        Args:
            ensemble_size: number of forward passes (samples)
            generate_kwargs: kwargs to pass to model.generate()

        Returns:
            Tensor of shape [B, N, T, Q] (ensemble dim = N)
        """
        if self.dropout_enabled:
            self._set_dropout_mode(True)

        preds = []
        try:
            for _ in range(ensemble_size):
                out = self.model.generate(**generate_kwargs)
                preds.append(out)
        finally:
            if self.dropout_enabled:
                self._set_dropout_mode(False)  # restore dropout to default eval mode

        return torch.stack(preds, dim=1)  # [B, N, T, Q]
