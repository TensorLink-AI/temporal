from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    # This avoids a circular import while still allowing type hints
    from temporal.models.transformer_model import TransformerTemporalModel


class TrainingStrategy(ABC):
    """Abstract base class for defining training strategies."""

    @abstractmethod
    def __call__(
        self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Determines the input for the decoder at each training step.

        Args:
            model (TransformerTemporalModel): The main transformer model instance.
            decoder_inputs (torch.Tensor): The ground-truth decoder inputs.
            **kwargs: Additional context like current_step, encoder_hidden_states, etc.

        Returns:
            torch.Tensor: The potentially modified decoder inputs for the forward pass.
        """
        raise NotImplementedError()
    
    def get_loss(self) -> torch.Tensor | None:
        """
        Returns a custom loss computed by the strategy, if any.
        By default, strategies do not compute their own loss.
        """
        return None


class TeacherForcingStrategy(TrainingStrategy):
    """The standard training strategy that always uses ground-truth inputs."""

    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return decoder_inputs


class ScheduledSamplingStrategy(TrainingStrategy):
    """
    Implements Scheduled Sampling, which stochastically replaces ground-truth
    inputs with the model's own predictions during training.
    """
    def __init__(self, total_steps: int, sampling_probability: float = 0.5):
        self.total_steps = total_steps
        self.sampling_probability = sampling_probability

    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        current_step = kwargs.get("current_step", 0)
        
        if torch.rand(1).item() < self._get_sampling_probability(current_step):
            with torch.no_grad():
                targets = kwargs.get("targets")
                if targets is None:
                    raise ValueError("ScheduledSamplingStrategy requires 'targets' to determine the correct prediction_length.")
                
                generation_kwargs = {
                    "encoder_inputs": kwargs.get("encoder_inputs"),
                    "decoder_inputs": decoder_inputs,
                    "prediction_length": targets.shape[1], # FIX: Use target length
                    "attention_mask": kwargs.get("attention_mask"),
                    "return_bundle": False,
                }
                generated_output = model.generate(**generation_kwargs)
                num_input_features = decoder_inputs.shape[-1]
                return generated_output[..., :num_input_features]

        return decoder_inputs

    def _get_sampling_probability(self, current_step: int) -> float:
        if self.total_steps <= 0:
            return 0.0
        return self.sampling_probability * min(1.0, current_step / self.total_steps)


class RLTrainingStrategy(TrainingStrategy):
    """
    Implements a "direct reward optimization" strategy. It uses a composite
    loss (CRPS + sMAPE) calculated from the model's direct ensemble output
    and backpropagates through it.
    """

    def __init__(self, reward_loss_fn: callable):
        """
        Args:
            reward_loss_fn (callable): A function that takes the generated ensemble
                                       and targets, and returns a scalar loss.
        """
        self.reward_loss_fn = reward_loss_fn
        self.latest_loss = None

    def __call__(
        self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        self._rl_forward(model, decoder_inputs, **kwargs)
        return decoder_inputs

    def _rl_forward(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs):
        targets = kwargs.get("targets")
        if targets is None:
            raise ValueError("RLTrainingStrategy requires 'targets' for loss calculation.")

        generation_kwargs = {
            "encoder_inputs": kwargs.get("encoder_inputs"),
            "decoder_inputs": decoder_inputs,
            "prediction_length": targets.shape[1],
            "attention_mask": kwargs.get("attention_mask"),
        }

        # --- 1. Generate the Ensemble (with gradients) ---
        # Call `generate` with gradients enabled and `return_params=True`.
        # Based on your description, this returns the ensemble of 100 paths.
        with torch.enable_grad():
            model.train()
            # This is our ensemble of shape [B, T, F, 100]
            ensemble_paths = model.generate(**generation_kwargs, return_params=True)

        # --- 2. Calculate the Composite Loss ---
        # Ensure target shape matches the feature dimension of the ensemble
        if targets.ndim == 2 and ensemble_paths.ndim == 4:
            targets = targets.unsqueeze(2).expand(-1, -1, ensemble_paths.shape[2])

        # The reward function now directly returns our final loss
        self.latest_loss = self.reward_loss_fn(ensemble_paths, targets)

    def get_loss(self) -> torch.Tensor | None:
        """Allows the training loop to retrieve the computed RL loss."""
        return self.latest_loss


# This code should be placed in your training script or a utility file.
import torch
import torch.nn.functional as F
from temporal.losses import crps_ensemble


# --- sMAPE metric (unchanged) ---
def smape_loss(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    numerator = torch.abs(preds - targets)
    denominator = (torch.abs(preds) + torch.abs(targets)) / 2.0 + epsilon
    smape = torch.mean(numerator / denominator, dim=list(range(1, numerator.ndim)))
    return smape

# --- The Corrected Composite Loss Function ---
class CompositeRLoss:
    """
    A callable class that computes a composite loss based on ensemble CRPS and sMAPE.
    It normalizes the inputs to ensure both metrics are on a similar scale.
    """
    def __init__(self, crps_weight: float = 0.5, smape_weight: float = 0.5, epsilon: float = 1e-8):
        self.crps_scorer = crps_ensemble(reduction='none')
        self.crps_weight = crps_weight
        self.smape_weight = smape_weight
        self.epsilon = epsilon

    def __call__(self, ensemble_forecasts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the composite loss from a normalized ensemble of forecasts.

        Args:
            ensemble_forecasts (torch.Tensor): The generated ensemble, shape `(B, T, F, N)`.
            targets (torch.Tensor): The ground truth values, shape `(B, T, F)`.

        Returns:
            torch.Tensor: The final composite loss for the batch, as a scalar.
        """
        # --- 1. Normalize the data ---
        # We normalize based on the scale of the target data to make CRPS scale-invariant.
        # The mean over time and batch gives a stable scaling factor.
        with torch.no_grad():
            scale = torch.abs(targets).mean() + self.epsilon
        
        normalized_targets = targets / scale
        normalized_ensemble = ensemble_forecasts / scale

        # --- 2. Calculate CRPS Loss on Normalized Data ---
        crps_per_feature = []
        for f in range(normalized_ensemble.shape[2]):
            crps_loss_f = self.crps_scorer(normalized_ensemble[:, :, f, :], normalized_targets[:, :, f])
            crps_per_feature.append(crps_loss_f.mean())
        
        crps_loss = torch.stack(crps_per_feature).mean()

        # --- 3. Calculate sMAPE Loss ---
        # sMAPE is already scale-invariant, but we calculate it on the original data for true interpretation.
        point_forecast = ensemble_forecasts.mean(dim=-1)
        smape_score = smape_loss(point_forecast, targets).mean()
        
        # --- 4. Combine the losses ---
        # Now that CRPS is also a small, scale-invariant number, the weights are meaningful.
        combined_loss = (self.crps_weight * crps_loss) + (self.smape_weight * smape_score)
        
        return combined_loss