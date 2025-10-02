from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    # Avoid circular import while allowing type hints
    from temporal.models.transformer_model import TransformerTemporalModel


class TrainingStrategy(ABC):
    """Abstract base class for defining training strategies."""

    @abstractmethod
    def __call__(
        self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Determines the input for the decoder at each training step.
        """
        raise NotImplementedError()

    def get_loss(self) -> torch.Tensor | None:
        """Optional: return a custom scalar loss computed by the strategy."""
        return None


class TeacherForcingStrategy(TrainingStrategy):
    """Always uses ground-truth inputs."""
    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return decoder_inputs


class ScheduledSamplingStrategy(TrainingStrategy):
    """
    Scheduled Sampling: sometimes replace ground-truth decoder inputs with the model's own predictions.
    NOTE: In your patch-latent AR design, *true* scheduled sampling belongs in latent space.
    This strategy uses native-space point forecasts as a pragmatic approximation.
    """

    def __init__(self, total_steps: int, sampling_probability: float = 0.5,
                 prediction_method: str = "median", return_raw: bool = True):
        """
        Args:
            total_steps: steps over which we ramp up the sampling probability.
            sampling_probability: max probability of replacing inputs.
            prediction_method: 'median' (default) or 'mean', passed to head.predict().
            return_raw: if your loss/model expect normalized space, keep True.
        """
        self.total_steps = total_steps
        self.sampling_probability = sampling_probability
        self.prediction_method = prediction_method
        self.return_raw = return_raw

    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        current_step = kwargs.get("current_step", 0)
        p = self._get_sampling_probability(current_step)
        if torch.rand(1, device=decoder_inputs.device).item() < p:
            targets = kwargs.get("targets")
            if targets is None:
                raise ValueError("ScheduledSamplingStrategy requires 'targets' to infer prediction_length.")
            # Inference-only generate: point forecast [B, T, F]
            gen = model.generate(
                encoder_inputs=kwargs.get("encoder_inputs"),
                decoder_inputs=decoder_inputs,
                prediction_length=targets.shape[1],
                attention_mask=kwargs.get("attention_mask"),
                differentiable=False,          # <- no grads for scheduled sampling replacement
                return_samples=False,          # <- get a point forecast, not paths
                sampling=False,
                prediction_strategy=self.prediction_method,
                return_raw=self.return_raw,
                return_bundle=False,           # <- just the tensor
                quantile_levels=None,          # ensure we don't get a [B,T,F,Q]
            )
            # Make sure shapes match decoder_inputs (usually [B, T0, F_in])
            num_input_features = decoder_inputs.shape[-1]
            if gen.ndim == 2:
                gen = gen.unsqueeze(-1)        # [B,T] -> [B,T,1]
            gen = gen[..., :num_input_features]
            return gen
        return decoder_inputs

    def _get_sampling_probability(self, current_step: int) -> float:
        if self.total_steps <= 0:
            return 0.0
        return self.sampling_probability * min(1.0, current_step / self.total_steps)


class RLTrainingStrategy(TrainingStrategy):
    """
    Direct-reward optimization: compute a differentiable reward on the model's *ensemble paths*
    and backprop through it.
    """

    def __init__(self, reward_loss_fn: callable, num_samples: int = 100, return_raw: bool = True):
        """
        Args:
            reward_loss_fn: callable(outputs, targets) -> scalar tensor.
                            Should accept preds as [B,T,F,S] or handle shape internally.
            num_samples: S (ensemble size) for path generation.
            return_raw: if your reward expects normalized space, keep True.
        """
        self.reward_loss_fn = reward_loss_fn
        self.num_samples = int(num_samples)
        self.return_raw = bool(return_raw)
        self.latest_loss: Optional[torch.Tensor] = None

    def __call__(
        self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        self._rl_forward(model, decoder_inputs, **kwargs)
        # We return decoder_inputs so your outer loop can still call model(...)
        # If you want, you could return the samples instead (not necessary).
        return decoder_inputs

    def _rl_forward(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs):
        targets = kwargs.get("targets")
        if targets is None:
            raise ValueError("RLTrainingStrategy requires 'targets' for reward computation.")

        # Ensure float + device alignment (avoids copy-construct warnings)
        device = decoder_inputs.device
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
        else:
            targets = targets.to(device=device, dtype=torch.float32)

        # 1) Generate ensemble *paths* with gradients:
        #    returns a Tensor [B, T, F, S] because we pass return_samples=True
        samples = model.generate(
            encoder_inputs=kwargs.get("encoder_inputs"),
            decoder_inputs=decoder_inputs,
            prediction_length=targets.shape[1],
            attention_mask=kwargs.get("attention_mask"),
            differentiable=True,          # <- grads flow through reconstructor + head (+ sampling if reparam)
            return_samples=True,          # <- paths only
            num_samples=self.num_samples, # <- S
            return_raw=self.return_raw,   # <- keep normalized if your reward expects it
        )

        # 2) Compute reward (scalar). Your CRPS+SMAPE composite can handle [B,T,F,S] and [B,T] targets.
        self.latest_loss = self.reward_loss_fn(samples, targets)

    def get_loss(self) -> torch.Tensor | None:
        return self.latest_loss
