from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TrainingStrategy(ABC):
    @abstractmethod
    def __call__(self, model, decoder_inputs, **kwargs):
        raise NotImplementedError()


class TeacherForcingStrategy(TrainingStrategy):
    def __call__(self, model, decoder_inputs, **kwargs):
        return decoder_inputs


class ScheduledSamplingStrategy(TrainingStrategy):
    def __init__(self, total_steps: int, sampling_probability: float = 0.5):
        self.total_steps = total_steps
        self.sampling_probability = sampling_probability

    def __call__(self, model, decoder_inputs, current_step: int, **kwargs):
        if torch.rand(1).item() < self._get_sampling_probability(current_step):
            with torch.no_grad():
                return model.generate(
                    **kwargs,
                    future_window=decoder_inputs.shape[1],
                    prediction_mode="sample",
                ).prediction
        return decoder_inputs

    def _get_sampling_probability(self, current_step: int) -> float:
        return self.sampling_probability * (current_step / self.total_steps)
