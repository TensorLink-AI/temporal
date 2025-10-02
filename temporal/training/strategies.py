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
        """
        Args:
            total_steps (int): The total number of training steps for the annealing schedule.
            sampling_probability (float): The maximum probability of using student-forcing.
        """
        self.total_steps = total_steps
        self.sampling_probability = sampling_probability

    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        current_step = kwargs.get("current_step", 0)
        
        # Determine if we should use the model's own generation (student forcing)
        if torch.rand(1).item() < self._get_sampling_probability(current_step):
            with torch.no_grad():
                # Generate a sequence using the model's autoregressive capabilities.
                # Pass all relevant context from the forward pass to the generate function.
                generation_kwargs = {
                    "encoder_hidden_states": kwargs.get("encoder_hidden_states"),
                    "future_window": decoder_inputs.shape[1],
                    "prediction_mode": "sample",
                }
                return model.generate(**generation_kwargs).prediction
        
        # Otherwise, use the ground-truth inputs (teacher forcing)
        return decoder_inputs

    def _get_sampling_probability(self, current_step: int) -> float:
        """Linearly anneals the sampling probability from 0 to the max value."""
        return self.sampling_probability * min(1.0, current_step / self.total_steps)


class RLTrainingStrategy(TrainingStrategy):
    """
    Implements a Reinforcement Learning strategy using a policy gradient
    (REINFORCE) approach. This strategy computes its own loss.
    """

    def __init__(self, reward_function: callable):
        """
        Args:
            reward_function (callable): A function that takes two tensors 
                                        (generated_sequence, target_sequence)
                                        and returns a scalar reward for each
                                        item in the batch.
        """
        self.reward_function = reward_function
        self.latest_loss = None

    def __call__(
        self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        targets = kwargs.get("targets")
        if targets is None:
            raise ValueError("RLTrainingStrategy requires 'targets' to be passed for generation length and reward calculation.")

        # This strategy computes a custom loss and does not modify the decoder inputs
        # for the standard forward pass. The training loop must be adapted to use
        # the loss from `get_loss()` instead of the one from the model output.

        # 1. Generate a full sequence autoregressively.
        # The generate function is polymorphic and handles both decoder-only (by ignoring
        # encoder_hidden_states=None) and encoder-decoder architectures.
        generation_kwargs = {
            "encoder_hidden_states": kwargs.get("encoder_hidden_states"),
            "max_length": targets.shape[1],
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        generation_result = model.generate(**generation_kwargs)
        generated_sequence = generation_result.sequences

        # 2. Score the generated sequence using the custom reward function.
        with torch.no_grad():
            rewards = self.reward_function(generated_sequence, targets).to(generated_sequence.device)

        # 3. Calculate the Policy Gradient Loss.
        # Stack the logits from each generation step.
        all_logits = torch.stack(generation_result.scores, dim=1)  # [B, L, VocabSize]
        
        # Get the log probabilities of the tokens that were actually generated.
        log_softmax_logits = F.log_softmax(all_logits, dim=-1)
        token_log_probs = torch.gather(log_softmax_logits, 2, generated_sequence.unsqueeze(-1)).squeeze(-1)
        
        # Sum log probabilities for the entire sequence.
        sequence_log_probs = token_log_probs.sum(dim=-1)

        # 4. Compute the final loss using reward baseline to reduce variance.
        baseline = rewards.mean()
        advantage = (rewards - baseline).detach()
        
        # Policy gradient loss. The negative sign is because optimizers perform gradient descent.
        self.latest_loss = -torch.mean(sequence_log_probs * advantage)

        # Return the original decoder_inputs. The main training loop will ignore
        # the model's standard loss and use the one from this strategy.
        return decoder_inputs

    def get_loss(self) -> torch.Tensor | None:
        """Allows the training loop to retrieve the computed RL loss."""
        return self.latest_loss