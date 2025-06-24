
import torch
import torch.nn as nn
from typing import Optional

class MultiStepMixin:
    """
    A mixin for multi-step time series forecasting.

    This class provides a `generate` method that performs multi-step forecasting
    by feeding the model's own predictions back as inputs for subsequent steps.
    It is designed for models that predict multiple time steps at once.

    The mixin supports:
    - Iterative forecasting where the context is updated with predictions.
    - Handling of encoder-decoder architectures.
    """

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_iterations: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generates a sequence by iteratively predicting and feeding back.

        Args:
            encoder_inputs (torch.Tensor): The initial context for the model.
                Shape: `(batch_size, context_length, feature_size)`.
            attention_mask (Optional[torch.Tensor]): A mask for the encoder inputs.
            num_iterations (int): The number of times to iterate the prediction
                process. The total prediction length will be
                `num_iterations * prediction_length`.
            **kwargs: Additional keyword arguments.
                - `probabilistic` (bool): Whether to sample from the output
                  distribution for feedback. Defaults to True.
                - `feedback_quantile` (float): The quantile to use for feedback
                  in deterministic generation. Defaults to 0.5.

        Returns:
            torch.Tensor: The generated sequence of predictions.
                Shape: `(batch_size, num_iterations * prediction_length, output_feature_size)`.
        """
        self.eval()

        probabilistic = kwargs.get("probabilistic", True)
        all_predictions = []

        current_encoder_inputs = encoder_inputs

        for _ in range(num_iterations):
            # The decoder input is the last part of the encoder context
            decoder_inputs = current_encoder_inputs[:, -self.config.prediction_length:, :]

            # The main forward pass now handles all preprocessing, including patching.
            outputs = self.forward(
                encoder_inputs=current_encoder_inputs,
                decoder_inputs=decoder_inputs,
                attention_mask=attention_mask,
            )

            # The model's forward pass handles any necessary padding for patching,
            # so we take the logits corresponding to the actual prediction length.
            logits = outputs.logits[:, :self.config.prediction_length, :]

            # Generate the next sequence from the logits
            if probabilistic:
                distr = torch.distributions.Normal(logits, 1.0)
                predictions = distr.sample()
            else:
                if self.config.num_quantiles > 1:
                    quantile_levels = self.config.quantiles
                    feedback_quantile = kwargs.get("feedback_quantile", 0.5)
                    feedback_q_index = quantile_levels.index(feedback_quantile)
                    predictions = logits[..., feedback_q_index]
                else:
                    predictions = logits

            all_predictions.append(predictions)

            # Update the context for the next iteration
            current_encoder_inputs = torch.cat([current_encoder_inputs, predictions], dim=1)

        return torch.cat(all_predictions, dim=1)
