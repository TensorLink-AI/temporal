
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Union

class AutoregressiveMixin:
    """
    A mixin for autoregressive sequence generation.

    This class provides a `generate` method that can be used by any model
    that inherits from it. The method performs autoregressive decoding, where
    the model's prediction at each time step is fed back as input for the
    next time step.

    The mixin is designed to be flexible and supports:
    - Probabilistic generation (sampling from the output distribution).
    - Deterministic generation (taking the argmax or a specific quantile).
    - Handling of encoder-decoder and decoder-only architectures.
    - Use of a Key-Value (KV) cache for efficient decoding.

    To use this mixin, a model must have a `self.preprocessor` attribute that
    can process raw inputs into embeddings and masks.
    """

    def enable_dropout(self):
        """
        Enables dropout layers during generation.

        This is useful for techniques like Monte Carlo (MC) Dropout, where
        model uncertainty is estimated by performing multiple forward passes
        with dropout enabled.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generates an autoregressive sequence.

        Args:
            context (torch.Tensor): The input sequence for the encoder
                (or the initial context for a decoder-only model).
                Shape: `(batch_size, context_length, feature_size)`.
            attention_mask (Optional[torch.Tensor]): A mask to prevent attention
                to padding tokens in the `context` tensor.
                Shape: `(batch_size, context_length)`.
            prediction_length (Optional[int]): The number of time steps to predict.
                If not provided, it defaults to the model's configured
                `prediction_length`.
            **kwargs: Additional keyword arguments.
                - `use_cache` (bool): Whether to use the KV cache. Defaults to True.
                - `probabilistic` (bool): Whether to sample from the output
                  distribution. Defaults to True.
                - `feedback_quantile` (float): The quantile to use for feedback in
                  deterministic generation. Defaults to 0.5.
                - `return_full_sequence` (bool): If True, returns the context and
                  the predictions concatenated. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If `return_full_sequence` is False (default), returns the generated
              sequence of shape `(batch_size, prediction_length, output_feature_size)`.
            - If `return_full_sequence` is True, returns a tuple containing the
              full sequence and the generated sequence.
        """
        self.eval()

        pred_len = prediction_length if prediction_length is not None else self.config.prediction_length
        use_cache = kwargs.get("use_cache", True)
        probabilistic = kwargs.get("probabilistic", True)
        return_full_sequence = kwargs.get("return_full_sequence", False)

        if self.encoder:
            encoder_inputs = context
            decoder_inputs = context[:, -1:, :]
        else:
            encoder_inputs = None
            decoder_inputs = context

        full_sequence = [context] if self.encoder else [decoder_inputs]
        generated_sequence = []
        past_key_values = None

        for i in range(pred_len):
            current_seq_len = decoder_inputs.shape[1]
            
            # --- Preprocessing Step ---
            processed_decoder = self.preprocessor.process(
                input_values=decoder_inputs,
                past_key_values_length=past_key_values[0][0].shape[2] if past_key_values else 0,
            )

            # The forward pass now uses the preprocessed inputs
            model_outputs = self.forward(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs, # Pass raw inputs
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            # The model's forward pass handles patching, so we take the logit
            # corresponding to the *actual* last token.
            next_token_logits = model_outputs.logits[:, current_seq_len - 1, :]
            past_key_values = model_outputs.past_key_values

            if probabilistic:
                distr = torch.distributions.Normal(next_token_logits, 1.0)
                next_token = distr.sample()
            else:
                if self.config.num_quantiles > 1:
                    quantile_levels = self.config.quantiles
                    feedback_quantile = kwargs.get("feedback_quantile", 0.5)
                    feedback_q_index = quantile_levels.index(feedback_quantile)
                    next_token = next_token_logits[..., feedback_q_index]
                else:
                    next_token = next_token_logits
            
            if next_token.ndim == 2:
                next_token = next_token.unsqueeze(1)
            
            generated_sequence.append(next_token)
            decoder_inputs = torch.cat([decoder_inputs, next_token], dim=1)

            if return_full_sequence:
                full_sequence.append(next_token)

        generated_sequence = torch.cat(generated_sequence, dim=1)

        if return_full_sequence:
            full_sequence = torch.cat(full_sequence, dim=1)
            return full_sequence, generated_sequence
        
        return generated_sequence
