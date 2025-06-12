import torch
import torch.nn as nn
from typing import Optional

class MultiStepMixin:
    """A mixin for models that support direct multi-step forecasting.

    This class provides a `generate_multistep` method that predicts the entire
    forecast horizon in a single forward pass. This is in contrast to autoregressive
    generation, which predicts one step at a time.
    """
    def enable_dropout(self):
        """Enables dropout layers in the model.

        While typically not used for deterministic multi-step generation, this
        can be useful for creating ensembles via Monte Carlo dropout if desired.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def generate_multistep(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generates the entire forecast horizon in a single forward pass.

        This method is suitable for models trained to directly predict a
        sequence of future values.

        Args:
            input_ids (torch.Tensor): The input sequence. For encoder-decoder models,
                this is the encoder input. For decoder-only models, this is the
                historical context.
            prediction_length (int): The number of future time steps to predict.
            attention_mask (Optional[torch.Tensor]): A padding mask for the
                `input_ids`.
            decoder_attention_mask (Optional[torch.Tensor]): An explicit mask for
                the decoder's self-attention mechanism. If not provided, a causal
                mask is typically generated.
            **kwargs: Additional keyword arguments to be passed to the model's
                forward pass.

        Returns:
            torch.Tensor: The predicted sequence of shape
            `[batch_size, prediction_length, output_features]`.
        """
        has_encoder = hasattr(self, 'encoder') and self.encoder is not None
        
        # In multi-step generation, the full output is computed at once, so KV caching is not used.
        kwargs['use_cache'] = False
        
        # 1. Prepare inputs for the decoder.
        if has_encoder:
            # Encoder-Decoder: The decoder input is a placeholder for the future.
            # We use the last known value from the input as a simple placeholder.
            last_value = input_ids[:, -1:, :]
            decoder_input = last_value.expand(-1, prediction_length, -1).clone()
            
            # The encoder processes the historical context.
            encoder_kwargs = kwargs.get("encoder_kwargs", {})
            encoder_outputs = self.encoder(
                input_values=input_ids,
                attention_mask=attention_mask,
                **encoder_kwargs
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
        else:
            # Decoder-Only: The decoder receives both the history and future placeholders.
            last_value = input_ids[:, -1:, :]
            future_placeholders = last_value.expand(-1, prediction_length, -1).clone()
            decoder_input = torch.cat([input_ids, future_placeholders], dim=1)
            encoder_hidden_states = None

        # 2. Perform a single forward pass through the main model.
        # The `forward` method of the main model is responsible for handling
        # the creation of the final attention masks.
        model_outputs = self.forward(
            encoder_inputs=input_ids if has_encoder else None,
            decoder_inputs=decoder_input,
            attention_mask=attention_mask, # For encoder padding or decoder history padding
            **kwargs,
        )
        
        # The logits from the forward pass contain the predictions.
        logits = model_outputs.logits

        # For decoder-only models, the output contains predictions for the
        # historical part as well, so we must slice out only the future part.
        if not has_encoder:
            return logits[:, -prediction_length:, :]
        else:
            return logits
