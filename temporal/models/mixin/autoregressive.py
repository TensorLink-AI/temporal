import torch
import torch.nn as nn
from typing import Optional, Union, Any, List

class AutoregressiveMixin:
    """A mixin for models that perform autoregressive generation.

    This class provides a `generate_autoregressive` method that implements a
    standard autoregressive decoding loop. It is designed to be mixed into a
    `BaseTemporalModel` and can handle both decoder-only and encoder-decoder
    architectures.
    """
    def enable_dropout(self):
        """Enables dropout layers in the model for Monte Carlo sampling.

        This method sets all `nn.Dropout` modules in the model to training mode,
        which is useful for generating ensembles of forecasts via MC dropout.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_scalar_value(self, value: Any, name: str) -> Optional[float]:
        """Safely converts a potential tensor or other type to a float scalar.

        Args:
            value: The value to convert.
            name (str): The name of the value, used for logging warnings.

        Returns:
            Optional[float]: The scalar float value, or None if the input is None.
        """
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() > 1:
                print(f"Warning: '{name}' tensor had {value.numel()} elements. Taking the first element.")
            return float(value.flatten()[0].item())
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert '{name}' (value: {value}, type: {type(value).__name__}) to a float. Error: {e}")

    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,
        eos_token_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generates a sequence autoregressively.

        This method produces a forecast by feeding the model's output from one
        time step back as the input for the next time step.

        Args:
            input_ids (torch.Tensor): The initial input sequence. For encoder-decoder
                models, this is the encoder input. For decoder-only models, this is
                the initial context for the decoder.
            prediction_length (int): The number of time steps to generate.
            attention_mask (Optional[torch.Tensor]): The padding mask for the
                `input_ids`.
            decoder_attention_mask (Optional[torch.Tensor]): An optional padding mask
                for the initial decoder sequence (if different from `attention_mask`).
            use_cache (bool): If True, uses the Key-Value cache for faster generation.
            decoder_start_token_id (Optional[Any]): The token used to start the
                decoding process. Required for encoder-decoder models.
            eos_token_id (Optional[Any]): An optional end-of-sequence token that
                can trigger early stopping.
            **kwargs: Additional keyword arguments to be passed to the model's
                forward pass.

        Returns:
            torch.Tensor: The generated sequence of shape
            `[batch_size, prediction_length, output_features]`.
        """
        batch_size, device = input_ids.shape[0], input_ids.device
        has_encoder = hasattr(self, 'encoder') and self.encoder is not None

        # 1. Process Encoder (if applicable)
        encoder_hidden_states = None
        if has_encoder:
            encoder_outputs = self.encoder(
                input_values=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}),
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state

        # 2. Initialize Decoder Input
        if has_encoder:
            # For encoder-decoder, we must create a start token.
            start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(self.config, "decoder_start_token_id", None)
            if start_token_id is None:
                raise ValueError("An encoder-decoder model requires a 'decoder_start_token_id' to be provided.")
            start_value = self._get_scalar_value(start_token_id, "decoder_start_token_id")
            
            # The feature dimension of the decoder input should match the model's expected feature size.
            decoder_feature_size = getattr(self.config, "feature_size", 1)
            decoder_input_ids = torch.full(
                (batch_size, 1, decoder_feature_size),
                start_value,
                dtype=encoder_hidden_states.dtype,
                device=device,
            )
        else:
            # For decoder-only, the provided input_ids are the initial context.
            decoder_input_ids = input_ids

        # 3. Autoregressive Loop
        predictions: List[torch.Tensor] = []
        past_key_values = None
        current_decoder_attention_mask = decoder_attention_mask
        eos_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        for _ in range(prediction_length):
            # Determine the input for the current step.
            # If using cache, we only need to provide the last generated token.
            step_input = decoder_input_ids[:, -1:, :] if use_cache and past_key_values is not None else decoder_input_ids
            
            if not hasattr(self, 'decoder') or self.decoder is None:
                raise AttributeError("The model must have a 'decoder' attribute for autoregressive generation.")

            # Perform a forward pass through the decoder.
            model_outputs = self.forward(
                decoder_inputs=step_input,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                decoder_attention_mask=current_decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            # The model's forward pass should return logits from the output head.
            # We take the prediction for the very last time step.
            next_prediction = model_outputs.logits[:, -1:, :]
            predictions.append(next_prediction)
            
            # Use the output head's `predict` method to get the input for the next step.
            # This allows heads (like DistPred) to reduce the prediction to a point estimate.
            if hasattr(self.output_heads, "predict"):
                next_input_step = self.output_heads.predict(next_prediction)
            else:
                # Default behavior: assume the head's output can be directly used as input.
                expected_input_features = getattr(self.config, "feature_size", 1)
                next_input_step = next_prediction[:, :, :expected_input_features].contiguous()

            # Append the new prediction to the decoder input sequence.
            decoder_input_ids = torch.cat([decoder_input_ids, next_input_step], dim=1)
            
            # Update attention mask and KV cache for the next iteration.
            if current_decoder_attention_mask is not None:
                new_mask_col = torch.ones((batch_size, 1), dtype=current_decoder_attention_mask.dtype, device=device)
                current_decoder_attention_mask = torch.cat([current_decoder_attention_mask, new_mask_col], dim=1)

            if use_cache:
                past_key_values = model_outputs.past_key_values

            # Check for early stopping condition.
            if eos_scalar is not None and torch.isclose(next_input_step[:, :, 0], torch.tensor(eos_scalar, device=device)).all():
                break

        if not predictions:
            return torch.empty((batch_size, 0, 1), device=device) # Return empty if nothing was generated.

        return torch.cat(predictions, dim=1)
