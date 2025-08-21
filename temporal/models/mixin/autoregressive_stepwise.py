import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Tuple, Callable, Dict, List

logger = logging.getLogger(__name__)

class AutoregressiveStepwiseMixin:
    """
    A mixin class for autoregressive generation capabilities in neural network models.
    """
    @staticmethod
    def _get_cache_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        # layer 0, key tensor
        first_layer = past_key_values[0]
        key_tensor = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer["k"]
        # handle common shapes
        if key_tensor.ndim == 4:
            # possibilities: (B,H,L,D) or (B,L,H,D)
            nh = getattr(self.config, "num_attention_heads", None)
            if nh is not None:
                if key_tensor.shape[1] == nh:   # (B,H,L,D)
                    return int(key_tensor.shape[2])
                if key_tensor.shape[2] == nh:   # (B,L,H,D)
                    return int(key_tensor.shape[1])
            # fallback: take the larger of the middle dims as length
            return int(max(key_tensor.shape[1], key_tensor.shape[2]))
        if key_tensor.ndim == 3:
            # e.g., fused heads: (B,L,D)
            return int(key_tensor.shape[1])
        raise ValueError(f"Unexpected KV shape: {tuple(key_tensor.shape)}")

    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """ Safely converts a potential tensor value to a float scalar. """
        if value is None:
            return None
        if torch.is_tensor(value):
            temp_value = value
            while temp_value.numel() > 1:
                logger.warning(f"Tensor for '{name}' had {temp_value.numel()} elements. Taking the first element.")
                temp_value = temp_value[0]
            if temp_value.numel() == 1:
                return float(temp_value.item())
            else:
                raise ValueError(f"Could not reduce '{name}' tensor (original shape {value.shape}) to a scalar.")
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert '{name}'={value} (type {type(value)}) to float scalar. Error: {e}")

    # --- Helper methods for core loop logic ---

    def _get_head_output(self, last_hidden: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Applies output head(s) to the last hidden state."""
        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads, which is required for autoregressive generation.")

        if isinstance(self.output_heads, nn.ModuleList):
            return [head(last_hidden) for head in self.output_heads]
        else:
            return self.output_heads(last_hidden)
            
    def _normalize_levels(self, quantile_levels):
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module  # primary head if ModuleList
    ) -> Union[torch.Tensor, Dict[str, Any], List[torch.Tensor]]:
        """
        Decide what to store for this AR step:
        - If strategy given and head supports predict(): use it
        - Else if quantiles requested and head supports sample_quantiles(): sample them
        - Else store raw output.
        Works for single head or ModuleList.
        """
        levels = self._normalize_levels(quantile_levels)

        # --- Multi-head path ---
        if isinstance(raw_head_output, list):
            assert isinstance(self.output_heads, nn.ModuleList), \
                "raw_head_output is a list but self.output_heads is not ModuleList."

            per_head = []
            for h, y in zip(self.output_heads, raw_head_output):
                if prediction_strategy is not None and hasattr(h, "predict") and callable(getattr(h, "predict")):
                    per_head.append(h.predict(y, method=prediction_strategy))  # tensor [B,1,F] or similar
                elif levels is not None and hasattr(h, "sample_quantiles") and callable(getattr(h, "sample_quantiles")):
                    qy = h.sample_quantiles(y, quantile_levels=levels)         # tensor [B,1,F,Q] or [B,1,Q]
                    per_head.append(qy)
                else:
                    per_head.append(y)

            # If you want to aggregate for storage when multiple heads exist:
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                try:
                    return self.head_aggregator(per_head)  # your aggregator must handle shapes
                except Exception as e:
                    logger.warning(f"head_aggregator failed during store; returning per-head list. Error: {e}")
                    return per_head
            return per_head

        # --- Single-head path ---
        y = raw_head_output
        if prediction_strategy is not None and hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
            return output_head.predict(y, method=prediction_strategy)
        if levels is not None and hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            return output_head.sample_quantiles(y, quantile_levels=levels)
        return y


    def _compute_next_decoder_input_value(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        output_head: nn.Module # Pass the specific head or the primary head if ModuleList
    ) -> torch.Tensor:
        """
        Determines the single, collapsed value to feed back into the decoder.
        This value should always be a torch.Tensor of shape [B, 1, F].
        """
        feedback_source = raw_head_output
        if isinstance(raw_head_output, List): # If multiple heads and no aggregator
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                feedback_source = self.head_aggregator(raw_head_output)
            else:
                logger.warning(
                    "Multiple output heads detected without a 'head_aggregator'. "
                    "Defaulting to the output of the first head for autoregressive feedback. "
                    "Consider implementing a 'head_aggregator' for robust multi-head handling."
                )
                feedback_source = raw_head_output[0]

        # Prioritize `predict` method on the head for feedback
        if hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
            feedback_method = prediction_strategy if prediction_strategy is not None else "mean"
            return output_head.predict(feedback_source, method=feedback_method)

        # Fallback to `sample_quantiles` for probabilistic heads
        elif hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            feedback_quantile = prediction_strategy if isinstance(prediction_strategy, float) else 0.5
            sampled_for_feedback = output_head.sample_quantiles(feedback_source, quantile_levels=[feedback_quantile])
            return sampled_for_feedback.squeeze(-1) # [B, 1, F, 1] -> [B, 1, F]

        # Generic fallback for tensor outputs
        elif isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4: # [B, 1, F, K/Q] -> take mean across K/Q for feedback
                return feedback_source.mean(dim=-1)
            elif feedback_source.ndim == 3: # [B, 1, F] or [B, 1, 1]
                return feedback_source
            else:
                raise ValueError(f"Unexpected feedback_source tensor dimension: {feedback_source.ndim}. Expected 3 or 4.")
        elif isinstance(feedback_source, Dict):
            raise TypeError(
                "MixtureOutputHead (or similar output head returning a dict) must implement a "
                "'predict' method to provide a single tensor for autoregressive feedback."
            )
        else:
            raise TypeError(f"Unhandled feedback_source type: {type(feedback_source)}")

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,
        eos_token_id: Optional[Any] = None,
        early_stopping: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        prediction_strategy: Optional[Union[str, float, int]] = None,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Autoregressively generates a sequence of features.

        Args:
            encoder_inputs (torch.Tensor): The initial input sequence to the encoder.
            decoder_inputs (torch.Tensor, optional): The initial input sequence for the decoder.
            prediction_length (int): The number of future steps to predict.
            attention_mask (Optional[torch.Tensor]): Mask for encoder attention.
            decoder_attention_mask (Optional[torch.Tensor]): Mask for decoder attention.
            use_cache (bool): Whether to use past key values for faster decoding.
            decoder_start_token_id (Optional[Any]): The ID for the starting token of the
                decoder sequence. Defaults to 0 if not provided in config.
            eos_token_id (Optional[Any]): The end-of-sequence token ID for early stopping.
            early_stopping (bool): Whether to stop generation if `eos_token_id` is predicted.
            output_attentions (bool): Whether to return attentions from the model.
            output_hidden_states (bool): Whether to return hidden states from the model.
            prediction_strategy (Optional[Union[str, float, int]]): Strategy to collapse the
                head's output into a single feature value.
            quantile_levels (Optional[List[float]]): A list of quantile levels to sample.
            validate_shapes (bool): Whether to validate input shapes.
            verbose (bool): Whether to print processing information.
            **kwargs: Additional arguments passed to the encoder/decoder.

        Returns:
            Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
                The generated sequence of predicted features.
        """
        self.eval()
        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        if prediction_length is None:
            prediction_length = getattr(self.config, 'prediction_length', 0)

        ref_tensor = decoder_inputs if encoder_inputs is None else encoder_inputs
        batch_size, device, dtype = ref_tensor.shape[0], ref_tensor.device, ref_tensor.dtype

        encoder_outputs = None
        if self.encoder and encoder_inputs is not None:
            processed_encoder = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            encoder_outputs = self.encoder(
                hidden_states=processed_encoder["hidden_states"],
                attention_mask=processed_encoder["attention_mask"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state if encoder_outputs else None
        
        primary_output_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
        
        if decoder_inputs is None:
            start_token_id = getattr(self.config, "decoder_start_token_id", 0) if decoder_start_token_id is None else decoder_start_token_id
            start_val = self._get_scalar_value(start_token_id, "decoder_start_token_id")
            decoder_inputs = torch.full((batch_size, 1, self.config.feature_size), start_val, device=device, dtype=dtype)

        predictions = []
        past_key_values = None
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        # The next input to the decoder starts as the initial token(s).
        next_input = decoder_inputs

        for _ in range(prediction_length):
            # The input for this step is just the single, most recent token.
            # The model's history is managed by `past_key_values`.
            step_input = next_input

            # Get the length of the cache for the preprocessor
            past_kv_length = self._get_cache_length(past_key_values)
            
            processed_decoder = self.preprocessor.process(
                input_values=step_input,
                past_key_values_length=past_kv_length,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            
            decoder_outputs = self.decoder(
                hidden_states=processed_decoder["hidden_states"],
                attention_mask=processed_decoder["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            last_hidden = decoder_outputs.last_hidden_state # Shape: [B, 1, D]

            current_step_raw_head_output = self._get_head_output(last_hidden)

            prediction_to_store = self._compute_prediction_to_store(
                current_step_raw_head_output,
                prediction_strategy,
                quantile_levels,
                primary_output_head
            )
            predictions.append(prediction_to_store)

            next_decoder_input_value = self._compute_next_decoder_input_value(
                current_step_raw_head_output,
                prediction_strategy,
                primary_output_head
            )

            # --- KEY CHANGE ---
            # Instead of concatenating, the output of this step becomes the input for the next.
            next_input = next_decoder_input_value

            # Update the cache for the next iteration. This is the crucial step.
            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None:
                if torch.isclose(next_decoder_input_value.squeeze(), torch.tensor(eos_value_scalar, device=device)).all():
                    logger.info(f"Early stopping triggered at step {_ + 1} due to EOS token prediction.")
                    break
        
        if not predictions:
            return torch.empty((batch_size, 0, self.config.feature_size), device=device, dtype=dtype)

        if isinstance(predictions[0], Dict):
            logger.info("AutoregressiveMixin returning a list of dictionaries. Skipping denormalization.")
            return predictions
        else:
            final_predictions = torch.cat(predictions, dim=1)
            if hasattr(self.preprocessor, 'denormalize'):
                logger.info("Denormalizing final predictions.")
                final_predictions = self.preprocessor.denormalize(final_predictions)
            return final_predictions

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        A user-friendly wrapper for the `generate` method, tailored for forecasting tasks.

        This method simplifies the forecasting process by automatically handling the
        distinction between encoder-decoder and decoder-only models based on the
        model's architecture.

        Args:
            inputs (torch.Tensor): The input data.
                - For Encoder-Decoder models: This is the historical context sequence.
                - For Decoder-Only models: This is the initial prompt sequence.
            prediction_length (int): The number of future steps to forecast.
            quantiles (Optional[List[float]]): A list of quantile levels to sample.
            **kwargs: Additional arguments to be passed directly to the
                      underlying `generate` method (e.g., `prediction_strategy`).
        """
        if hasattr(self, 'encoder') and self.encoder is not None:
            return self.generate(
                encoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                **kwargs,
            )
        else:
            return self.generate(
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                **kwargs,
            )