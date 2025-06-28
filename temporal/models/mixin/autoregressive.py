import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Tuple, Callable, Dict, List

logger = logging.getLogger(__name__)

class AutoregressiveMixin:
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

    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module # Pass the specific head or the primary head if ModuleList
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Determines the prediction to store based on strategy and head capabilities.
        This is what will be returned in the final 'predictions' list.
        """
        if prediction_strategy is not None:
            # If a specific strategy is requested for the output
            if hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
                return output_head.predict(raw_head_output, method=prediction_strategy)
            elif isinstance(prediction_strategy, float) and hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
                # Sample the specific quantile as the output
                sampled_output = output_head.sample_quantiles(raw_head_output, quantile_levels=[prediction_strategy])
                return sampled_output.squeeze(-1) # [B, 1, F, 1] -> [B, 1, F]
            else:
                logger.warning(f"Prediction strategy '{prediction_strategy}' not supported by head for direct output. Storing raw head output.")
                return raw_head_output
        else:
            # No specific prediction_strategy requested, store raw output or all specified quantiles
            if hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")) and quantile_levels is not None:
                # If `quantile_levels` are explicitly provided, sample them for the output
                return output_head.sample_quantiles(raw_head_output, quantile_levels=quantile_levels)
            else:
                # Default: store the raw output of the head's forward pass
                return raw_head_output

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
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,
        eos_token_id: Optional[Any] = None,
        early_stopping: bool = False,
        output_attentions: bool = False,
        prediction_strategy: Optional[Union[str, float, int]] = None,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Autoregressively generates a sequence of features.

        This method supports both raw feature inputs and pre-processed embeddings.
        - If `input_ids` has 3 dimensions and the last dimension matches `config.feature_size`,
          it is treated as raw features and passed through the `preprocessor`.
        - Otherwise, it is assumed to be pre-processed embeddings.

        Args:
            input_ids (torch.Tensor): The initial input sequence to the model.
                Shape can be `[B, S, F]` for raw features or `[B, S, D]` for embeddings.
            prediction_length (int): The number of future steps to predict.
            attention_mask (Optional[torch.Tensor]): Mask for encoder attention.
            decoder_attention_mask (Optional[torch.Tensor]): Mask for decoder attention.
            use_cache (bool): Whether to use past key values for faster decoding.
            decoder_start_token_id (Optional[Any]): The ID for the starting token of the
                decoder sequence. Defaults to 0 if not provided in config.
            eos_token_id (Optional[Any]): The end-of-sequence token ID for early stopping.
            early_stopping (bool): Whether to stop generation if `eos_token_id` is predicted.
            output_attentions (bool): Whether to return attentions from the model.
            prediction_strategy (Optional[Union[str, float, int]]): Strategy to collapse the
                head's output into a single feature value for *feedback into the decoder*.
                If provided and relevant for the head type (e.g., probabilistic heads),
                it also dictates what is stored as the *final prediction output*.
                - For `DistPredHead`: "mean", "median", float (quantile), int (index).
                - For `GaussianHead`: "mean" (uses mu), float (quantile).
                - For `MixtureOutputHead`: Requires a `predict` method that handles "mean", float (quantile) etc.
                - If `None` (default): The raw output of the head's forward pass is stored as the final prediction.
                  The feedback to the decoder will default to the head's `predict("mean")` if available, or the median.
            quantile_levels (Optional[List[float]]): A list of quantile levels to sample.
                This is primarily used when `prediction_strategy` is `None` and the output head
                is a probabilistic one (like `GaussianHead` or `QuantileRegressionOutputHead`)
                and you want the `generate` function to return multiple quantile samples.
                If `prediction_strategy` is a float, that specific float is used as the quantile for output.
            **kwargs: Additional arguments passed to the encoder/decoder.

        Returns:
            Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
                The generated sequence of predicted features.
                - If `prediction_strategy` is used for a probabilistic head (e.g., "mean" or a specific quantile),
                  the output shape is typically `[B, prediction_length, feature_size]`.
                - If `prediction_strategy` is `None` and the head is probabilistic (e.g., Gaussian, QuantileRegression, Mixture),
                  the output shape can vary:
                    - `GaussianHead`: `[B, prediction_length, feature_size * 2]` (mu, log_sigma)
                    - `QuantileRegressionOutputHead`: `[B, prediction_length, feature_size, num_quantiles]`
                    - `MixtureOutputHead`: `List[Dict]` for each step (parameters of the mixture)
                - For `LinearOutputHead` and `DistPredHead` (when `prediction_strategy` is `None`):
                  `[B, prediction_length, output_size]` or `[B, prediction_length, feature_size, num_outputs]` respectively.
        """
        self.eval()
        batch_size, device = input_ids.shape[0], input_ids.device

        # 1) Preprocess inputs if they are raw features
        if hasattr(self, "preprocessor") and input_ids.ndim == 3 and input_ids.shape[-1] == self.config.feature_size:
            proc = self.preprocessor.process(
                input_values=input_ids,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=False,
                verbose=False,
            )
            processed_inputs = proc["hidden_states"]
            attention_mask = proc["attention_mask"]
        else:
            processed_inputs = input_ids  # Assumed to be embeddings

        # 2) Prepare Encoder Output
        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder is not None:
            encoder_outputs = self.encoder(
                hidden_states=processed_inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}),
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            hidden_size = encoder_hidden_states.shape[-1]
        else:
            # Decoder-only model
            hidden_size = processed_inputs.shape[-1]

        # Cache output_heads type and primary head for hoisting checks
        is_multi_head = isinstance(self.output_heads, nn.ModuleList)
        primary_output_head = self.output_heads[0] if is_multi_head else self.output_heads

        # Check for `predict` and `sample_quantiles` capabilities once
        has_predict_method = hasattr(primary_output_head, "predict") and callable(getattr(primary_output_head, "predict"))
        has_sample_quantiles_method = hasattr(primary_output_head, "sample_quantiles") and callable(getattr(primary_output_head, "sample_quantiles"))

        # 3) Initialize Decoder Input Sequence
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(self.config, "decoder_start_token_id", 0)
        
        start_val = self._get_scalar_value(decoder_start_token_id, "decoder_start_token_id")
        
        decoder_start_tensor = torch.zeros(hidden_size, device=device, dtype=processed_inputs.dtype)
        if hidden_size > 0:
            decoder_start_tensor[0] = start_val
        
        decoder_inputs = decoder_start_tensor.repeat(batch_size, 1, 1)

        if not (hasattr(self, 'encoder') and self.encoder is not None):
            decoder_inputs = torch.cat([processed_inputs, decoder_inputs], dim=1)

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        # 4) Autoregressive Loop
        for _ in range(prediction_length):
            step_attention_mask = internal_decoder_attention_mask if not use_cache or past_key_values is None else None
            step_inputs = decoder_inputs[:, -1:, :] if use_cache and past_key_values is not None else decoder_inputs

            if not hasattr(self, 'decoder'):
                raise AttributeError("Model is missing a decoder, which is required for autoregressive generation.")
            
            decoder_outputs = self.decoder(
                hidden_states=step_inputs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("decoder_kwargs", {}),
            )
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :].contiguous()

            # Get raw output from head(s)
            current_step_raw_head_output = self._get_head_output(last_hidden)

            # Determine what to store as the final prediction for this step
            prediction_to_store = self._compute_prediction_to_store(
                current_step_raw_head_output,
                prediction_strategy,
                quantile_levels,
                primary_output_head # Pass the primary head for method checks
            )
            predictions.append(prediction_to_store)

            # Determine the single value to feed back into the decoder
            next_decoder_input_value = self._compute_next_decoder_input_value(
                current_step_raw_head_output,
                prediction_strategy,
                primary_output_head # Pass the primary head for method checks
            )

            # Re-embed the generated features for the next step
            if hasattr(self, "preprocessor"):
                next_decoder_input_step = self.preprocessor.value_embedding(next_decoder_input_value)
            else:
                next_decoder_input_step = next_decoder_input_value

            decoder_inputs = torch.cat([decoder_inputs, next_decoder_input_step], dim=1)

            if internal_decoder_attention_mask is not None and not use_cache:
                new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None:
                # Check the first feature of the value used for feedback for the EOS token
                if torch.is_tensor(next_decoder_input_value) and next_decoder_input_value.ndim >= 3 and torch.isclose(next_decoder_input_value[:, :, 0], torch.tensor(eos_value_scalar, device=device)).all():
                    logger.info(f"Early stopping triggered at step {_ + 1} due to EOS token prediction.")
                    break
        
        # --- Handle empty predictions or concatenate ---
        if not predictions:
            # Determine the appropriate output_feature_size for an empty tensor
            output_feature_size = 1 # Default
            if hasattr(self, 'output_heads'):
                try:
                    out_head = primary_output_head
                    if isinstance(out_head, (LinearOutputHead, DistPredHead)):
                        if prediction_strategy is not None and has_predict_method:
                            # If strategy used, it typically collapses to feature_size
                            output_feature_size = getattr(out_head, 'feature_size', out_head.output_size)
                        else:
                            output_feature_size = out_head.output_size # Raw output size (F or F*K)
                    elif isinstance(out_head, GaussianHead):
                        output_feature_size = out_head.feature_size * 2 if prediction_strategy is None else out_head.feature_size
                    elif isinstance(out_head, QuantileRegressionOutputHead):
                        if prediction_strategy is not None:
                            output_feature_size = out_head.feature_size
                        elif quantile_levels is not None:
                            output_feature_size = out_head.feature_size * len(quantile_levels)
                        else:
                            output_feature_size = out_head.output_size # Raw output (F * NumQuantiles)
                    elif isinstance(out_head, MixtureOutputHead):
                        # Mixture head returns List[Dict], so an empty Tensor cannot be returned
                        return [] # Returning an empty list of dicts.
                except Exception as e:
                    logger.warning(f"Could not determine output_feature_size for empty tensor: {e}. Defaulting to 1.")
            return torch.empty((batch_size, 0, output_feature_size), device=device)

        # Concatenate results or return list of dicts
        if isinstance(predictions[0], Dict):
            logger.info("AutoregressiveMixin returning a list of dictionaries as prediction output.")
            return predictions # Returns List[Dict] (e.g., from MixtureOutputHead)
        else:
            return torch.cat(predictions, dim=1) # Returns torch.Tensor (e.g., from Linear, Gaussian, Quantile, DistPred)