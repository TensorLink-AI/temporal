import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, List

logger = logging.getLogger(__name__)

class AutoregressiveStepwiseMixin:
    """
    A mixin class for autoregressive generation capabilities in neural network models.
    """
    # FIX: Removed @staticmethod decorator
    def _get_cache_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        first_layer = past_key_values[0]
        key_tensor = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer["k"]
        if key_tensor.ndim == 4:
            nh = getattr(self.config, "num_attention_heads", None)
            if nh is not None:
                if key_tensor.shape[1] == nh: return int(key_tensor.shape[2])
                if key_tensor.shape[2] == nh: return int(key_tensor.shape[1])
            return int(max(key_tensor.shape[1], key_tensor.shape[2]))
        if key_tensor.ndim == 3:
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
            t = value
            while t.numel() > 1:
                logger.warning(f"Tensor for '{name}' had {t.numel()} elements. Taking the first.")
                t = t[0]
            if t.numel() == 1:
                return float(t.item())
            else:
                raise ValueError(f"Could not reduce '{name}' tensor (original shape {value.shape}) to a scalar.")
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert '{name}'={value} (type {type(value)}) to float scalar. Error: {e}")

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

    def _compute_prediction_to_store(self, raw_head_output, prediction_strategy, quantile_levels, output_head):
        levels = self._normalize_levels(quantile_levels)
        if isinstance(raw_head_output, list):
            per_head = []
            for h, y in zip(self.output_heads, raw_head_output):
                if prediction_strategy and hasattr(h, "predict"):
                    per_head.append(h.predict(y, method=prediction_strategy))
                elif levels and hasattr(h, "sample_quantiles"):
                    per_head.append(h.sample_quantiles(y, quantile_levels=levels))
                else:
                    per_head.append(y)
            return per_head
        y = raw_head_output
        if prediction_strategy and hasattr(output_head, "predict"):
            return output_head.predict(y, method=prediction_strategy)
        if levels and hasattr(output_head, "sample_quantiles"):
            return output_head.sample_quantiles(y, quantile_levels=levels)
        return y

    def _compute_next_decoder_input_value(self, raw_head_output, prediction_strategy, output_head):
        """
        Determines the single, collapsed value to feed back into the decoder.
        """
        feedback_source = raw_head_output
        if isinstance(raw_head_output, List):
            logger.warning("Multiple output heads detected. Defaulting to the output of the first head for autoregressive feedback.")
            feedback_source = raw_head_output[0]

        if hasattr(output_head, "predict"):
            return output_head.predict(feedback_source, method=prediction_strategy or "mean")
        elif hasattr(output_head, "sample_quantiles"):
            q = prediction_strategy if isinstance(prediction_strategy, float) else 0.5
            return output_head.sample_quantiles(feedback_source, quantile_levels=[q]).squeeze(-1)
        
        elif isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4:
                return feedback_source.mean(dim=-1)
            elif feedback_source.ndim == 3:
                expected_features = getattr(self.config, "feature_size", 1)
                if feedback_source.shape[-1] != expected_features:
                    logger.warning(
                        f"Feedback tensor has {feedback_source.shape[-1]} features, but model expects {expected_features}. "
                        "Assuming this is a quantile dimension and taking the mean for feedback."
                    )
                    return feedback_source.mean(dim=-1, keepdim=True)
                return feedback_source
            else:
                raise ValueError(f"Unexpected feedback tensor ndim: {feedback_source.ndim}")
        else:
            raise TypeError(f"Unhandled feedback source type: {type(feedback_source)}")

    @torch.no_grad()
    def generate(
        self, encoder_inputs=None, decoder_inputs=None, prediction_length=None, attention_mask=None,
        decoder_attention_mask=None, use_cache=True, decoder_start_token_id=None, eos_token_id=None,
        early_stopping=False, output_attentions=False, output_hidden_states=False,
        prediction_strategy=None, quantile_levels=None, validate_shapes=True, verbose=True, **kwargs
    ):
        self.eval()
        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")
        if prediction_length is None:
            prediction_length = getattr(self.config, 'prediction_length', 0)
        if prediction_length == 0:
            return torch.empty(0)

        ref_tensor = decoder_inputs if encoder_inputs is None else encoder_inputs
        batch_size, device, dtype = ref_tensor.shape[0], ref_tensor.device, ref_tensor.dtype

        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder and encoder_inputs is not None:
            processed_encoder = self.preprocessor.process(input_values=encoder_inputs, attention_mask=attention_mask, is_causal=False)
            encoder_outputs = self.encoder(hidden_states=processed_encoder["hidden_states"], attention_mask=processed_encoder["attention_mask"], return_dict=True)
            encoder_hidden_states = encoder_outputs.last_hidden_state
        
        primary_output_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
        
        if decoder_inputs is None:
            start_val = self._get_scalar_value(decoder_start_token_id or getattr(self.config, "decoder_start_token_id", 0), "decoder_start_token_id")
            decoder_inputs = torch.full((batch_size, 1, self.config.feature_size), start_val, device=device, dtype=dtype)
        
        past_key_values, next_input = None, decoder_inputs
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        # Dummy step to infer output shape for pre-allocation
        _processed_decoder = self.preprocessor.process(input_values=next_input, past_key_values_length=0, is_causal=True)
        _decoder_outputs = self.decoder(hidden_states=_processed_decoder["hidden_states"], encoder_hidden_states=encoder_hidden_states, return_dict=True)
        _last_hidden = _decoder_outputs.last_hidden_state
        if _last_hidden.shape[1] > 1:
            _last_hidden = _last_hidden.mean(dim=1, keepdim=True)
        _raw_head_output = self._get_head_output(_last_hidden)
        _prediction_to_store = self._compute_prediction_to_store(_raw_head_output, prediction_strategy, quantile_levels, primary_output_head)
        
        all_predictions = torch.zeros((batch_size, prediction_length, *_prediction_to_store.shape[2:]), device=device, dtype=_prediction_to_store.dtype)

        for i in range(prediction_length):
            # FIX: Corrected the call to be self._get_cache_length(past_key_values)
            processed_decoder = self.preprocessor.process(input_values=next_input, past_key_values_length=self._get_cache_length(past_key_values), is_causal=True)
            
            decoder_outputs = self.decoder(
                hidden_states=processed_decoder["hidden_states"], attention_mask=processed_decoder["attention_mask"],
                encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values, use_cache=use_cache, return_dict=True
            )
            last_hidden = decoder_outputs.last_hidden_state

            if last_hidden.shape[1] > 1:
                last_hidden = last_hidden.mean(dim=1, keepdim=True)

            current_step_raw_head_output = self._get_head_output(last_hidden)
            prediction_to_store = self._compute_prediction_to_store(current_step_raw_head_output, prediction_strategy, quantile_levels, primary_output_head)
            all_predictions[:, i] = prediction_to_store.squeeze(1)
            
            next_input = self._compute_next_decoder_input_value(current_step_raw_head_output, prediction_strategy, primary_output_head)
            
            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None and torch.isclose(next_input.squeeze(), torch.tensor(eos_value_scalar, device=device)).all():
                logger.info(f"Early stopping at step {i + 1}.")
                all_predictions = all_predictions[:, :i+1]
                break
        
        final_predictions = all_predictions
        if hasattr(self.preprocessor, 'denormalize'):
            final_predictions = self.preprocessor.denormalize(final_predictions)
        return final_predictions

    def forecast(self, inputs, prediction_length, quantiles=None, **kwargs):
        """
        A user-friendly wrapper for the `generate` method, tailored for forecasting tasks.
        """
        if hasattr(self, 'encoder') and self.encoder is not None:
            return self.generate(encoder_inputs=inputs, prediction_length=prediction_length, quantile_levels=quantiles, **kwargs)
        else:
            return self.generate(decoder_inputs=inputs, prediction_length=prediction_length, quantile_levels=quantiles, **kwargs)