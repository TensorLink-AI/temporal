import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Tuple, Callable, Dict, List

logger = logging.getLogger(__name__)

class AutoregressivePatchMixin:
    """
    A mixin class for autoregressive generation capabilities in neural network models.

    The primary `generate` method is designed for modern, patch-based models
    that use a "generate-then-merge" paradigm. It assumes the model's preprocessor
    has a dedicated method for handling latent embeddings during generation.

    The class also retains legacy helper methods for traditional, step-by-step
    autoregressive generation.
    """

    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    # ... [ _get_scalar_value, _get_head_output, and legacy helpers remain unchanged ] ...
    def _normalize_levels(self, quantile_levels):
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _safe_denormalize(self, x):
        if not hasattr(self.preprocessor, 'denormalize'):
            return x
        def _den(t):
            try:
                return self.preprocessor.denormalize(t)
            except Exception:
                return t
        if torch.is_tensor(x):
            return _den(x)
        if isinstance(x, dict):
            return {k: _den(v) if torch.is_tensor(v) else v for k, v in x.items()}
        if isinstance(x, list):
            out = []
            for v in x:
                if torch.is_tensor(v):
                    out.append(_den(v))
                elif isinstance(v, dict):
                    out.append({k: _den(t) if torch.is_tensor(t) else t for k, t in v.items()})
                else:
                    out.append(v)
            return out
        return x

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """
        Safely converts a potential tensor value to a float scalar.
        """
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

    def _get_head_output(
        self,
        head_input: torch.Tensor,
        prediction_strategy: Optional[Union[str, float, int]] = None,
        quantile_levels: Optional[List[float]] = None,
    ):
        """
        Project via head(s) first, then:
        - if prediction_strategy: use head.predict(...)
        - elif quantile_levels:   use head.sample_quantiles(...)
        - else:                   return point forecast (projected)
        Works for single head or ModuleList; supports optional self.head_aggregator.
        """
        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads, which is required for generation.")

        levels = self._normalize_levels(quantile_levels)

        # ---------- ModuleList ----------
        if isinstance(self.output_heads, nn.ModuleList):
            projected = [head(head_input) for head in self.output_heads]  # project all

            per_head_outputs = []
            for head, y in zip(self.output_heads, projected):
                if (prediction_strategy is not None) and hasattr(head, "predict") and callable(getattr(head, "predict")):
                    per_head_outputs.append(head.predict(y, method=prediction_strategy))
                elif (levels is not None) and hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
                    per_head_outputs.append(head.sample_quantiles(y, quantile_levels=levels))
                else:
                    per_head_outputs.append(y)

            # Optional aggregation (like stepwise)
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                try:
                    return self.head_aggregator(per_head_outputs)
                except Exception as e:
                    logger.warning(f"head_aggregator failed in patch mixin; returning per-head list. Error: {e}")
                    return per_head_outputs
            return per_head_outputs

        # ---------- Single head ----------
        head = self.output_heads
        projected = head(head_input)

        if (prediction_strategy is not None) and hasattr(head, "predict") and callable(getattr(head, "predict")):
            return head.predict(projected, method=prediction_strategy)

        if (levels is not None) and hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
            return head.sample_quantiles(projected, quantile_levels=levels)

        return projected

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = 0.0,
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
        Autoregressively generates a sequence of features, correctly designed for patch-based models.

        This process assumes the model's preprocessor has two distinct functions:
        1. `process()`: For handling the initial raw `encoder_inputs`.
        2. `_prepare_decoder_inputs_for_generation()`: For handling latent patch embeddings
           inside the autoregressive loop (adds positional encoding and creates causal masks).

        Args:
            encoder_inputs (torch.Tensor): The initial context sequence of raw time-series data.
            prediction_length (int): The number of time steps to predict. This will be
                converted to the number of patches to generate.
            attention_mask (Optional[torch.Tensor]): Mask for encoder attention.
            decoder_attention_mask (Optional[torch.Tensor]): Pre-computed mask for the decoder's
                self-attention. Typically not needed when using `use_cache=True`.
            use_cache (bool): Whether to use past key values for faster decoding.
            output_attentions (bool): Whether to return attentions from the model.
            output_hidden_states (bool): Whether to return hidden states from the model.
            **kwargs: Additional arguments passed to the encoder/decoder.

        Returns:
            The generated sequence of predicted features.
        """
        self.eval()

        # --- 1. Robust Initialization & Validation ---
        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        reference_tensor = decoder_inputs if encoder_inputs is None else encoder_inputs
        batch_size, device = reference_tensor.shape[0], reference_tensor.device

        patch_size = self.preprocessor.patch_size
        num_patches_to_generate = (prediction_length + patch_size - 1) // patch_size  # ceil-div

        # --- 2. Architectural Path: Prepare context and initial decoder state ---
        if hasattr(self, 'encoder') and self.encoder is not None and encoder_inputs is not None:
            # --- Encoder-Decoder Path ---
            processed_encoder = self.preprocessor.process(
                input_values=encoder_inputs, attention_mask=attention_mask, is_causal=False
            )
            context_patches = processed_encoder["hidden_states"]
            encoder_outputs = self.encoder(
                hidden_states=context_patches, attention_mask=processed_encoder["attention_mask"], return_dict=True,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # Start generation from the last patch of the context
            decoder_sequence_patches = context_patches[:, -1:, :]

        elif decoder_inputs is not None:
            # --- Decoder-Only Path ---
            processed_decoder_context = self.preprocessor.process(
                input_values=decoder_inputs, attention_mask=attention_mask, is_causal=True
            )
            context_patches = processed_decoder_context["hidden_states"]
            encoder_hidden_states = None # No encoder context to pass to the decoder
            
            # The entire prompt is the initial sequence for the decoder
            decoder_sequence_patches = context_patches
            
        else:
            raise ValueError("Could not determine model architecture. For encoder-decoder models, provide 'encoder_inputs'. For decoder-only, provide 'decoder_inputs'.")

        generated_patches = []
        past_key_values = None

        # --- Step 3: Autoregressively generate patch embeddings ---
        for _ in range(num_patches_to_generate):
            input_patches_for_step = decoder_sequence_patches[:, -1:, :] if use_cache and past_key_values else decoder_sequence_patches
            past_kv_length = past_key_values[0][0][0].shape[2] if past_key_values is not None else 0
            
            processed_decoder = self.preprocessor._prepare_decoder_inputs_for_generation(
                patch_embeds=input_patches_for_step,
                attention_mask=decoder_attention_mask if not (use_cache and past_key_values) else None,
                past_key_values_length=past_kv_length,
                is_causal=True,
            )
            
            decoder_outputs = self.decoder(
                hidden_states=processed_decoder["hidden_states"],
                attention_mask=processed_decoder["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            next_patch_embedding = decoder_outputs.last_hidden_state[:, -1:, :]
            generated_patches.append(next_patch_embedding)
            decoder_sequence_patches = torch.cat([decoder_sequence_patches, next_patch_embedding], dim=1)
            
            if use_cache:
                past_key_values = decoder_outputs.past_key_values       

        if not generated_patches:
            return torch.empty((reference_tensor.shape[0], 0, reference_tensor.shape[-1]), device=device)

        # --- Step 4: Merge generated patches into final predictions ---
        all_generated_patches = torch.cat(generated_patches, dim=1)
        
        reconstructed_output = self.output_patch_reconstructor(all_generated_patches)

        B, T_tok, _ = reconstructed_output.shape
        output_patch_size = self.preprocessor.patch_size
        d_model = self.config.d_model
        point_predictions = reconstructed_output.view(B, T_tok * output_patch_size, d_model)


        # --- Step 5: Apply final output heads (e.g., for probabilistic forecasts) ---
        final_output = self._get_head_output(
            point_predictions,
            prediction_strategy=prediction_strategy,
            quantile_levels=quantile_levels
        )
        
        # --- Step 6: Denormalize the final output if necessary ---
        if hasattr(self.preprocessor, 'denormalize'):
            logger.info("Denormalizing final patch-based predictions.")
            final_output = self.preprocessor.denormalize(final_output)
        
        # --- Step 7: Trim to the requested prediction length ---
        return final_output[:, :prediction_length, :]

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
            **kwargs: Additional arguments passed to the underlying `generate` method.
        """
        # This wrapper function is identical to the one from the stepwise mixin.
        # It inspects the model's architecture and calls `generate` correctly.
        if hasattr(self, 'encoder') and self.encoder is not None:
            # Encoder-Decoder Path
            logger.info("Encoder-Decoder model detected. Using `inputs` as `encoder_inputs` for forecasting.")
            return self.generate(
                encoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                **kwargs,
            )
        else:
            # Decoder-Only Path
            logger.info("Decoder-Only model detected. Using `inputs` as `decoder_inputs` for forecasting.")
            return self.generate(
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                **kwargs,
            )