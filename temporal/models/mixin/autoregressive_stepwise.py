import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class AutoregressiveStepwiseMixin:
    """
    A mixin class for autoregressive generation capabilities in neural network models.
    Assumes the host model defines:
      - self.config with .feature_size, .hidden_size (or decoder hidden dim),
        optionally .prediction_length, .num_attention_heads
      - self.preprocessor with .process(...) and optional .denormalize(...)
      - self.decoder (and optional self.encoder)
      - self.output_heads (nn.Module or nn.ModuleList)
      - (optional) self.head_aggregator for multi-head aggregation
    """

    # --------------------------- KV cache utilities ---------------------------

    def _get_cache_length(self, past_key_values) -> int:
        """
        Robustly infer the cached sequence length L from past_key_values.
        Supports common layouts: (B,H,L,D), (B,L,H,D), and (B,L,D).
        """
        if past_key_values is None:
            return 0
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)):
            key_tensor = first_layer[0]
        elif isinstance(first_layer, dict):
            key_tensor = first_layer.get("k", None)
        else:
            key_tensor = None
        if key_tensor is None:
            raise ValueError(f"Cannot locate key tensor in past_key_values[0]: {type(first_layer)}")

        if key_tensor.ndim == 4:  # (B,H,L,D) or (B,L,H,D)
            nh = getattr(self.config, "num_attention_heads", None)
            if nh is not None:
                if key_tensor.shape[1] == nh:   # (B,H,L,D)
                    return int(key_tensor.shape[2])
                if key_tensor.shape[2] == nh:   # (B,L,H,D)
                    return int(key_tensor.shape[1])
            # Fallback: treat the larger middle dim as length
            return int(max(key_tensor.shape[1], key_tensor.shape[2]))

        if key_tensor.ndim == 3:   # (B,L,D)
            return int(key_tensor.shape[1])

        raise ValueError(f"Unexpected KV shape: {tuple(key_tensor.shape)}")

    # --------------------------- misc helpers ---------------------------------

    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """Safely convert a value or tensor to a float scalar (or None)."""
        if value is None:
            return None
        if torch.is_tensor(value):
            temp_value = value
            while temp_value.numel() > 1:
                logger.warning(f"Tensor for '{name}' had {temp_value.numel()} elements. Taking the first element.")
                temp_value = temp_value[0]
            if temp_value.numel() == 1:
                return float(temp_value.item())
            raise ValueError(f"Could not reduce '{name}' tensor (original shape {value.shape}) to a scalar.")
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert '{name}'={value} (type {type(value)}) to float scalar. Error: {e}")

    def _get_head_output(self, last_hidden: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        """Apply output head(s) to the last hidden state."""
        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads, required for autoregressive generation.")
        if isinstance(self.output_heads, nn.ModuleList):
            return [head(last_hidden) for head in self.output_heads]
        return self.output_heads(last_hidden)

    def _normalize_levels(self, quantile_levels):
        """Validate and sort quantile levels in (0,1) if provided."""
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _ensure_b1f(self, x: torch.Tensor, feature_size: int) -> torch.Tensor:
        """
        Ensure feedback tensor has shape [B, 1, F]. Avoid broadcasting a scalar unless F==1.
        Accepts [B,1,F,K]/[B,1,F]/[B,1,1]/[B,F].
        """
        if x.ndim == 4:  # [B,1,F,K] -> mean over K
            x = x.mean(dim=-1)
        if x.ndim == 3:  # [B,1,F] or [B,1,1]
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        if x.ndim == 2:  # [B,F] -> [B,1,F]
            x = x.unsqueeze(1)
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        if x.ndim == 1:  # [B] -> [B,1,1] (then maybe expand)
            x = x.view(x.shape[0], 1, 1)
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        raise ValueError(f"Expected feedback tensor with 1-4 dims, got {x.shape}")

    # --------------------------- sampling helpers -----------------------------

    def _get_primary_head(self) -> nn.Module:
        """Return the primary head for feedback sampling."""
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    def _call_head_sample(
        self,
        output_head: nn.Module,
        head_output: Union[torch.Tensor, Dict[str, Any], List[Any]],
        *,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Call head.sample(...) if available.
        - If sample returns (y, new_state), we propagate/merge new_state into self._gen_state.
        - Returns (y_b1f, updated_state_or_None).
        """
        sampling_kwargs = dict(sampling_kwargs or {})
        # Ensure a persistent gen state exists for heads that need it (e.g., DistPred)
        if not hasattr(self, "_gen_state") or self._gen_state is None:
            self._gen_state = {}

        # If the head's sample accepts 'state', pass it through.
        if "state" in getattr(output_head.sample, "__code__", None).co_varnames:
            sampling_kwargs.setdefault("state", self._gen_state)

        out = output_head.sample(head_output, **sampling_kwargs)  # could be Tensor or (Tensor, dict)
        new_state = None
        if isinstance(out, tuple) and len(out) == 2:
            y, new_state = out
        else:
            y = out

        # Merge/replace state if provided
        if isinstance(new_state, dict):
            # shallow merge is enough; heads use distinct keys
            self._gen_state.update(new_state)

        y = self._ensure_b1f(y, getattr(self.config, "feature_size", y.shape[-1]))
        return y, new_state

    # --------------------------- store vs feedback ----------------------------

    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module,  # primary head if ModuleList
    ) -> Union[torch.Tensor, Dict[str, Any], List[torch.Tensor]]:
        """
        Decide what to store for this AR step (DEFAULT path, not using sample()):
          - If strategy given and head supports predict(): use it
          - Else if quantiles requested and head supports sample_quantiles(): sample them
          - Else store raw output.
        Works for single head or ModuleList.
        """
        levels = self._normalize_levels(quantile_levels)

        # Multi-head
        if isinstance(raw_head_output, list):
            assert isinstance(self.output_heads, nn.ModuleList), \
                "raw_head_output is a list but self.output_heads is not ModuleList."
            per_head = []
            for h, y in zip(self.output_heads, raw_head_output):
                if prediction_strategy is not None and hasattr(h, "predict") and callable(getattr(h, "predict")):
                    per_head.append(h.predict(y, method=prediction_strategy))
                elif levels is not None and hasattr(h, "sample_quantiles") and callable(getattr(h, "sample_quantiles")):
                    qy = h.sample_quantiles(y, quantile_levels=levels)
                    per_head.append(qy)
                else:
                    per_head.append(y)

            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                try:
                    return self.head_aggregator(per_head)  # aggregator must handle shapes
                except Exception as e:
                    logger.warning(f"head_aggregator failed during store; returning per-head list. Error: {e}")
                    return per_head
            return per_head

        # Single-head
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
        output_head: nn.Module,  # primary head if ModuleList
        *,
        use_sampling: bool,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Collapse head output into a single feedback value of shape [B, 1, F].
        Prefers head.sample() when use_sampling=True and the head supports it.
        """
        feedback_source = raw_head_output
        # If multiple heads and no aggregator, fallback to head[0]
        if isinstance(raw_head_output, list):
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                feedback_source = self.head_aggregator(raw_head_output)
                # If aggregator returns tensors, sampling is not defined — fall back below.
                if use_sampling and hasattr(output_head, "sample") and callable(getattr(output_head, "sample")):
                    logger.warning("Aggregator used; sampling from primary head only.")
            else:
                logger.warning(
                    "Multiple output heads detected without a 'head_aggregator'. "
                    "Using head[0] for autoregressive feedback."
                )
                feedback_source = raw_head_output[0]

        # Prefer sampling if available
        if use_sampling and hasattr(output_head, "sample") and callable(getattr(output_head, "sample")):
            y, _ = self._call_head_sample(output_head, feedback_source, sampling_kwargs=sampling_kwargs)
            return self._ensure_b1f(y, self.config.feature_size)

        # Else prefer .predict() if available; default to "median"
        if hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
            method = prediction_strategy if prediction_strategy is not None else "median"
            out = output_head.predict(feedback_source, method=method)
            return self._ensure_b1f(out, self.config.feature_size)

        # Fallback to quantiles if available
        if hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            q = prediction_strategy if isinstance(prediction_strategy, float) else 0.5
            out = output_head.sample_quantiles(feedback_source, quantile_levels=[q])  # [B,1,F,1] or [B,1,1]
            if torch.is_tensor(out) and out.ndim == 4:
                out = out.squeeze(-1)  # [B,1,F]
            return self._ensure_b1f(out, self.config.feature_size)

        # Raw tensor fallback
        if isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4:  # [B,1,F,K] -> mean K
                feedback_source = feedback_source.mean(dim=-1)
            return self._ensure_b1f(feedback_source, self.config.feature_size)

        if isinstance(feedback_source, dict):
            raise TypeError(
                "Output head returning a dict must implement .sample() or .predict() to provide a tensor for AR feedback."
            )
        raise TypeError(f"Unhandled feedback_source type: {type(feedback_source)}")

    # --------------------------- generation APIs ------------------------------

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,   # kept for signature compatibility
        eos_token_id: Optional[Any] = None,
        early_stopping: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        prediction_strategy: Optional[Union[str, float, int]] = None,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        *,
        # NEW — sampling controls
        sampling: bool = True,                          # use head.sample() for feedback if available
        sampling_kwargs: Optional[Dict[str, Any]] = None,  # passed into head.sample()
        store_sampled: bool = False,                    # if True, store the sampled feedback instead of predict()/quantiles/raw
        enable_mc_dropout: bool = False,                # flips dropout on during generation for diversity
        return_raw: bool = False,                       # bypass denorm
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Autoregressively generate a sequence of predictions.
        - sampling=True uses each head's `sample()` for feedback (anti-collapse).
        - store_sampled=True stores the *sampled trajectory*; otherwise uses predict()/quantiles/raw for storage.
        - sampling_kwargs (dict) are passed to head.sample(); e.g. {'temperature':1.1, 'top_p':0.9, 'stickiness':0.9}
        - enable_mc_dropout=True enables dropout layers for MC sampling.
        """
        self.eval()
        if enable_mc_dropout:
            self.enable_dropout()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        if prediction_length is None:
            prediction_length = getattr(self.config, 'prediction_length', 0)

        ref_tensor = decoder_inputs if encoder_inputs is None else encoder_inputs
        batch_size, device, dtype = ref_tensor.shape[0], ref_tensor.device, ref_tensor.dtype

        if prediction_length == 0:
            return torch.empty((batch_size, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)

        # Encoder pass (if present)
        encoder_outputs = None
        if hasattr(self, 'encoder') and self.encoder is not None and encoder_inputs is not None:
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
        encoder_hidden_states = encoder_outputs.last_hidden_state if encoder_outputs is not None else None

        # Select primary output head
        primary_output_head = self._get_primary_head()

        # Seed decoder prompt
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                # Avoid constant BOS; decoder-only needs a real prompt
                raise ValueError("Decoder-only generation needs a real prompt in decoder_inputs; avoid constant start tokens.")

        # -------------- preallocate output (shape inferred via a REAL step) --------------
        warm_step = decoder_inputs
        warm_mask = (
            decoder_attention_mask
            if decoder_attention_mask is not None
            else torch.ones(warm_step.size(0), warm_step.size(1), device=warm_step.device, dtype=torch.float32)
        )
        warm_proc = self.preprocessor.process(
            input_values=warm_step,
            past_key_values_length=0,
            attention_mask=warm_mask,
            is_causal=True,
            validate_shapes=validate_shapes,
            verbose=verbose,
        )
        warm_out = self.decoder(
            hidden_states=warm_proc["hidden_states"],
            attention_mask=warm_proc["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        warm_last = warm_out.last_hidden_state[:, -1:, :]  # [B,1,D]
        warm_head = self._get_head_output(warm_last)

        # Decide what one-step "store" looks like
        if store_sampled and sampling and hasattr(primary_output_head, "sample"):
            # produce a warm sample to infer shape
            warm_sample_b1f, _ = self._call_head_sample(primary_output_head, warm_head,
                                                        sampling_kwargs=sampling_kwargs)
            warm_pred = warm_sample_b1f  # [B,1,F]
        else:
            warm_pred = self._compute_prediction_to_store(
                warm_head, prediction_strategy, quantile_levels, primary_output_head
            )

        use_preallocation = torch.is_tensor(warm_pred)
        if use_preallocation:
            step_out_shape = warm_pred.shape[2:]  # after [B,1,...]
            all_predictions = torch.zeros(
                (batch_size, prediction_length, *step_out_shape),
                device=warm_pred.device,
                dtype=warm_pred.dtype,
            )
        else:
            predictions_list: List[Any] = []

        # -------------- AR loop --------------
        past_key_values = None
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")
        next_input = decoder_inputs  # first iter uses full prompt (primes KV); next iters use single token

        # Persistent per-sequence generation state (sticky paths, etc.)
        if not hasattr(self, "_gen_state") or self._gen_state is None:
            self._gen_state = {}

        for i in range(prediction_length):
            step_input = next_input
            past_kv_length = self._get_cache_length(past_key_values)

            # Build 2D decoder mask matching training behavior
            if decoder_attention_mask is not None:
                if decoder_attention_mask.dim() != 2 or decoder_attention_mask.size(0) != step_input.size(0):
                    raise ValueError("decoder_attention_mask must be [B, T_step].")
                dec2d_mask = decoder_attention_mask
            else:
                dec2d_mask = torch.ones(step_input.size(0), step_input.size(1), device=step_input.device, dtype=torch.float32)

            processed_decoder = self.preprocessor.process(
                input_values=step_input,
                past_key_values_length=past_kv_length,
                attention_mask=dec2d_mask,
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

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]  # [B, 1, D]

            # Heads (current step raw outputs)
            current_step_raw_head_output = self._get_head_output(last_hidden)

            # Feedback (prefer sampling)
            next_decoder_input_value = self._compute_next_decoder_input_value(
                current_step_raw_head_output,
                prediction_strategy,
                primary_output_head,
                use_sampling=sampling,
                sampling_kwargs=sampling_kwargs,
            )

            # Store output for this step
            if store_sampled and torch.is_tensor(next_decoder_input_value):
                # store the same sampled feedback
                pred_to_store = next_decoder_input_value
            else:
                pred_to_store = self._compute_prediction_to_store(
                    current_step_raw_head_output, prediction_strategy, quantile_levels, primary_output_head
                )

            # Accumulate
            if use_preallocation and torch.is_tensor(pred_to_store):
                all_predictions[:, i] = pred_to_store.squeeze(1)
            else:
                if use_preallocation and not torch.is_tensor(pred_to_store):
                    # switch to list mode if head returns non-tensor unexpectedly
                    use_preallocation = False
                    predictions_list = []
                    # backfill previous stored items from all_predictions
                    for j in range(i):
                        predictions_list.append(all_predictions[:, j:j+1])
                predictions_list.append(pred_to_store)

            # Prepare next input
            next_input = next_decoder_input_value  # [B,1,F]

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            # Early stopping (robust compare)
            if early_stopping and eos_value_scalar is not None:
                target = torch.full_like(next_decoder_input_value, eos_value_scalar)
                if torch.allclose(next_decoder_input_value, target, rtol=0.0, atol=1e-6):
                    logger.info(f"Early stopping triggered at step {i + 1}.")
                    if use_preallocation:
                        all_predictions = all_predictions[:, :i+1]
                    break

        # -------------- assemble final tensor --------------
        if use_preallocation:
            final_predictions = all_predictions
        else:
            if not predictions_list:
                return torch.empty((batch_size, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)
            if isinstance(predictions_list[0], dict):
                logger.info("Returning list of dicts; skipping concatenation and denormalization.")
                return predictions_list
            final_predictions = torch.cat(predictions_list, dim=1)

        # -------------- denormalize if compatible --------------
        if isinstance(final_predictions, torch.Tensor):
            last_dim = final_predictions.size(-1) if final_predictions.dim() >= 2 else None
            can_denorm = (last_dim == getattr(self.config, "feature_size", last_dim))
            if hasattr(self.preprocessor, 'denormalize') and can_denorm and not return_raw:
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
        User-friendly wrapper for generate():
          - Encoder-Decoder: pass inputs as encoder_inputs.
          - Decoder-Only: pass inputs as decoder_inputs (prompt).

        Common sampling usage:
            model.forecast(x, L, sampling=True,
                           sampling_kwargs={'temperature':1.1,'top_p':0.9},
                           store_sampled=True, enable_mc_dropout=True)
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
