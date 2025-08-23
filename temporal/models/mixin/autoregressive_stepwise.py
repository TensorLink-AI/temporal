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
        Accepts [B,1,F,K]/[B,1,F]/[B,1,1]/[B,F]/[B].
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
        if x.ndim == 1:  # [B] -> [B,1,1]
            x = x.view(x.shape[0], 1, 1)
            if feature_size > 1:
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
        if not hasattr(self, "_gen_state") or self._gen_state is None:
            self._gen_state = {}

        if hasattr(output_head, "sample") and callable(getattr(output_head, "sample")):
            # Best effort: if the head.sample signature includes 'state', pass it.
            try:
                if "state" in getattr(output_head.sample, "__code__", None).co_varnames:
                    sampling_kwargs.setdefault("state", self._gen_state)
            except Exception:
                pass

            out = output_head.sample(head_output, **sampling_kwargs)  # could be Tensor or (Tensor, dict)
            new_state = None
            if isinstance(out, tuple) and len(out) == 2:
                y, new_state = out
            else:
                y = out

            if isinstance(new_state, dict):
                self._gen_state.update(new_state)

            y = self._ensure_b1f(y, getattr(self.config, "feature_size", y.shape[-1]))
            return y, new_state

        raise AttributeError("output_head.sample(...) not found.")

    # --------------------------- store vs feedback ----------------------------

    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module,  # primary head if ModuleList
        *,
        defer_quantiles: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any], List[torch.Tensor]]:
        """
        Decide what to store for this AR step (DEFAULT path, not using sample()):
        - If strategy given and head supports predict(): use it
        - Else if quantiles requested and head supports sample_quantiles(): sample them
        - Else store raw output.

        When defer_quantiles=True, the quantile branch is skipped and we store raw outputs.
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
                elif (levels is not None) and (not defer_quantiles) and hasattr(h, "sample_quantiles") and callable(getattr(h, "sample_quantiles")):
                    y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
                    per_head.append(h.sample_quantiles(y_in, quantile_levels=levels))
                else:
                    per_head.append(y)

            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                try:
                    return self.head_aggregator(per_head)
                except Exception as e:
                    logger.warning(f"head_aggregator failed during store; returning per-head list. Error: {e}")
                    return per_head
            return per_head

        # Single-head
        y = raw_head_output
        if prediction_strategy is not None and hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
            return output_head.predict(y, method=prediction_strategy)
        if (levels is not None) and (not defer_quantiles) and hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
            return output_head.sample_quantiles(y_in, quantile_levels=levels)
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
            src = feedback_source["paths"] if isinstance(feedback_source, dict) and "paths" in feedback_source else feedback_source
            out = output_head.sample_quantiles(src, quantile_levels=[q])  # [B,1,F,1] or [B,1,1]
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

    # --------------------------- post-quantiles (robust) ----------------------

    def _post_quantiles_any(
        self,
        predictions: Union[torch.Tensor, Dict[str, Any], List[Any]],
        head: nn.Module,
        quantile_levels: List[float],
    ) -> Union[torch.Tensor, List[Any], Dict[str, Any]]:
        """
        Apply head.sample_quantiles post-loop regardless of predictions container type.
        - Tensor  -> head.sample_quantiles(tensor, qs)
        - Dict    -> head.sample_quantiles(dict,   qs)
        - List    -> per-head call (requires ModuleList), optionally aggregate
        Returns quantiles with same outer container structure.
        """
        qs = self._normalize_levels(quantile_levels)
        if qs is None:
            return predictions
        if not hasattr(head, "sample_quantiles") or not callable(getattr(head, "sample_quantiles")):
            raise TypeError("Primary head does not implement sample_quantiles().")

        # Tensor case
        if torch.is_tensor(predictions):
            return head.sample_quantiles(predictions, qs)

        # Dict case (e.g., mixture head)
        if isinstance(predictions, dict):
            return head.sample_quantiles(predictions, qs)

        # List case (multi-head)
        if isinstance(predictions, list):
            if not isinstance(self.output_heads, nn.ModuleList) or len(self.output_heads) != len(predictions):
                raise TypeError("Got a list of predictions but output_heads is not a matching ModuleList.")

            quantile_list: List[Any] = []
            for h, y in zip(self.output_heads, predictions):
                if hasattr(h, "sample_quantiles") and callable(getattr(h, "sample_quantiles")):
                    y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
                    quantile_list.append(h.sample_quantiles(y_in, qs))
                else:
                    # If a head can't produce quantiles post-hoc, keep raw
                    quantile_list.append(y)

            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                try:
                    return self.head_aggregator(quantile_list)
                except Exception as e:
                    logger.warning(f"head_aggregator failed during post-quantiles; returning per-head list. Error: {e}")
                    return quantile_list
            return quantile_list

        raise TypeError(f"Unsupported predictions container for post-quantiles: {type(predictions)}")

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
        decoder_start_token_id: Optional[Any] = None,
        eos_token_id: Optional[Any] = None,
        early_stopping: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        prediction_strategy: Optional[Union[str, float, int]] = None,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        *,
        sampling: bool = True,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        store_sampled: bool = False,
        enable_mc_dropout: bool = False,
        return_raw: bool = False,
        post_quantiles: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:

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

        # ----- encoder (optional) -----
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

        # primary head
        primary_output_head = self._get_primary_head() if hasattr(self, "_get_primary_head") else \
            (self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads)

        # seed decoder
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                raise ValueError("Decoder-only generation needs a real prompt in decoder_inputs; avoid constant start tokens.")

        # ---- warm step to infer output shape for preallocation ----
        warm_step = decoder_inputs
        warm_mask = decoder_attention_mask if decoder_attention_mask is not None \
            else torch.ones(warm_step.size(0), warm_step.size(1), device=warm_step.device, dtype=torch.float32)

        warm_proc = self.preprocessor.process(
            input_values=warm_step,
            past_key_values_length=0,
            attention_mask=warm_mask,
            is_causal=True,
            validate_shapes=validate_shapes,
            verbose=verbose,
        )
        warm_hidden_in = warm_proc["hidden_states"]
        if hasattr(self, "_values_to_hidden"):
            warm_hidden_in = self._values_to_hidden(warm_hidden_in)

        warm_out = self.decoder(
            hidden_states=warm_hidden_in,
            attention_mask=warm_proc["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        warm_last = warm_out.last_hidden_state[:, -1:, :]       # [B,1,D]
        warm_head = self._get_head_output(warm_last)

        warm_pred = self._compute_prediction_to_store(
            warm_head,
            prediction_strategy,
            quantile_levels,
            primary_output_head,
            defer_quantiles=(post_quantiles and quantile_levels is not None),
        )

        use_preallocation = torch.is_tensor(warm_pred)
        if use_preallocation:
            step_out_shape = warm_pred.shape[2:]  # [B,1,...] -> ...
            all_predictions = torch.zeros(
                (batch_size, prediction_length, *step_out_shape),
                device=warm_pred.device,
                dtype=warm_pred.dtype,
            )
        else:
            predictions_list: List[Any] = []

        # ---- AR loop ----
        past_key_values = None
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")
        next_input = decoder_inputs

        for i in range(prediction_length):
            step_input = next_input
            past_kv_length = self._get_cache_length(past_key_values)

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
            dec_hidden_in = processed_decoder["hidden_states"]
            if hasattr(self, "_values_to_hidden"):
                dec_hidden_in = self._values_to_hidden(dec_hidden_in)

            decoder_outputs = self.decoder(
                hidden_states=dec_hidden_in,
                attention_mask=processed_decoder["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]  # [B,1,D]

            # Heads
            current_step_raw_head_output = self._get_head_output(last_hidden)

            # Store (defer quantiles if requested)
            prediction_to_store = self._compute_prediction_to_store(
                current_step_raw_head_output,
                prediction_strategy,
                quantile_levels,
                primary_output_head,
                defer_quantiles=(post_quantiles and quantile_levels is not None),
            )

            if use_preallocation:
                if not torch.is_tensor(prediction_to_store):
                    use_preallocation = False
                    predictions_list = [prediction_to_store]
                else:
                    all_predictions[:, i] = prediction_to_store.squeeze(1)
            else:
                predictions_list.append(prediction_to_store)

            # Feedback (sampling or predict/median)
            next_decoder_input_value = self._compute_next_decoder_input_value(
                current_step_raw_head_output,
                prediction_strategy,
                primary_output_head,
                use_sampling=sampling,
                sampling_kwargs=sampling_kwargs,
            )
            next_input = next_decoder_input_value  # [B,1,F]

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            # Early stopping
            if early_stopping and eos_value_scalar is not None:
                target = torch.full_like(next_decoder_input_value, eos_value_scalar)
                if torch.allclose(next_decoder_input_value, target, rtol=0.0, atol=1e-6):
                    logger.info(f"Early stopping at step {i + 1}.")
                    if use_preallocation:
                        all_predictions = all_predictions[:, :i+1]
                    break

        # ---- assemble final ----
        if use_preallocation:
            final_predictions = all_predictions
        else:
            if not predictions_list:
                return torch.empty((batch_size, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)
            if isinstance(predictions_list[0], dict):
                # If head returns dict per step, we return a list of dicts (one per step)
                # In that case, post-quantiles below can still handle the dict if you
                # aggregated externally into a single dict; otherwise we keep list.
                # For simplicity, concatenate only tensors; for dicts, keep list.
                logger.info("Returning list of dicts; skipping concatenation.")
                final_predictions = predictions_list  # may still be handled by post-quantiles if you aggregate externally
            else:
                final_predictions = torch.cat(predictions_list, dim=1)

        # ---- post-hoc quantiles (universal) ----
        did_quantiles = False
        if post_quantiles and (quantile_levels is not None):
            try:
                if store_sampled:
                    # Typically not possible post-hoc
                    logger.info("store_sampled=True -> post-quantiles skipped (need raw params/paths).")
                else:
                    # If final_predictions is a list of per-step dicts, we can't pass a list directly.
                    # (You can aggregate those dicts outside; or change store path to keep a single dict.)
                    if isinstance(final_predictions, list):
                        logger.info("Final predictions is a list; post-quantiles expects tensor/dict or per-head list. Skipping.")
                    else:
                        final_predictions = self._post_quantiles_any(final_predictions, primary_output_head, quantile_levels)
                        did_quantiles = True
            except Exception as e:
                logger.warning(f"Post-quantiles failed; returning stored predictions. Error: {e}")

        # ---- optional denorm (skip if quantiles were produced) ----
        if isinstance(final_predictions, torch.Tensor) and not did_quantiles:
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
