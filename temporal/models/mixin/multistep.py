import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MultistepMixin:
    """
    Multistep autoregressive generation with first-class collection/stacking.

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
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
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
        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads, required for autoregressive generation.")
        if isinstance(self.output_heads, nn.ModuleList):
            return [head(last_hidden) for head in self.output_heads]
        return self.output_heads(last_hidden)

    def _normalize_levels(self, quantile_levels):
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _ensure_b1f(self, x: torch.Tensor, feature_size: int) -> torch.Tensor:
        if x.ndim == 4:  # [B,1,F,K] -> mean K
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

    def _get_primary_head(self) -> nn.Module:
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    # --------------------------- sampling helpers -----------------------------

    def _call_head_sample(
        self,
        output_head: nn.Module,
        head_output: Union[torch.Tensor, Dict[str, Any], List[Any]],
        *,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        sampling_kwargs = dict(sampling_kwargs or {})
        if not hasattr(self, "_gen_state") or self._gen_state is None:
            self._gen_state = {}
        if hasattr(output_head, "sample") and callable(getattr(output_head, "sample")):
            # If the head.sample signature includes 'state', pass our state
            if "state" in getattr(output_head.sample, "__code__", None).co_varnames:
                sampling_kwargs.setdefault("state", self._gen_state)
            out = output_head.sample(head_output, **sampling_kwargs)
        else:
            raise AttributeError(f"{output_head.__class__.__name__} does not implement .sample(...)")

        new_state = None
        if isinstance(out, tuple) and len(out) == 2:
            y, new_state = out
        else:
            y = out
        if isinstance(new_state, dict):
            self._gen_state.update(new_state)
        y = self._ensure_b1f(y, getattr(self.config, "feature_size", y.shape[-1]))
        return y, new_state

    # --------------------------- store vs feedback ----------------------------

    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module,
    ) -> Union[torch.Tensor, Dict[str, Any], List[torch.Tensor]]:
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
                    # unwrap dicts (e.g., DistPred) if needed
                    y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
                    qy = h.sample_quantiles(y_in, quantile_levels=levels)
                    per_head.append(qy)
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
        if levels is not None and hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
            return output_head.sample_quantiles(y_in, quantile_levels=levels)
        return y

    def _compute_next_decoder_input_value(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        output_head: nn.Module,
        *,
        use_sampling: bool,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        feedback_source = raw_head_output
        if isinstance(raw_head_output, list):
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                feedback_source = self.head_aggregator(raw_head_output)
                if use_sampling and hasattr(output_head, "sample") and callable(getattr(output_head, "sample")):
                    logger.warning("Aggregator used; sampling from primary head only.")
            else:
                logger.warning("Multiple heads but no 'head_aggregator'. Using head[0] for feedback.")
                feedback_source = raw_head_output[0]

        if use_sampling and hasattr(output_head, "sample") and callable(getattr(output_head, "sample")):
            y, _ = self._call_head_sample(output_head, feedback_source, sampling_kwargs=sampling_kwargs)
            return self._ensure_b1f(y, self.config.feature_size)

        if hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
            method = prediction_strategy if prediction_strategy is not None else "median"
            out = output_head.predict(feedback_source, method=method)
            return self._ensure_b1f(out, self.config.feature_size)

        if hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            q = prediction_strategy if isinstance(prediction_strategy, float) else 0.5
            y_in = feedback_source["paths"] if isinstance(feedback_source, dict) and "paths" in feedback_source else feedback_source
            out = output_head.sample_quantiles(y_in, quantile_levels=[q])
            if torch.is_tensor(out) and out.ndim == 4:
                out = out.squeeze(-1)
            return self._ensure_b1f(out, self.config.feature_size)

        if isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4:
                feedback_source = feedback_source.mean(dim=-1)
            return self._ensure_b1f(feedback_source, self.config.feature_size)

        if isinstance(feedback_source, dict):
            raise TypeError("Head returning a dict must implement .sample() or .predict() to provide AR feedback.")
        raise TypeError(f"Unhandled feedback_source type: {type(feedback_source)}")

    # --------------------------- collectors / stacking ------------------------

    @staticmethod
    def _stack_distpred_steps(steps: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Stack a list of per-step DistPred dicts into time-major tensors.

        Args:
            steps: list where each item has 'paths': [B,1,F,K] (or [B,1,K])
                   and optional 'path_logits': [B,1,K] or [B,K]

        Returns:
            paths_BTFK:  [B,T,F,K]
            logits_BTK:  [B,T,K] or None
        """
        if not steps:
            raise ValueError("Empty steps list for DistPred stacking.")

        paths_list, logits_list = [], []
        for d in steps:
            p = d["paths"]  # [B,1,F,K] or [B,1,K]
            if p.ndim == 3:       # -> [B,1,1,K]
                p = p.unsqueeze(-2)
            paths_list.append(p)
            if "path_logits" in d and d["path_logits"] is not None:
                pl = d["path_logits"]  # [B,1,K] or [B,K]
                if pl.ndim == 2:
                    pl = pl.unsqueeze(1)  # -> [B,1,K]
                logits_list.append(pl)

        paths_BTFK = torch.cat(paths_list, dim=1)
        logits_BTK = torch.cat(logits_list, dim=1) if logits_list else None
        return paths_BTFK, logits_BTK

    # --------------------------- main API ------------------------------------

    @torch.no_grad()
    def generate_multistep(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
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
        # New: how to package multistep outputs
        collect: str = "auto",   # {'auto','none','per_step','distpred'}
        **kwargs,
    ) -> Union[
        torch.Tensor,
        List[Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor]
    ]:
        """
        Autoregressively generate a sequence and *collect* per-step outputs.

        collect:
          - 'none'      -> return the usual tensor (or list-of-dicts if head forces it)
          - 'per_step'  -> always return list of per-step items (dicts or tensors)
          - 'distpred'  -> if head returns dicts with 'paths', stack into {'paths':[B,T,F,K], 'path_logits':[B,T,K]?}
          - 'auto'      -> if head returns dicts, stack DistPred automatically, else tensor
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

        # Encoder (optional)
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

        # Primary head
        primary_output_head = self._get_primary_head()

        # Seed prompt
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                raise ValueError("Decoder-only generation needs a real prompt in decoder_inputs; avoid constant start tokens.")

        # Warm step to infer output shape for *stored* outputs (tensor path)
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

        if store_sampled and sampling and hasattr(primary_output_head, "sample"):
            warm_sample_b1f, _ = self._call_head_sample(primary_output_head, warm_head, sampling_kwargs=sampling_kwargs)
            warm_pred = warm_sample_b1f  # [B,1,F]
        else:
            warm_pred = self._compute_prediction_to_store(
                warm_head, prediction_strategy, quantile_levels, primary_output_head
            )

        # Decide initial collection containers
        predictions_is_tensor = torch.is_tensor(warm_pred)
        if predictions_is_tensor:
            step_out_shape = warm_pred.shape[2:]
            all_predictions = torch.zeros(
                (batch_size, prediction_length, *step_out_shape),
                device=warm_pred.device,
                dtype=warm_pred.dtype,
            )
        per_step_items: List[Any] = []  # we always capture per-step too for possible 'auto'/'distpred'

        # AR loop
        past_key_values = None
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")
        next_input = decoder_inputs

        if not hasattr(self, "_gen_state") or self._gen_state is None:
            self._gen_state = {}

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
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]  # [B,1,D]

            # Head output for this step
            head_out = self._get_head_output(last_hidden)

            # Feedback (sample or predict)
            next_decoder_input_value = self._compute_next_decoder_input_value(
                head_out, prediction_strategy, primary_output_head,
                use_sampling=sampling, sampling_kwargs=sampling_kwargs,
            )
            next_input = next_decoder_input_value

            # What we *store* (tensor path) + we also keep per-step raw
            if store_sampled and torch.is_tensor(next_decoder_input_value):
                pred_to_store = next_decoder_input_value
            else:
                pred_to_store = self._compute_prediction_to_store(
                    head_out, prediction_strategy, quantile_levels, primary_output_head
                )

            # Accumulate tensor path
            if predictions_is_tensor and torch.is_tensor(pred_to_store):
                all_predictions[:, i] = pred_to_store.squeeze(1)
            else:
                predictions_is_tensor = False  # fallback to per-step collection
            # Always keep the raw per-step item for possible stacking later
            per_step_items.append(head_out if isinstance(head_out, dict) else pred_to_store)

            # Cache & early stop
            if use_cache:
                past_key_values = decoder_outputs.past_key_values
            if early_stopping and eos_value_scalar is not None:
                target = torch.full_like(next_decoder_input_value, eos_value_scalar)
                if torch.allclose(next_decoder_input_value, target, rtol=0.0, atol=1e-6):
                    logger.info(f"Early stopping at step {i + 1}.")
                    if predictions_is_tensor:
                        all_predictions = all_predictions[:, :i+1]
                    per_step_items = per_step_items[:i+1]
                    break

        # Decide packaging
        if collect == "none":
            # same behavior as stepwise generate
            if predictions_is_tensor:
                final = all_predictions
            else:
                # If first item is dict, return list of dicts; else cat tensors
                if isinstance(per_step_items[0], dict):
                    final = per_step_items
                else:
                    final = torch.cat(per_step_items, dim=1)
        elif collect == "per_step":
            final = per_step_items
        elif collect in ("distpred", "auto"):
            # If we have dict steps with 'paths', stack into dict; else fall back.
            if isinstance(per_step_items[0], dict) and "paths" in per_step_items[0]:
                paths_BTFK, logits_BTK = self._stack_distpred_steps(per_step_items)
                out_dict = {"paths": paths_BTFK}
                if logits_BTK is not None:
                    out_dict["path_logits"] = logits_BTK
                final = out_dict
            else:
                # fall back to tensor or per-step depending on what we have
                if predictions_is_tensor:
                    final = all_predictions
                else:
                    final = per_step_items
        else:
            raise ValueError(f"Unknown collect mode: {collect!r}")

        # Denormalize when appropriate and requested
        if isinstance(final, torch.Tensor):
            last_dim = final.size(-1) if final.dim() >= 2 else None
            can_denorm = (last_dim == getattr(self.config, "feature_size", last_dim))
            if hasattr(self.preprocessor, 'denormalize') and can_denorm and not return_raw:
                logger.info("Denormalizing final predictions.")
                final = self.preprocessor.denormalize(final)

        return final

    @torch.no_grad()
    def forecast_multistep(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Convenience wrapper:
          - Encoder-Decoder: pass inputs as encoder_inputs.
          - Decoder-Only: pass inputs as decoder_inputs (prompt).
        """
        if hasattr(self, 'encoder') and self.encoder is not None:
            return self.generate_multistep(
                encoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                **kwargs,
            )
        else:
            return self.generate_multistep(
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                **kwargs,
            )
