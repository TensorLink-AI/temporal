import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveStepwiseMixin:
    """
    Stepwise autoregressive generation that returns a consistent ForecastBundle.

    This mixin assumes the host model provides:
      - self.config:
          • feature_size: int  (number of target features/channels)
          • hidden_size: int   (decoder hidden size)
          • prediction_length: Optional[int]
          • num_attention_heads: Optional[int]
      - self.preprocessor with:
          • process(input_values, attention_mask, past_key_values_length, is_causal, ...)
          • denormalize(tensor)           # optional; applied to point only
      - self.decoder (and optional self.encoder) compatible with HuggingFace-style outputs:
          • returns an object with .last_hidden_state and optionally .past_key_values
      - self.output_heads:
          • nn.Module or nn.ModuleList whose .forward(last_hidden) returns:
            - a tensor of params, or
            - a dict of params (e.g. MDN/DistPred), or
            - a list (if ModuleList) of the above per head
          • Heads should implement:
            - predict(params, method="median" | "mean" | float quantile | int index)
            - sample(params, **kwargs)                     # optional; used for AR feedback
            - sample_quantiles(params, quantile_levels)    # optional; used post-loop
      - optional: self._values_to_hidden(hidden_states) if your decoder expects a different space
      - optional: self.head_aggregator(list_of_head_outputs) -> head_output to use for feedback

    Key guarantees:
      * bundle.point is always a Tensor [B, T, F]
      * bundle.quantiles is either None or a Tensor [B, T, F, Q]
      * bundle.params is the stacked raw head outputs:
          - Tensor for Gaussian/StudentT/QuantileRegression/etc.
          - Dict[str, Tensor] for Mixture (MDN) or DistPred
      * No dicts are ever placed into bundle.quantiles (prevents `.shape` errors)
    """

    # ---------------------------------------------------------------------
    # Basic utilities
    # ---------------------------------------------------------------------
    def enable_dropout(self) -> None:
        """Enable dropout layers (for MC dropout)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_cache_length(self, past_key_values) -> int:
        """
        Infer current KV cache length L from a HuggingFace-style past_key_values
        structure by looking at the first layer's key tensor.

        Returns
        -------
        int
            Sequence length currently cached.
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

        # Common HF shapes: (B, H, L, D) or (B, L, H, D) or (B, L, D)
        if key_tensor.ndim == 4:
            nh = getattr(self.config, "num_attention_heads", None)
            if nh is not None:
                if key_tensor.shape[1] == nh:
                    return int(key_tensor.shape[2])   # (B, H, L, D)
                if key_tensor.shape[2] == nh:
                    return int(key_tensor.shape[1])   # (B, L, H, D)
            return int(max(key_tensor.shape[1], key_tensor.shape[2]))
        if key_tensor.ndim == 3:
            return int(key_tensor.shape[1])

        raise ValueError(f"Unexpected KV shape: {tuple(key_tensor.shape)}")

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """
        Convert a value to float if possible; reduce 0-d/1-d tensors to a scalar.

        Parameters
        ----------
        value : Tensor | float | int | Any
        name  : str

        Returns
        -------
        Optional[float]
        """
        if value is None:
            return None
        if torch.is_tensor(value):
            temp = value
            while temp.numel() > 1:
                logger.warning(f"Tensor for '{name}' had {temp.numel()} elements; taking the first.")
                temp = temp[0]
            if temp.numel() == 1:
                return float(temp.item())
            raise ValueError(f"Could not reduce '{name}' tensor of shape {value.shape} to scalar.")
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert '{name}'={value} (type {type(value)}) to float. Error: {e}")

    def _get_head_output(self, last_hidden: torch.Tensor) -> Union[torch.Tensor, List[Any], Dict[str, Any]]:
        """
        Run output head(s) on the last decoder hidden state.

        Returns
        -------
        Tensor | Dict[str, Tensor] | List[Tensor|Dict]
        """
        if not hasattr(self, "output_heads"):
            raise AttributeError("Model is missing output_heads, required for autoregressive generation.")
        if isinstance(self.output_heads, nn.ModuleList):
            return [head(last_hidden) for head in self.output_heads]
        return self.output_heads(last_hidden)

    def _normalize_levels(self, quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
        """
        Validate and sort requested quantile levels.

        Returns
        -------
        Optional[List[float]]
            None if input is None, else strictly increasing values in (0,1).
        """
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _ensure_b1f(self, x: torch.Tensor, feature_size: int) -> torch.Tensor:
        """
        Ensure feedback shape [B, 1, F] from common variants.

        Accepts
        -------
        [B, F]            -> [B, 1, F]
        [B, 1, F]         -> [B, 1, F]
        [B, 1, 1]         -> [B, 1, F] (broadcast if F>1)
        [B, 1, F, K]      -> mean over K -> [B, 1, F]

        Returns
        -------
        Tensor [B, 1, F]
        """
        if x.ndim == 4:  # [B,1,F,K]
            x = x.mean(dim=-1)
        if x.ndim == 3:
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        if x.ndim == 2:
            x = x.unsqueeze(1)
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        if x.ndim == 1:
            x = x.view(x.shape[0], 1, 1)
            if feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        raise ValueError(f"Expected feedback tensor with 1–4 dims, got {x.shape}")

    def _get_primary_head(self) -> nn.Module:
        """Return the first (primary) head."""
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    # ---------------------------------------------------------------------
    # Sampling helpers
    # ---------------------------------------------------------------------
    def _call_head_sample(
        self,
        output_head: nn.Module,
        head_output: Union[torch.Tensor, Dict[str, Any], List[Any]],
        *,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Call output_head.sample(...) with optional state threading.

        Returns
        -------
        (y_next[B,1,F], new_state: Optional[dict])
        """
        sampling_kwargs = dict(sampling_kwargs or {})
        if not hasattr(self, "_gen_state") or self._gen_state is None:
            self._gen_state = {}
        if hasattr(output_head, "sample"):
            code = getattr(output_head.sample, "__code__", None)
            if code and "state" in code.co_varnames:
                sampling_kwargs.setdefault("state", self._gen_state)

        out = output_head.sample(head_output, **sampling_kwargs)
        new_state = None
        if isinstance(out, tuple) and len(out) == 2:
            y, new_state = out
        else:
            y = out

        if isinstance(new_state, dict):
            # persist state across steps
            self._gen_state.update(new_state)

        y = self._ensure_b1f(y, getattr(self.config, "feature_size", y.shape[-1]))
        return y, new_state

    # ---------------------------------------------------------------------
    # Store vs feedback computation
    # ---------------------------------------------------------------------
    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[Any], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module,
        *,
        defer_quantiles: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any], List[Any]]:
        """
        Decide what to append to the *stored* trajectory at each step.

        If a head supports `sample_quantiles` and quantiles are requested (and not deferred),
        we store the quantile tensor. Otherwise we store the head's `predict(...)` or raw params.

        Returns
        -------
        Tensor | Dict[str, Tensor] | List[Tensor|Dict]
        """
        levels = self._normalize_levels(quantile_levels)

        # Multi-head case: operate per-head, then optionally aggregate
        if isinstance(raw_head_output, list):
            assert isinstance(self.output_heads, nn.ModuleList), \
                "raw_head_output is a list but self.output_heads is not ModuleList."
            per_head = []
            for h, y in zip(self.output_heads, raw_head_output):
                if prediction_strategy is not None and hasattr(h, "predict"):
                    per_head.append(h.predict(y, method=prediction_strategy))
                elif (levels is not None) and (not defer_quantiles) and hasattr(h, "sample_quantiles"):
                    y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
                    per_head.append(h.sample_quantiles(y_in, quantile_levels=levels))
                else:
                    per_head.append(y)
            if hasattr(self, "head_aggregator") and self.head_aggregator:
                try:
                    return self.head_aggregator(per_head)
                except Exception as e:
                    logger.warning(f"head_aggregator failed during store; returning per-head list. Error: {e}")
                    return per_head
            return per_head

        # Single-head
        y = raw_head_output
        if prediction_strategy is not None and hasattr(output_head, "predict"):
            return output_head.predict(y, method=prediction_strategy)
        if (levels is not None) and (not defer_quantiles) and hasattr(output_head, "sample_quantiles"):
            y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
            return output_head.sample_quantiles(y_in, quantile_levels=levels)
        return y

    def _compute_next_decoder_input_value(
        self,
        raw_head_output: Union[torch.Tensor, List[Any], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        output_head: nn.Module,
        *,
        use_sampling: bool,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Choose the feedback value fed to the next AR step.

        Preference order per head:
          1) .sample(...) if use_sampling
          2) .predict(..., method=prediction_strategy or 'median')
          3) .sample_quantiles(..., [0.5]) as a fallback
          4) raw tensor (possibly reduced) if head returns a tensor
        """
        feedback_source = raw_head_output
        if isinstance(raw_head_output, list):
            if hasattr(self, "head_aggregator") and self.head_aggregator:
                feedback_source = self.head_aggregator(raw_head_output)
                if use_sampling and hasattr(output_head, "sample"):
                    logger.warning("Aggregator used; sampling from primary head only.")
            else:
                logger.warning("Multiple output heads without 'head_aggregator'. Using head[0] for feedback.")
                feedback_source = raw_head_output[0]

        if use_sampling and hasattr(output_head, "sample"):
            y, _ = self._call_head_sample(output_head, feedback_source, sampling_kwargs=sampling_kwargs)
            return self._ensure_b1f(y, self.config.feature_size)

        if hasattr(output_head, "predict"):
            method = prediction_strategy if prediction_strategy is not None else "median"
            out = output_head.predict(feedback_source, method=method)
            return self._ensure_b1f(out, self.config.feature_size)

        if hasattr(output_head, "sample_quantiles"):
            q = prediction_strategy if isinstance(prediction_strategy, float) else 0.5
            out = output_head.sample_quantiles(feedback_source, quantile_levels=[q])  # [B,1,F,1] or [B,1,1,1]
            if torch.is_tensor(out) and out.ndim == 4:
                out = out.squeeze(-1)
            return self._ensure_b1f(out, self.config.feature_size)

        if isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4:  # e.g. [B,1,F,Q]
                feedback_source = feedback_source.mean(dim=-1)
            return self._ensure_b1f(feedback_source, self.config.feature_size)

        # Heads returning dict must implement .sample() or .predict()
        raise TypeError(
            "Output head returning a dict must implement .sample() or .predict() to provide a tensor for AR feedback."
        )

    # ---------------------------------------------------------------------
    # Param stacking and post-quantiles
    # ---------------------------------------------------------------------
    def _accum_params_dict_step(
        self,
        acc_tensors: Optional[Dict[str, List[torch.Tensor]]],
        acc_meta: Optional[Dict[str, Any]],
        step_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]:
        """
        Accumulate dict-based head outputs over time (for MDN/DistPred).

        Tensors are appended with T==1 enforced. Non-tensors (e.g. 'components') are
        stored once in metadata.
        """
        if acc_tensors is None:
            acc_tensors = {}
        if acc_meta is None:
            acc_meta = {}

        for k, v in step_dict.items():
            if torch.is_tensor(v):
                vv = v
                if vv.ndim >= 2 and vv.shape[1] != 1:
                    vv = vv[:, -1:, ...]  # keep the last step
                acc_tensors.setdefault(k, []).append(vv)
            else:
                acc_meta.setdefault(k, v)
        return acc_tensors, acc_meta

    def _stack_params_dict(
        self,
        acc_tensors: Dict[str, List[torch.Tensor]],
        acc_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Concatenate dict-of-lists along time and reattach meta keys."""
        out: Dict[str, Any] = {}
        for k, vs in acc_tensors.items():
            out[k] = torch.cat(vs, dim=1)  # [B, T, ...]
        if acc_meta:
            out.update(acc_meta)
        return out

    def _ensure_components_present(self, params: Dict[str, Any], head: nn.Module) -> Dict[str, Any]:
        """
        Ensure a MDN-like dict has a 'components' key when the head exposes it.
        """
        if "components" not in params and hasattr(head, "components"):
            try:
                params = dict(params)  # shallow copy
                params["components"] = list(getattr(head, "components"))
            except Exception:
                pass
        return params

    def _normalize_quantile_shape(self, q: torch.Tensor, *, feature_size: int, Q: int) -> torch.Tensor:
        """
        Normalize various quantile shapes to [B, T, F, Q].

        Accepts:
          • [B, T, Q]      -> [B, T, 1, Q]
          • [B, T, F, Q]   -> as-is
          • [B, T, Q, F]   -> [B, T, F, Q]
          • [B, Q]         -> [B, 1, 1, Q]
        """
        if q.ndim == 4:
            if q.shape[-1] == Q:
                return q
            if q.shape[-2] == Q:
                return q.permute(0, 1, 3, 2)
            return q
        if q.ndim == 3 and q.shape[-1] == Q:
            return q.unsqueeze(-2)
        if q.ndim == 2 and q.shape[-1] == Q:
            return q.unsqueeze(1).unsqueeze(2)
        raise ValueError(f"Cannot normalize quantile tensor of shape {tuple(q.shape)} to [B,T,F,Q].")

    def _post_quantiles_any(
        self,
        params_or_preds: Union[torch.Tensor, Dict[str, Any], List[Any]],
        head: nn.Module,
        quantile_levels: List[float],
    ) -> Union[torch.Tensor, Dict[str, Any], List[Any]]:
        """
        Compute quantiles once after the AR loop for tensor OR dict OR list-of-heads.

        Returns a tensor [B, T, F, Q] when the primary head supports quantiles.
        For multi-head scenarios, returns a list aligned with heads.
        """
        F = getattr(self.config, "feature_size", 1)
        Q = len(quantile_levels)

        if hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
            try:
                src = params_or_preds
                if isinstance(src, dict):
                    src = self._ensure_components_present(src, head)
                out = head.sample_quantiles(src, quantile_levels)
                if torch.is_tensor(out):
                    return self._normalize_quantile_shape(out, feature_size=F, Q=Q)
                return out
            except Exception as e:
                logger.warning(f"Post-quantiles on primary head failed; falling back. Error: {e}")

        # Multi-head: try per-head
        if isinstance(params_or_preds, list) and isinstance(self.output_heads, nn.ModuleList):
            out_list = []
            for sub_head, sub_y in zip(self.output_heads, params_or_preds):
                if hasattr(sub_head, "sample_quantiles") and callable(getattr(sub_head, "sample_quantiles")):
                    try:
                        src = sub_y
                        if isinstance(src, dict):
                            src = self._ensure_components_present(src, sub_head)
                        sub_out = sub_head.sample_quantiles(src, quantile_levels)
                        if torch.is_tensor(sub_out):
                            sub_out = self._normalize_quantile_shape(sub_out, feature_size=F, Q=Q)
                        out_list.append(sub_out)
                    except Exception as e:
                        logger.warning(f"Post-quantiles on subhead failed; keeping raw. Error: {e}")
                        out_list.append(sub_y)
                else:
                    out_list.append(sub_y)
            return out_list

        return params_or_preds

    def _extract_bundle_parts(
        self,
        head: nn.Module,
        params_stacked: Union[torch.Tensor, Dict[str, torch.Tensor]],
        quantile_levels: Optional[List[float]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Produce (point, quantiles, params_for_bundle) in a head-agnostic way.

        * For point:
            - Prefer head.predict(..., method='median'). Falls back to 'mean' if needed.
        * For quantiles:
            - If quantile_levels provided and head implements sample_quantiles(...),
              compute and normalize to [B,T,F,Q] (tensor only).
            - Special-case Quantile Regression: if params_stacked is already the quantile
              grid (last dim == Q), normalize and return it.
            - Otherwise return None.
        * Params:
            - Returned as `params_stacked` (Tensor for Gaussian/etc., Dict for MDN/DistPred).

        Returns
        -------
        (point[B,T,F], quantiles[B,T,F,Q] | None, params)
        """
        F = getattr(self.config, "feature_size", 1)

        # ----- point -----
        src_for_point: Union[torch.Tensor, Dict[str, torch.Tensor]]
        if isinstance(params_stacked, dict):
            src_for_point = self._ensure_components_present(params_stacked, head)
        else:
            src_for_point = params_stacked

        if hasattr(head, "predict"):
            try:
                point = head.predict(src_for_point, method="median")
            except Exception:
                point = head.predict(src_for_point, method="mean")
            if point.ndim == 2:
                point = point.unsqueeze(-1)  # [B,T] -> [B,T,1]
        else:
            raise TypeError("Output head must implement .predict(...) to provide point forecasts.")

        # ----- quantiles -----
        q_tensor: Optional[torch.Tensor] = None
        if quantile_levels:
            if hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
                q_raw = head.sample_quantiles(src_for_point, quantile_levels)
                if not torch.is_tensor(q_raw):
                    raise TypeError("sample_quantiles must return a Tensor for bundling.")
                q_tensor = self._normalize_quantile_shape(q_raw, feature_size=F, Q=len(quantile_levels))
            else:
                # QuantileRegression case: params already is the quantile grid
                if torch.is_tensor(params_stacked) and params_stacked.shape[-1] == len(quantile_levels):
                    q_tensor = self._normalize_quantile_shape(params_stacked, feature_size=F, Q=len(quantile_levels))

        return point, q_tensor, params_stacked

    def _compute_point_from_params(
        self,
        source: Union[torch.Tensor, Dict[str, Any], List[Any], None],
        head: nn.Module,
        *,
        quantiles_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Legacy helper: compute a point forecast either from a given quantile tensor
        (take the median slice) or by calling head.predict(...).
        """
        if isinstance(quantiles_tensor, torch.Tensor):
            mid = quantiles_tensor.shape[-1] // 2
            point = quantiles_tensor[..., mid]
            if point.ndim == 2:
                point = point.unsqueeze(-1)
            return point

        if source is not None and hasattr(head, "predict"):
            src = source
            if isinstance(src, dict):
                src = self._ensure_components_present(src, head)
            try:
                pv = head.predict(src, method="median")
            except Exception:
                pv = head.predict(src, method="mean")
            if pv.ndim == 2:
                pv = pv.unsqueeze(-1)
            return pv

        raise TypeError("Cannot compute point forecast from given params; add 'predict()' to this head.")

    # ---------------------------------------------------------------------
    # Generation API
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,  # compatibility only
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
        return_bundle: bool = False,
        **kwargs,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
        ForecastBundle,
        List[Dict[str, Union[torch.Tensor, List[str]]]],
    ]:
        """
        Stepwise autoregressive forecast.

        Parameters
        ----------
        encoder_inputs : Tensor | None
            Input sequence for encoder (if model has an encoder).
        decoder_inputs : Tensor | None
            Initial decoder prompt values [B, T0, F]. If None and encoder exists,
            uses the last encoder input step as the seed.
        prediction_length : int | None
            Number of autoregressive steps to generate. Defaults to config.prediction_length.
        attention_mask, decoder_attention_mask : Tensor | None
            Masks for encoder/decoder inputs.
        use_cache : bool
            Whether to thread past_key_values through the decoder.
        eos_token_id : Any | None
            If provided, enables an equality-based early stopping check on feedback values.
        prediction_strategy : str | float | int | None
            Strategy for head.predict() during feedback/storage if sampling is disabled:
              • "mean" or "median"
              • float in (0,1): quantile(e.g., 0.1 or 0.9)
              • int: take component/index when supported by the head (e.g., DistPred).
        quantile_levels : List[float] | None
            Requested quantiles to compute once after the AR loop.
        sampling : bool
            If True and the head implements .sample(...), feedback uses sampling.
        sampling_kwargs : dict | None
            Extra kwargs passed to head.sample(...), e.g., temperature/top_p/state.
        store_sampled : bool
            If True, the returned `point` is the actually sampled AR path.
        enable_mc_dropout : bool
            If True, puts dropout layers in train mode during eval for MC sampling.
        return_raw : bool
            If False and preprocessor.denormalize exists, denormalize the `point`.
        post_quantiles : bool
            If True, compute quantiles once on stacked params (preferred for MDN).
        return_bundle : bool
            If True, return a ForecastBundle; else return `quantiles` (if tensor) or `point`.

        Returns
        -------
        ForecastBundle | Tensor | Dict | list
            - ForecastBundle(point, quantiles, params, extras) if return_bundle=True
            - If not returning a bundle:
                • quantiles tensor [B,T,F,Q] when computed
                • else point tensor [B,T,F]
        """
        self.eval()
        if enable_mc_dropout:
            self.enable_dropout()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        if prediction_length is None:
            prediction_length = getattr(self.config, "prediction_length", 0)

        ref = decoder_inputs if encoder_inputs is None else encoder_inputs
        B, device, dtype = ref.shape[0], ref.device, ref.dtype

        if prediction_length == 0:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        # Optional encoder pass
        encoder_hidden_states = None
        if hasattr(self, "encoder") and self.encoder is not None and encoder_inputs is not None:
            enc_proc = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            enc_out = self.encoder(
                hidden_states=enc_proc["hidden_states"],
                attention_mask=enc_proc["attention_mask"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state

        # Primary head
        primary_head = self._get_primary_head()

        # Seed
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                raise ValueError("Decoder-only generation needs a real prompt in decoder_inputs.")

        # Warm step to infer storage shape
        warm_mask = (
            decoder_attention_mask
            if decoder_attention_mask is not None
            else torch.ones(decoder_inputs.size(0), decoder_inputs.size(1), device=decoder_inputs.device, dtype=torch.float32)
        )
        warm_proc = self.preprocessor.process(
            input_values=decoder_inputs,
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
        warm_last = warm_out.last_hidden_state[:, -1:, :]  # [B,1,D]
        warm_head = self._get_head_output(warm_last)

        defer_q = (post_quantiles and quantile_levels is not None)
        warm_pred = self._compute_prediction_to_store(
            warm_head, prediction_strategy, quantile_levels, primary_head, defer_quantiles=defer_q
        )

        # Preallocate if we are storing a tensor; otherwise use a list
        use_prealloc = torch.is_tensor(warm_pred)
        if use_prealloc:
            step_out_shape = warm_pred.shape[2:]  # drop T=1
            store_acc = torch.zeros(
                (B, prediction_length, *step_out_shape),
                device=warm_pred.device,
                dtype=warm_pred.dtype,
            )
        else:
            store_list: List[Any] = []
            store_acc = None

        sampled_steps: List[torch.Tensor] = []
        params_acc_tensor: Optional[torch.Tensor] = None
        params_acc_dict_tensors: Optional[Dict[str, List[torch.Tensor]]] = None
        params_acc_dict_meta: Optional[Dict[str, Any]] = None

        # AR loop
        past_key_values = None
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")
        next_input = decoder_inputs

        for i in range(prediction_length):
            step_input = next_input
            past_len = self._get_cache_length(past_key_values)

            if decoder_attention_mask is not None:
                if decoder_attention_mask.dim() != 2 or decoder_attention_mask.size(0) != step_input.size(0):
                    raise ValueError("decoder_attention_mask must be [B, T_step].")
                dec2d_mask = decoder_attention_mask
            else:
                dec2d_mask = torch.ones(step_input.size(0), step_input.size(1), device=step_input.device, dtype=torch.float32)

            dec_proc = self.preprocessor.process(
                input_values=step_input,
                past_key_values_length=past_len,
                attention_mask=dec2d_mask,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            dec_hidden_in = dec_proc["hidden_states"]
            if hasattr(self, "_values_to_hidden"):
                dec_hidden_in = self._values_to_hidden(dec_hidden_in)

            dec_out = self.decoder(
                hidden_states=dec_hidden_in,
                attention_mask=dec_proc["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            last_hidden = dec_out.last_hidden_state[:, -1:, :]  # [B,1,D]
            head_out = self._get_head_output(last_hidden)

            # Collect raw params for later bundling
            if torch.is_tensor(head_out):
                v = head_out if head_out.ndim >= 3 else head_out.unsqueeze(1)  # [B,1,...]
                params_acc_tensor = v if params_acc_tensor is None else torch.cat([params_acc_tensor, v], dim=1)
            elif isinstance(head_out, dict):
                params_acc_dict_tensors, params_acc_dict_meta = self._accum_params_dict_step(
                    params_acc_dict_tensors, params_acc_dict_meta, head_out
                )

            # Append stored trajectory (possibly deferring quantiles)
            to_store = self._compute_prediction_to_store(
                head_out, prediction_strategy, quantile_levels, primary_head, defer_quantiles=defer_q
            )
            if use_prealloc:
                if not torch.is_tensor(to_store):
                    # switch to list mode
                    use_prealloc = False
                    store_list = [store_acc[:, 0:1]] if store_acc is not None else []
                    store_acc = None
                    store_list.append(to_store)
                else:
                    store_acc[:, i] = to_store.squeeze(1)
            else:
                store_list.append(to_store)

            # Feedback for next step
            next_val = self._compute_next_decoder_input_value(
                head_out, prediction_strategy, primary_head,
                use_sampling=sampling, sampling_kwargs=sampling_kwargs
            )
            if store_sampled:
                sampled_steps.append(next_val)
            next_input = next_val

            if use_cache:
                past_key_values = dec_out.past_key_values

            # Optional early stop
            if early_stopping and eos_value_scalar is not None:
                target = torch.full_like(next_val, eos_value_scalar)
                if torch.allclose(next_val, target, rtol=0.0, atol=1e-6):
                    logger.info(f"Early stopping triggered at step {i + 1}.")
                    if use_prealloc and store_acc is not None:
                        store_acc = store_acc[:, : i + 1]
                    break

        # Assemble stored trajectory for legacy return paths
        if use_prealloc and store_acc is not None:
            stored = store_acc
        else:
            if store_list and isinstance(store_list[0], dict):
                stored = store_list
            elif store_list:
                stored = torch.cat(store_list, dim=1)
            else:
                stored = None

        # --------- stack params for bundle ---------
        params_stacked: Union[torch.Tensor, Dict[str, Any], None] = None
        if params_acc_tensor is not None:
            params_stacked = params_acc_tensor               # e.g., Gaussian/StudentT/QR => [B,T,...]
        elif params_acc_dict_tensors is not None:
            params_stacked = self._stack_params_dict(params_acc_dict_tensors, params_acc_dict_meta)  # MDN/DistPred dict

        # --------- point & quantiles (MDN-safe) ---------
        levels = self._normalize_levels(quantile_levels)

        if params_stacked is None:
            # Fallback: nothing accumulated (rare)
            if torch.is_tensor(stored) and stored.size(-1) == getattr(self.config, "feature_size", stored.size(-1)):
                point = stored
                q_tensor = None
                params_for_bundle = None
            else:
                raise RuntimeError("No head params were collected during generation.")
        else:
            point, q_tensor, params_for_bundle = self._extract_bundle_parts(primary_head, params_stacked, levels)

        # If requested, override point with the actually sampled AR path
        if store_sampled and sampled_steps:
            point = torch.cat(sampled_steps, dim=1)

        # Optional denormalization on point only
        if hasattr(self.preprocessor, "denormalize") and not return_raw:
            if isinstance(point, torch.Tensor) and point.size(-1) == getattr(self.config, "feature_size", point.size(-1)):
                point = self.preprocessor.denormalize(point)

        # Build bundle
        extras_dict: Dict[str, Any] = {}
        try:
            extras_obj = HeadExtras(**extras_dict)  # type: ignore[arg-type]
        except Exception:
            extras_obj = extras_dict  # type: ignore[assignment]

        bundle = ForecastBundle(
            point=point,                 # [B, T, F]
            quantiles=q_tensor,          # [B, T, F, Q] or None (always Tensor if present)
            params=params_for_bundle,    # Tensor for Gaussian/StudentT/QR; Dict for MDN/DistPred
            extras=extras_obj,
        )

        if return_bundle:
            return bundle
        if isinstance(q_tensor, torch.Tensor):
            return q_tensor
        return point

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Convenience wrapper around `generate` for encoder/decoder configurations.

        Parameters
        ----------
        inputs : Tensor
            Input sequence [B, T, F].
        prediction_length : int
            Number of steps to forecast.
        quantiles : List[float] | None
            Quantiles to compute post-hoc (e.g., [0.1, 0.5, 0.9]).
        """
        if hasattr(self, "encoder") and self.encoder is not None:
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
