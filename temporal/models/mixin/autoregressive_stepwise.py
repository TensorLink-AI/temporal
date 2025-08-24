import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Dict, List, Tuple

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveStepwiseMixin:
    """
    Autoregressive generation with a consistent ForecastBundle return option.

    Assumes the host model defines:
      - self.config with:
          .feature_size (int), .hidden_size (int or decoder dim),
          optionally .prediction_length, .num_attention_heads
      - self.preprocessor with:
          .process(values, attention_mask, past_key_values_length, is_causal, ...)
          optional .denormalize(tensor)
      - self.decoder (and optional self.encoder)
      - self.output_heads (nn.Module or nn.ModuleList)
      - (optional) self.head_aggregator for multi-head aggregation
      - (optional) self._values_to_hidden(hidden_states)
    """

    # --------------------- KV cache utilities ---------------------
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
                if key_tensor.shape[1] == nh:
                    return int(key_tensor.shape[2])
                if key_tensor.shape[2] == nh:
                    return int(key_tensor.shape[1])
            return int(max(key_tensor.shape[1], key_tensor.shape[2]))
        if key_tensor.ndim == 3:
            return int(key_tensor.shape[1])
        raise ValueError(f"Unexpected KV shape: {tuple(key_tensor.shape)}")

    # --------------------- misc helpers ---------------------------
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
        raise ValueError(f"Expected feedback tensor with 1–4 dims, got {x.shape}")

    def _get_primary_head(self) -> nn.Module:
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    # --------------------- sampling helpers -----------------------
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
            self._gen_state.update(new_state)
        y = self._ensure_b1f(y, getattr(self.config, "feature_size", y.shape[-1]))
        return y, new_state

    # --------------------- store vs feedback ----------------------
    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module,
        *,
        defer_quantiles: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any], List[torch.Tensor]]:
        levels = self._normalize_levels(quantile_levels)

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
                logger.warning("Multiple output heads without 'head_aggregator'. Using head[0] for feedback.")
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
            out = output_head.sample_quantiles(feedback_source, quantile_levels=[q])  # [B,1,F,1]
            if torch.is_tensor(out) and out.ndim == 4:
                out = out.squeeze(-1)
            return self._ensure_b1f(out, self.config.feature_size)

        if isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4:
                feedback_source = feedback_source.mean(dim=-1)
            return self._ensure_b1f(feedback_source, self.config.feature_size)

        if isinstance(feedback_source, dict):
            raise TypeError(
                "Output head returning a dict must implement .sample() or .predict() to provide a tensor for AR feedback."
            )
        raise TypeError(f"Unhandled feedback_source type: {type(feedback_source)}")

    # ------------------ post-quantiles & param stacking ------------------

    def _accum_params_dict_step(
        self,
        acc_tensors: Optional[Dict[str, List[torch.Tensor]]],
        acc_meta: Optional[Dict[str, Any]],
        step_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]:
        """
        Collect dict head outputs over time.
        - Tensor entries are appended (enforced [B,1,...]).
        - Non-tensor entries (e.g. 'components') are stored once in acc_meta.
        """
        if acc_tensors is None:
            acc_tensors = {}
        if acc_meta is None:
            acc_meta = {}

        for k, v in step_dict.items():
            if torch.is_tensor(v):
                vv = v
                if vv.ndim >= 2 and vv.shape[1] != 1:
                    vv = vv[:, -1:, ...]  # keep last step if needed
                acc_tensors.setdefault(k, []).append(vv)
            else:
                acc_meta.setdefault(k, v)  # store first occurrence
        return acc_tensors, acc_meta

    def _stack_params_dict(
        self,
        acc_tensors: Dict[str, List[torch.Tensor]],
        acc_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Time-stack a dict-of-lists into dict-of-[B,T,...] and reattach meta keys."""
        out: Dict[str, Any] = {}
        for k, vs in acc_tensors.items():
            out[k] = torch.cat(vs, dim=1)
        if acc_meta:
            out.update(acc_meta)
        return out

    def _ensure_components_present(
        self,
        params: Dict[str, Any],
        head: nn.Module
    ) -> Dict[str, Any]:
        """
        Make sure dict has a 'components' key if the head exposes it.
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
        Coerce any head's quantile tensor to [B, T, F, Q].
        Accepts common variants:
        - [B,T,Q]      -> [B,T,1,Q]
        - [B,T,F,Q]    -> as-is
        - [B,T,Q,F]    -> -> [B,T,F,Q]
        - [B,Q]        -> [B,1,1,Q]
        """
        if q.ndim == 4:
            if q.shape[-1] == Q:
                return q
            if q.shape[-2] == Q:
                return q.permute(0, 1, 3, 2)
            return q
        if q.ndim == 3:
            if q.shape[-1] == Q:
                return q.unsqueeze(-2)
            return q.unsqueeze(-2)
        if q.ndim == 2 and q.shape[-1] == Q:
            return q.unsqueeze(1).unsqueeze(2)
        if q.ndim == 4 and q.shape[-2] == feature_size:
            return q
        raise ValueError(f"Cannot normalize quantile tensor of shape {tuple(q.shape)} to [B,T,F,Q].")

    def _post_quantiles_any(
        self,
        params_or_preds: Union[torch.Tensor, Dict[str, Any], List[Any]],
        head: nn.Module,
        quantile_levels: List[float],
    ) -> Union[torch.Tensor, Dict[str, Any], List[Any]]:
        """
        Compute quantiles once after the AR loop for tensor OR dict OR list-of-heads,
        and normalize to [B,T,F,Q] when a tensor is returned.
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

        # Multi-head: try per-head and normalize tensors
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

    # ------------------ point (from quantiles or predict) ------------------
    def _compute_point_from_params(
        self,
        source: Union[torch.Tensor, Dict[str, Any], List[Any], None],
        head: nn.Module,
        *,
        quantiles_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(quantiles_tensor, torch.Tensor):
            mid = quantiles_tensor.shape[-1] // 2
            point = quantiles_tensor[..., mid]
            if point.ndim == 2:
                point = point.unsqueeze(-1)
            return point

        if source is not None and hasattr(head, "predict") and callable(getattr(head, "predict")):
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

    # ------------------ generation API ------------------
    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,   # compatibility only
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
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], ForecastBundle, List[Dict[str, Union[torch.Tensor, List[str]]]]]:

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
            empty = torch.empty((batch_size, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        # optional encoder pass
        encoder_hidden_states = None
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
            encoder_hidden_states = encoder_outputs.last_hidden_state

        # primary head
        primary_output_head = self._get_primary_head()

        # seed prompt
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                raise ValueError("Decoder-only generation needs a real prompt in decoder_inputs.")

        # warm step (infer storage shape)
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
            warm_head, prediction_strategy, quantile_levels, primary_output_head, defer_quantiles=defer_q
        )

        use_preallocation = torch.is_tensor(warm_pred)
        if use_preallocation:
            step_out_shape = warm_pred.shape[2:]
            store_acc = torch.zeros(
                (batch_size, prediction_length, *step_out_shape),
                device=warm_pred.device, dtype=warm_pred.dtype,
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
            head_out = self._get_head_output(last_hidden)

            # collect raw params for bundle
            if torch.is_tensor(head_out):
                v = head_out if head_out.ndim >= 3 else head_out.unsqueeze(1)
                params_acc_tensor = v if params_acc_tensor is None else torch.cat([params_acc_tensor, v], dim=1)
            elif isinstance(head_out, dict):
                params_acc_dict_tensors, params_acc_dict_meta = self._accum_params_dict_step(
                    params_acc_dict_tensors, params_acc_dict_meta, head_out
                )

            # store (possibly deferring quantiles)
            to_store = self._compute_prediction_to_store(
                head_out, prediction_strategy, quantile_levels, primary_output_head, defer_quantiles=defer_q
            )
            if use_preallocation:
                if not torch.is_tensor(to_store):
                    use_preallocation = False
                    store_list = [store_acc[:, 0:1]] if store_acc is not None else []
                    store_acc = None
                    store_list.append(to_store)
                else:
                    store_acc[:, i] = to_store.squeeze(1)
            else:
                store_list.append(to_store)

            # feedback
            next_decoder_input_value = self._compute_next_decoder_input_value(
                head_out, prediction_strategy, primary_output_head,
                use_sampling=sampling, sampling_kwargs=sampling_kwargs
            )
            if store_sampled:
                sampled_steps.append(next_decoder_input_value)
            next_input = next_decoder_input_value

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None:
                target = torch.full_like(next_decoder_input_value, eos_value_scalar)
                if torch.allclose(next_decoder_input_value, target, rtol=0.0, atol=1e-6):
                    logger.info(f"Early stopping triggered at step {i + 1}.")
                    if use_preallocation and store_acc is not None:
                        store_acc = store_acc[:, :i+1]
                    break

        # assemble stored (legacy)
        if use_preallocation and store_acc is not None:
            stored = store_acc
        else:
            if store_list and isinstance(store_list[0], dict):
                stored = store_list
            elif store_list:
                stored = torch.cat(store_list, dim=1)
            else:
                stored = None

        # stack params for bundle
        params_stacked: Union[torch.Tensor, Dict[str, Any], None] = None
        if params_acc_tensor is not None:
            params_stacked = params_acc_tensor
        elif params_acc_dict_tensors is not None:
            params_stacked = self._stack_params_dict(params_acc_dict_tensors, params_acc_dict_meta)

        # post-hoc quantiles (once)
        levels = self._normalize_levels(quantile_levels)
        quantiles_out: Optional[Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]]] = None
        if levels is not None and params_stacked is not None:
            quantiles_out = self._post_quantiles_any(params_stacked, primary_output_head, levels)

        # point forecast
        try:
            if store_sampled and sampled_steps:
                point = torch.cat(sampled_steps, dim=1)
            else:
                source_for_point = params_stacked
                point = self._compute_point_from_params(
                    source_for_point, primary_output_head,
                    quantiles_tensor=(quantiles_out if isinstance(quantiles_out, torch.Tensor) else None),
                )
        except Exception:
            if torch.is_tensor(stored) and stored.size(-1) == getattr(self.config, "feature_size", stored.size(-1)):
                point = stored
            else:
                raise

        # optional denorm (point only)
        if hasattr(self.preprocessor, 'denormalize') and not return_raw:
            if isinstance(point, torch.Tensor) and point.size(-1) == getattr(self.config, "feature_size", point.size(-1)):
                point = self.preprocessor.denormalize(point)

        # bundle
        extras_dict: Dict[str, Any] = {}
        try:
            extras_obj = HeadExtras(**extras_dict)  # type: ignore[arg-type]
        except Exception:
            extras_obj = extras_dict  # type: ignore[assignment]

        bundle = ForecastBundle(
            point=point,
            quantiles=quantiles_out,
            params=params_stacked,
            extras=extras_obj
        )

        if return_bundle:
            return bundle

        if isinstance(quantiles_out, torch.Tensor):
            return quantiles_out
        return point

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ):
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
