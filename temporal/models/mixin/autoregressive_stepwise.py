import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Dict, List, Tuple

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)

class AutoregressiveStepwiseMixin:
    """
    Autoregressive generation with a consistent ForecastBundle return option.
    Assumes:
      - self.config.{feature_size, hidden_size[, prediction_length, num_attention_heads]}
      - self.preprocessor.process(...), optional self.preprocessor.denormalize(...)
      - self.decoder (and optional self.encoder)
      - self.output_heads (nn.Module or nn.ModuleList)
      - optional self.head_aggregator for multi-head aggregation
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
            raise TypeError(f"Could not convert '{name}'={value} to float. Error: {e}")

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
        raise ValueError(f"Expected feedback tensor with 1–4 dims, got {x.shape}")

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

    # --------------------------- store vs feedback ----------------------------

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
            assert isinstance(self.output_heads, nn.ModuleList)
            per_head = []
            for h, y in zip(self.output_heads, raw_head_output):
                if prediction_strategy is not None and hasattr(h, "predict"):
                    per_head.append(h.predict(y, method=prediction_strategy))
                elif (levels is not None) and (not defer_quantiles) and hasattr(h, "sample_quantiles"):
                    y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
                    per_head.append(h.sample_quantiles(y_in, quantile_levels=levels))
                else:
                    per_head.append(y)
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                try:
                    return self.head_aggregator(per_head)
                except Exception as e:
                    logger.warning(f"head_aggregator failed; returning per-head list. Error: {e}")
                    return per_head
            return per_head

        y = raw_head_output
        if prediction_strategy is not None and hasattr(output_head, "predict"):
            return output_head.predict(y, method=prediction_strategy)
        if (levels is not None) and (not defer_quantiles) and hasattr(output_head, "sample_quantiles"):
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
                if use_sampling and hasattr(output_head, "sample"):
                    logger.warning("Aggregator used; sampling from primary head only.")
            else:
                logger.warning("Multiple output heads without aggregator; using head[0] for AR feedback.")
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
            out = output_head.sample_quantiles(feedback_source, quantile_levels=[q])
            if torch.is_tensor(out) and out.ndim == 4:
                out = out.squeeze(-1)
            return self._ensure_b1f(out, self.config.feature_size)

        if isinstance(feedback_source, torch.Tensor):
            if feedback_source.ndim == 4:
                feedback_source = feedback_source.mean(dim=-1)
            return self._ensure_b1f(feedback_source, self.config.feature_size)

        raise TypeError("Output head returning a dict must implement .sample() or .predict() for AR feedback.")

    # --------------------------- post-quantiles & params stack ----------------

    def _post_quantiles_any(
        self,
        params_or_preds: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]],
        head: nn.Module,
        quantile_levels: List[float],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]]:
        """Compute quantiles once after loop for tensor OR dict OR list-of-heads."""
        if hasattr(head, "sample_quantiles"):
            try:
                return head.sample_quantiles(params_or_preds, quantile_levels)
            except Exception as e:
                logger.warning(f"post-quantiles failed on primary head: {e}")
        # Multi-head: per-head quantiles when possible
        if isinstance(params_or_preds, list) and isinstance(self.output_heads, nn.ModuleList):
            out_list = []
            for h, y in zip(self.output_heads, params_or_preds):
                if hasattr(h, "sample_quantiles"):
                    try:
                        out_list.append(h.sample_quantiles(y, quantile_levels))
                    except Exception as e:
                        logger.warning(f"post-quantiles failed on subhead: {e}")
                        out_list.append(y)
                else:
                    out_list.append(y)
            return out_list
        return params_or_preds

    def _accum_params_dict_step(self, acc: Optional[Dict[str, List[torch.Tensor]]], step_dict: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        if acc is None:
            acc = {k: [] for k in step_dict.keys() if torch.is_tensor(step_dict[k])}
        for k, v in step_dict.items():
            if not torch.is_tensor(v):
                continue
            # ensure T==1 tensors end up as [B,1,...]
            if v.ndim >= 2 and v.shape[1] != 1:
                v = v[:, -1:, ...]  # take last step if head returned multiple
            acc.setdefault(k, []).append(v)
        return acc

    def _stack_params_dict(self, acc: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.cat(vs, dim=1) for k, vs in acc.items()}  # [B,T,...] for each key

    # --------------------------- bundle assembly helpers ----------------------

    def _compute_point_from_params(
        self,
        source: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]],
        head: nn.Module,
        *,
        quantiles_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Derive point forecast [B,T,F] from:
          - quantiles (median),
          - or head.predict('median'|'mean') across time,
          - or raise with a clear error.
        """
        F = getattr(self.config, "feature_size", None)

        if isinstance(quantiles_tensor, torch.Tensor):
            # [B,T,F,Q] -> median
            qdim = quantiles_tensor.shape[-1]
            mid = qdim // 2
            point = quantiles_tensor[..., mid]
            if point.ndim == 2:  # [B,T] -> [B,T,1]
                point = point.unsqueeze(-1)
            return point

        if hasattr(head, "predict"):
            # We need [B,T,*] params; if dict, pass dict; if tensor, pass tensor
            try:
                pv = head.predict(source, method="median")
            except Exception:
                pv = head.predict(source, method="mean")
            if pv.ndim == 2:
                pv = pv.unsqueeze(-1)
            return pv

        raise TypeError("Cannot compute point forecast from given params; add 'predict()' to this head.")

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
        return_bundle: bool = False,
        **kwargs,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
        ForecastBundle,
        List[Dict[str, Union[torch.Tensor, List[str]]]],
    ]:
        """
        If return_bundle=False (default): returns tensor (point or quantiles).
        If return_bundle=True: returns ForecastBundle(point, quantiles, params, extras).
        """
        self.eval()
        if enable_mc_dropout:
            self.enable_dropout()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("Provide 'encoder_inputs' or 'decoder_inputs'.")

        if prediction_length is None:
            prediction_length = getattr(self.config, 'prediction_length', 0)

        ref_tensor = decoder_inputs if encoder_inputs is None else encoder_inputs
        batch_size, device, dtype = ref_tensor.shape[0], ref_tensor.device, ref_tensor.dtype

        if prediction_length == 0:
            empty = torch.empty((batch_size, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        # ----- encoder pass (optional)
        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder is not None and encoder_inputs is not None:
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

        # primary head
        primary_head = self._get_primary_head()

        # seed decoder inputs
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                raise ValueError("Decoder-only generation needs a real prompt in decoder_inputs.")

        # warm step to infer shapes
        warm_mask = decoder_attention_mask if decoder_attention_mask is not None \
            else torch.ones(decoder_inputs.size(0), decoder_inputs.size(1), device=decoder_inputs.device, dtype=torch.float32)

        warm_proc = self.preprocessor.process(
            input_values=decoder_inputs,
            past_key_values_length=0,
            attention_mask=warm_mask,
            is_causal=True,
            validate_shapes=validate_shapes,
            verbose=verbose,
        )
        warm_hidden = warm_proc["hidden_states"]
        if hasattr(self, "_values_to_hidden"):
            warm_hidden = self._values_to_hidden(warm_hidden)

        warm_out = self.decoder(
            hidden_states=warm_hidden,
            attention_mask=warm_proc["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        warm_last = warm_out.last_hidden_state[:, -1:, :]
        warm_head_out = self._get_head_output(warm_last)

        # what we store per step (defer quantiles if we will do post)
        defer_q = (post_quantiles and quantile_levels is not None)
        warm_store = self._compute_prediction_to_store(
            warm_head_out, prediction_strategy, quantile_levels, primary_head, defer_quantiles=defer_q
        )
        use_prealloc_tensor = torch.is_tensor(warm_store)

        # accumulators
        sampled_steps: List[torch.Tensor] = []     # [B,1,F] per step if store_sampled
        params_acc_tensor = None                   # [B,T,...] if tensor params
        params_acc_dict: Optional[Dict[str, List[torch.Tensor]]] = None  # dict-of-lists for dict heads
        store_acc: Optional[torch.Tensor] = None   # legacy output for non-bundle returns

        if use_prealloc_tensor:
            step_shape = warm_store.shape[2:]
            store_acc = torch.zeros((batch_size, prediction_length, *step_shape), device=warm_store.device, dtype=warm_store.dtype)
        else:
            store_list: List[Any] = []

        # include warm step in loop via same path
        past_key_values = None
        eos_value = self._get_scalar_value(eos_token_id, "eos_token_id")
        next_input = decoder_inputs

        for t in range(prediction_length):
            step_in = next_input
            past_len = self._get_cache_length(past_key_values)

            if decoder_attention_mask is not None:
                if decoder_attention_mask.dim() != 2 or decoder_attention_mask.size(0) != step_in.size(0):
                    raise ValueError("decoder_attention_mask must be [B, T_step].")
                dec2d = decoder_attention_mask
            else:
                dec2d = torch.ones(step_in.size(0), step_in.size(1), device=step_in.device, dtype=torch.float32)

            dec_proc = self.preprocessor.process(
                input_values=step_in,
                past_key_values_length=past_len,
                attention_mask=dec2d,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            dec_hidden = dec_proc["hidden_states"]
            if hasattr(self, "_values_to_hidden"):
                dec_hidden = self._values_to_hidden(dec_hidden)

            dec_out = self.decoder(
                hidden_states=dec_hidden,
                attention_mask=dec_proc["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            last_h = dec_out.last_hidden_state[:, -1:, :]

            head_out = self._get_head_output(last_h)

            # accumulate raw params for bundle
            if torch.is_tensor(head_out):
                # ensure [B,1,*]
                v = head_out if head_out.ndim >= 3 else head_out.unsqueeze(1)
                params_acc_tensor = v if params_acc_tensor is None else torch.cat([params_acc_tensor, v], dim=1)
            elif isinstance(head_out, dict):
                params_acc_dict = self._accum_params_dict_step(params_acc_dict, head_out)

            # store (defer per-step quantiles if post mode)
            to_store = self._compute_prediction_to_store(
                head_out, prediction_strategy, quantile_levels, primary_head, defer_quantiles=defer_q
            )
            if use_prealloc_tensor and torch.is_tensor(to_store):
                store_acc[:, t] = to_store.squeeze(1)
            else:
                if use_prealloc_tensor and not torch.is_tensor(to_store):
                    # switch to list mode
                    use_prealloc_tensor = False
                    store_list = [store_acc[:, 0:1]]
                    store_acc = None
                if not use_prealloc_tensor:
                    store_list.append(to_store)

            # feedback
            next_value = self._compute_next_decoder_input_value(
                head_out, prediction_strategy, primary_head, use_sampling=sampling, sampling_kwargs=sampling_kwargs
            )
            if store_sampled:
                sampled_steps.append(next_value)  # [B,1,F]
            next_input = next_value

            if use_cache:
                past_key_values = dec_out.past_key_values

            if early_stopping and eos_value is not None:
                target = torch.full_like(next_value, eos_value)
                if torch.allclose(next_value, target, rtol=0.0, atol=1e-6):
                    if use_prealloc_tensor and store_acc is not None:
                        store_acc = store_acc[:, :t+1]
                    break

        # assemble stored (legacy output path)
        stored = None
        if use_prealloc_tensor and store_acc is not None:
            stored = store_acc
        else:
            if 'store_list' in locals() and store_list:
                if isinstance(store_list[0], dict):
                    stored = store_list  # list of dicts (rarely desired for direct return)
                else:
                    stored = torch.cat(store_list, dim=1)

        # stack params dict if needed
        params_stacked: Union[torch.Tensor, Dict[str, torch.Tensor], None] = None
        if params_acc_tensor is not None:
            params_stacked = params_acc_tensor  # [B,T,*]
        elif params_acc_dict is not None:
            params_stacked = self._stack_params_dict(params_acc_dict)  # {k: [B,T,...]}

        # post-quantiles (single shot)
        levels = self._normalize_levels(quantile_levels)
        quantiles_out = None
        if levels is not None and params_stacked is not None:
            quantiles_out = self._post_quantiles_any(params_stacked, primary_head, levels)
            # quantiles_out usually [B,T,F,Q] (tensor). For dict/list heads we keep structure.

        # compute point forecast
        try:
            if store_sampled and sampled_steps:
                point = torch.cat(sampled_steps, dim=1)  # [B,T,F]
            else:
                source_for_point = quantiles_out if isinstance(quantiles_out, torch.Tensor) else params_stacked
                point = self._compute_point_from_params(source_for_point, primary_head,
                                                        quantiles_tensor=quantiles_out if isinstance(quantiles_out, torch.Tensor) else None)
        except Exception:
            # final fallback: use 'stored' if it's a tensor with last dim == F
            if torch.is_tensor(stored) and stored.size(-1) == getattr(self.config, "feature_size", stored.size(-1)):
                point = stored
            else:
                raise

        # optional denorm for point only (never denorm quantiles)
        if hasattr(self.preprocessor, 'denormalize') and not return_raw:
            if isinstance(point, torch.Tensor) and point.size(-1) == getattr(self.config, "feature_size", point.size(-1)):
                point = self.preprocessor.denormalize(point)

        # assemble bundle
        extras_dict: Dict[str, Any] = {}
        # you can stash per-head extras here if you produce them (e.g., path logits summaries)
        bundle = ForecastBundle(
            point=point,                              # [B,T,F]
            quantiles=quantiles_out,                  # [B,T,F,Q] or structure
            params=params_stacked,                    # [B,T,*] tensor or dict-of-[B,T,...]
            extras=extras_dict if hasattr(HeadExtras, "__annotations__") else extras_dict
        )

        if return_bundle:
            return bundle

        # legacy: if user asked for quantiles, return quantile tensor (if available), else point
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
