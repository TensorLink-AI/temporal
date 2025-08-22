import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MultiStepMixin:
    """
    Single-pass multi-step forecasting with chunked roll-forward.

    Behavior:
      - If prediction_length <= chunk_length: one decoder call over Tpred (true single pass).
      - If prediction_length >  chunk_length: run ceil(Tpred/chunk_length) passes; each pass
        predicts a full chunk_length in one shot and *rolls* those predictions as the next pass's inputs.

    Assumes the host model defines:
      - self.config with .feature_size (and optionally .prediction_length, .num_attention_heads)
      - self.preprocessor with .process(...) and optional .denormalize(...)
      - self.decoder (and optional self.encoder)
      - self.output_heads (nn.Module or nn.ModuleList)
      - optional self.head_aggregator (for multi-head aggregation)
    """

    # --------------------------- core helpers ---------------------------------

    def _get_head_output(self, hidden: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads, required for generation.")
        if isinstance(self.output_heads, nn.ModuleList):
            return [head(hidden) for head in self.output_heads]
        return self.output_heads(hidden)

    def _get_primary_head(self) -> nn.Module:
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    def _normalize_levels(self, quantile_levels):
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    @staticmethod
    def _stack_distpred_chunks(chunks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Concatenate a list of full-horizon DistPred dicts along time.
        Each item must have:
          - 'paths': [B, Tp, F, K]
          - optional 'path_logits': [B, Tp, K]
        Returns:
          {'paths': [B, sum(Tp), F, K], 'path_logits': [B, sum(Tp), K]?}
        """
        if not chunks:
            raise ValueError("Empty chunks list for DistPred stacking.")
        paths = torch.cat([c["paths"] for c in chunks], dim=1)
        out = {"paths": paths}
        if "path_logits" in chunks[0] and chunks[0]["path_logits"] is not None:
            logits = torch.cat([c["path_logits"] for c in chunks], dim=1)
            out["path_logits"] = logits
        return out

    def _compute_prediction_to_store(
        self,
        raw_head_output: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
        prediction_strategy: Optional[Union[str, float, int]],
        quantile_levels: Optional[List[float]],
        output_head: nn.Module,
    ) -> Union[torch.Tensor, Dict[str, Any], List[torch.Tensor]]:
        """
        Vectorized over time; heads are expected to handle [B,T,...] inputs.
        Priority: predict(method) > sample_quantiles(levels) > raw.
        Supports DistPred dicts by unwrapping for quantiles.
        """
        levels = self._normalize_levels(quantile_levels)

        # Multi-head
        if isinstance(raw_head_output, list):
            assert isinstance(self.output_heads, nn.ModuleList)
            per_head = []
            for h, y in zip(self.output_heads, raw_head_output):
                if prediction_strategy is not None and hasattr(h, "predict") and callable(getattr(h, "predict")):
                    per_head.append(h.predict(y, method=prediction_strategy))
                elif levels is not None and hasattr(h, "sample_quantiles") and callable(getattr(h, "sample_quantiles")):
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

        # Single-head
        y = raw_head_output
        if prediction_strategy is not None and hasattr(output_head, "predict") and callable(getattr(output_head, "predict")):
            return output_head.predict(y, method=prediction_strategy)
        if levels is not None and hasattr(output_head, "sample_quantiles") and callable(getattr(output_head, "sample_quantiles")):
            y_in = y["paths"] if isinstance(y, dict) and "paths" in y else y
            return output_head.sample_quantiles(y_in, quantile_levels=levels)
        return y

    def _build_future_inputs(
        self,
        *,
        batch_size: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
        feature_size: int,
        fill_future: str,
        encoder_inputs: Optional[torch.Tensor],
        previous_chunk_values: Optional[torch.Tensor],  # [B, steps_prev, F] in *model input domain*
    ) -> torch.Tensor:
        """
        Build decoder_inputs for a single-pass chunk.
        fill_future:
          - 'zeros': zeros
          - 'repeat_last': repeat encoder last value
          - 'prev_pred': use the previous chunk's predictions (if provided), else fallback to 'repeat_last'
        """
        if fill_future == "prev_pred" and previous_chunk_values is not None:
            # Use previous predictions as direct inputs for the next pass
            if previous_chunk_values.shape[1] != steps:
                # If steps differ (e.g., remainder), pad or crop
                if previous_chunk_values.shape[1] > steps:
                    return previous_chunk_values[:, :steps]
                else:
                    pad = steps - previous_chunk_values.shape[1]
                    pad_tail = torch.zeros(previous_chunk_values.shape[0], pad, previous_chunk_values.shape[2],
                                           device=previous_chunk_values.device, dtype=previous_chunk_values.dtype)
                    return torch.cat([previous_chunk_values, pad_tail], dim=1)

            return previous_chunk_values

        if fill_future == "repeat_last":
            if encoder_inputs is None:
                raise ValueError("fill_future='repeat_last' requires encoder_inputs.")
            last = encoder_inputs[:, -1:, :]  # [B,1,F]
            return last.expand(batch_size, steps, feature_size).contiguous()

        # zeros
        return torch.zeros(batch_size, steps, feature_size, device=device, dtype=dtype)

    # --------------------------- main API ------------------------------------

    @torch.no_grad()
    def forecast_single_pass_chunked(
        self,
        inputs: torch.Tensor,                  # encoder inputs if encoder-decoder; decoder prompt otherwise
        prediction_length: int,
        *,
        quantiles: Optional[List[float]] = None,
        chunk_length: Optional[int] = None,    # default: config.prediction_length or 256
        fill_future: str = "repeat_last",      # {'zeros','repeat_last','prev_pred'}
        collect: str = "auto",                 # {'auto','none','distpred','per_chunk'}
        prediction_strategy: Optional[Union[str, float, int]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        return_raw: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """
        Single-pass multi-step with chunked roll-forward.

        Returns:
          - collect='none'     -> aggregated tensor when possible (predict/quantiles/raw)
          - collect='distpred' -> {'paths':[B,T,F,K], 'path_logits': [B,T,K]?}
          - collect='per_chunk'-> list of per-chunk items (tensor or dict), already time-major
          - collect='auto'     -> if DistPred dicts, returns stacked dict; else tensor
        """
        self.eval()

        # Resolve chunk length
        if chunk_length is None:
            chunk_length = getattr(self.config, "prediction_length", None) or 256
        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {chunk_length}")

        B, device, dtype = inputs.shape[0], inputs.device, inputs.dtype
        F = getattr(self.config, "feature_size", None)
        if F is None:
            raise ValueError("config.feature_size is required.")

        # Optional encoder pass
        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder is not None:
            enc = self.preprocessor.process(
                input_values=inputs,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            enc_out = self.encoder(
                hidden_states=enc["hidden_states"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state  # [B, S_enc, D]

        primary_head = self._get_primary_head()

        total_T = prediction_length
        n_full = total_T // chunk_length
        rem = total_T % chunk_length

        chunk_sizes: List[int] = ([chunk_length] * n_full) + ([rem] if rem > 0 else [])
        outputs_per_chunk: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []

        previous_chunk_values: Optional[torch.Tensor] = None  # in *model input domain* (not denormed)

        for j, steps in enumerate(chunk_sizes):
            # Build decoder inputs for this chunk
            dec_in = self._build_future_inputs(
                batch_size=B,
                steps=steps,
                device=device,
                dtype=dtype,
                feature_size=F,
                fill_future=("prev_pred" if j > 0 else fill_future),
                encoder_inputs=inputs,
                previous_chunk_values=previous_chunk_values,
            )  # [B, steps, F]

            # Mask for this chunk
            dec_mask = decoder_attention_mask
            if dec_mask is None:
                dec_mask = torch.ones(B, steps, device=device, dtype=torch.float32)

            # One-pass processing for the chunk
            dec_proc = self.preprocessor.process(
                input_values=dec_in,
                past_key_values_length=0,
                attention_mask=dec_mask,
                is_causal=True,  # causal across the steps of THIS chunk
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            dec_out = self.decoder(
                hidden_states=dec_proc["hidden_states"],   # [B, steps, D]
                attention_mask=dec_proc["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                return_dict=True,
            )
            hidden_all = dec_out.last_hidden_state  # [B, steps, D]

            # Head over the entire chunk
            head_out = self._get_head_output(hidden_all)  # tensor | dict | list

            # Package per-chunk output
            store_item = self._compute_prediction_to_store(
                head_out, prediction_strategy, quantiles, primary_head
            )

            # Keep per-chunk (for later stacking/concat)
            outputs_per_chunk.append(store_item)

            # Prepare next chunk's decoder inputs (roll forward) in *model input domain*:
            # - If store_item is a tensor of [B, steps, F] or [B, steps, F, Q], we prefer the median track for roll.
            # - If it's a DistPred dict, roll the median (or mean) across K.
            if j < len(chunk_sizes) - 1:  # not last chunk
                if isinstance(head_out, dict) and "paths" in head_out:
                    # roll the median path over K (no denorm here)
                    paths = head_out["paths"]        # [B, steps, F, K]
                    med = torch.quantile(paths, q=0.5, dim=-1)  # [B, steps, F]
                    previous_chunk_values = med
                elif torch.is_tensor(store_item):
                    # If it's [B, steps, F, Q], roll the median quantile; else assume [B, steps, F]
                    if store_item.ndim == 4:
                        qdim = store_item.shape[-1]
                        if qdim == 1:
                            previous_chunk_values = store_item.squeeze(-1)
                        else:
                            # median over Q
                            previous_chunk_values = torch.median(store_item, dim=-1).values
                    else:
                        previous_chunk_values = store_item
                else:
                    # Multi-head list or unsupported type — fallback: use primary_head.predict(median) if possible
                    if hasattr(primary_head, "predict") and callable(getattr(primary_head, "predict")):
                        mid = primary_head.predict(head_out, method="median")
                        previous_chunk_values = mid
                    else:
                        raise TypeError(
                            "Cannot roll forward: head output type unsupported for building next chunk inputs."
                        )

        # ---- stack chunks along time ----
        if collect == "per_chunk":
            final = outputs_per_chunk
        else:
            first = outputs_per_chunk[0]

            # DistPred dict path
            if (collect in ("auto", "distpred")) and isinstance(first, dict):
                # If the item is an aggregated dict (e.g., predict() returned a dict), prefer raw head dicts
                # If your heads return proper DistPred dicts on forward for full horizon, you might want to
                # re-run stacking on raw head dicts instead. Here we assume store_item preserved DistPred dict.
                dist_chunks = []
                for item in outputs_per_chunk:
                    if not (isinstance(item, dict) and "paths" in item):
                        raise TypeError("collect='distpred' requested but a chunk is not a DistPred dict.")
                    dist_chunks.append(item)
                final = self._stack_distpred_chunks(dist_chunks)

            # Tensor path
            elif torch.is_tensor(first):
                try:
                    final = torch.cat(outputs_per_chunk, dim=1)  # [B, T, ...]
                except Exception as e:
                    # Some heads may yield shape changes — fall back to list
                    logger.warning(f"Concat failed for tensor outputs; returning per-chunk list. Error: {e}")
                    final = outputs_per_chunk
            else:
                # Multi-head or mixed objects: return the list
                final = outputs_per_chunk

        # ---- optional denormalize for tensors with last-dim == feature_size ----
        if isinstance(final, torch.Tensor):
            last_dim = final.size(-1) if final.dim() >= 2 else None
            can_denorm = (last_dim == getattr(self.config, "feature_size", last_dim))
            if hasattr(self.preprocessor, 'denormalize') and can_denorm and not return_raw:
                logger.info("Denormalizing single-pass chunked predictions.")
                final = self.preprocessor.denormalize(final)

        return final
