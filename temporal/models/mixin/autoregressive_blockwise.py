import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveBlockwiseMixin:
    """
    Blockwise autoregressive generation that mimics multi-horizon supervision.

    Unlike AutoregressiveStepwiseMixin (which rolls out 1 step at a time and
    threads a KV cache), this mixin rolls out *blocks* of future steps at once.

    Loop structure:
      1. Take the current known/predicted sequence as decoder context.
      2. Run a single forward-style decode that produces a block of length `block_len`
         (like we do in training when we predict a full future horizon in parallel).
      3. Append that entire predicted block to the running sequence ("feedback").
      4. Keep only as many steps from that block as we still need to reach the
         requested `prediction_length`.
      5. Repeat until `prediction_length` is satisfied.

    Key differences from AutoregressiveStepwiseMixin:
      • Much faster rollout for long horizons (fewer decoder calls).
      • No per-step KV caching.
      • No early stopping on EOS.
      • We treat an entire predicted block like one 'token' of rollout.
      • We still optionally produce:
          - point forecasts
          - quantiles
          - params
          - ForecastBundle
        so downstream evaluation code can look consistent.

    Assumptions about the host model:
      - self.config:
          • feature_size: int           # number of target channels
          • prediction_length: Optional[int]
      - self.preprocessor:
          • process(input_values, attention_mask, past_key_values_length, is_causal, ...)
          • denormalize(tensor)  # optional; applied to point at the end if requested
      - self.encoder and self.decoder:
          • Similar to HuggingFace encoder-decoder or decoder-only.
          • self.decoder returns an object with .last_hidden_state (B, T, D)
      - self.output_heads:
          • nn.Module or nn.ModuleList that maps hidden states -> forecast params
          • Each head supports:
                - predict(params, method="median"/"mean"/quantile/index)
                - sample_quantiles(params, quantile_levels)  (optional)
                - sample(params, **kwargs)                   (optional, stochastic)
      - self._primary_head() or self._get_primary_head():
          • returns the main head (for point decoding / sampling)
      - self.forward(...):
          • can accept (encoder_inputs=..., decoder_inputs=..., targets=fake_targets, ...)
          • returns an object with `.logits` shaped [B, block_len, F*] or compatible,
            and `.loss` if targets were real.
          • IMPORTANT: this forward path *must* reconstruct patches etc.
            and slice the hidden states so that `.logits` is aligned to an
            output horizon of length = targets.size(1).

    Returned values:
      * If return_bundle=True:
          ForecastBundle(point[B,T,F], quantiles[B,T,F,Q] | None, params, extras)
      * else:
          If quantile_levels requested and they resolve to a quantile tensor,
          we return that tensor [B,T,F,Q]; otherwise we return point [B,T,F].

    Shape conventions:
      - decoder_inputs: [B, T_hist, F]
      - encoder_inputs: [B, T_enc, F_enc] (optional if model is encoder-decoder)
      - prediction_length: scalar int, total T_future to generate
      - block_len: scalar int, how many steps each block predicts
      - output point: [B, prediction_length, F]
      - output quantiles: [B, prediction_length, F, Q]

    Notes:
      - We always feed back the *entire* block prediction to the running sequence,
        even if we only "needed" part of it to reach prediction_length. This matches
        the "append full block" behavior we discussed.
    """

    # ---- helpers reused from AutoregressiveStepwiseMixin ----
    def _normalize_levels(self, quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _get_primary_head(self) -> nn.Module:
        """Return the first (primary) head."""
        # mirror AutoregressiveStepwiseMixin._get_primary_head
        if not hasattr(self, "output_heads"):
            raise AttributeError("Model is missing output_heads.")
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    def _ensure_components_present(self, params: Dict[str, Any], head: nn.Module) -> Dict[str, Any]:
        if "components" not in params and hasattr(head, "components"):
            try:
                params = dict(params)
                params["components"] = list(getattr(head, "components"))
            except Exception:
                pass
        return params

    def _normalize_quantile_shape(
        self,
        q: torch.Tensor,
        *,
        feature_size: int,
        Q: int,
    ) -> torch.Tensor:
        """
        Normalize shapes to [B, T, F, Q].

        Accepts:
          - [B,T,Q]      -> [B,T,1,Q]
          - [B,T,F,Q]    -> as-is
          - [B,T,Q,F]    -> [B,T,F,Q]
          - [B,Q]        -> [B,1,1,Q]
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

    def _head_predict_point(
        self,
        head: nn.Module,
        params_block: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        method_priority: Tuple[str, ...] = ("median", "mean"),
    ) -> torch.Tensor:
        """
        Turn head params -> point forecast.

        Tries head.predict(..., method='median'), falls back to 'mean'.
        Ensures output shape [B,T,F].
        """
        src = params_block
        if isinstance(src, dict):
            src = self._ensure_components_present(src, head)

        if hasattr(head, "predict"):
            last_err = None
            for m in method_priority:
                try:
                    out = head.predict(src, method=m)
                    break
                except Exception as e:
                    last_err = e
            else:
                raise RuntimeError(
                    f"head.predict() failed for all methods {method_priority}: {last_err}"
                )
        else:
            raise TypeError("Output head must implement .predict(...) to get a point forecast.")

        # normalize [B,T] -> [B,T,1]
        if out.ndim == 2:
            out = out.unsqueeze(-1)
        return out

    def _head_quantiles(
        self,
        head: nn.Module,
        params_block: Union[torch.Tensor, Dict[str, torch.Tensor]],
        quantile_levels: List[float],
        feature_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Try to compute quantiles [B,T,F,Q] from the head's params for 1 block.
        If head doesn't support quantiles directly, but the params already
        LOOK like quantiles (Quantile Regression), we'll reshape those.
        Otherwise returns None.
        """
        if not quantile_levels:
            return None

        # First try if head has sample_quantiles
        if hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
            src = params_block
            if isinstance(src, dict):
                src = self._ensure_components_present(src, head)
            q_raw = head.sample_quantiles(src, quantile_levels)
            if not torch.is_tensor(q_raw):
                # If it's not a tensor, we can't cleanly stack across blocks later.
                return None
            return self._normalize_quantile_shape(
                q_raw,
                feature_size=feature_size,
                Q=len(quantile_levels),
            )

        # Otherwise, check for direct QR-style params
        if torch.is_tensor(params_block):
            if params_block.shape[-1] == len(quantile_levels):
                # interpret last dim as Q
                return self._normalize_quantile_shape(
                    params_block,
                    feature_size=feature_size,
                    Q=len(quantile_levels),
                )

        return None

    # ---------------------------------------------------------------------
    # Blockwise Generation API
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        *,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        block_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = False,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
    ) -> Union[
        ForecastBundle,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Blockwise autoregressive forecast.

        Each loop:
          1. Run the model once to produce a parallel block forecast of length `block_len`.
          2. Append the *entire* block forecast to the running decoder sequence.
          3. Take just the #steps we still need for the requested horizon.
          4. Repeat until we hit `prediction_length`.

        Parameters
        ----------
        encoder_inputs : Tensor | None
            [B, L_enc, F_enc] history for encoder (if encoder-decoder).
        decoder_inputs : Tensor | None
            [B, L_hist, F] initial known context for decoder.
            If None and encoder_inputs is provided, we seed decoder with the last
            encoder timestep. If both are None, we raise.
        prediction_length : int | None
            Total forecast horizon we want, T_future.
            Falls back to self.config.prediction_length if not provided.
        block_len : int | None
            How many steps to forecast per block call.
            Typically this matches your training horizon length. If None,
            falls back to self.config.prediction_length.
        attention_mask : Tensor | None
            Optional padding mask for encoder_inputs [B, L_enc].
        decoder_attention_mask : Tensor | None
            Optional padding mask for decoder_inputs [B, L_hist].
            (Not strongly used here beyond shape checks.)
        quantile_levels : List[float] | None
            Quantiles (e.g. [0.1,0.5,0.9]) to compute per block and stitch.
            If None, we only compute point.
        validate_shapes : bool
            Passed to preprocessor for debugging shape assertions.
        verbose : bool
            Passed to preprocessor for verbose shape printouts.
        denormalize : bool
            If True and self.preprocessor has .denormalize(), apply to final
            point forecast (but not to quantiles/params).
        return_bundle : bool
            If True, return a ForecastBundle(point, quantiles, params, extras).
        return_raw : bool
            If False and denormalize=True, we'll denormalize the point forecast.

        Returns
        -------
        If return_bundle=True:
            ForecastBundle(
                point      [B, prediction_length, F],
                quantiles  [B, prediction_length, F, Q] or None,
                params     Tensor or Dict[str,Tensor] stacked across all blocks,
                extras     HeadExtras | dict
            )
        else:
            If quantiles are available -> quantiles tensor [B, T, F, Q]
            else -> point tensor [B, T, F]
        """
        self.eval()

        # ---- resolve horizon / block_len ----
        if prediction_length is None:
            prediction_length = getattr(self.config, "prediction_length", None)
            if prediction_length is None:
                raise ValueError("prediction_length must be provided or set in config.prediction_length")

        if block_len is None:
            block_len = getattr(self.config, "prediction_length", None)
            if block_len is None:
                raise ValueError("block_len must be provided or set in config.prediction_length")

        # ---- choose reference for device/dtype ----
        ref = decoder_inputs if decoder_inputs is not None else encoder_inputs
        if ref is None:
            raise ValueError("You must provide either decoder_inputs or encoder_inputs.")
        B, device, dtype = ref.shape[0], ref.device, ref.dtype

        # ---- encoder forward (if present) ----
        encoder_hidden_states = None
        if hasattr(self, "encoder") and self.encoder is not None and encoder_inputs is not None:
            enc_proc = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                is_causal=False,  # encoders aren't causal
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            enc_out = self.encoder(
                hidden_states=enc_proc["hidden_states"],
                attention_mask=enc_proc["attention_mask"],
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state  # [B,L_enc,D_enc]

        # ---- seed decoder_inputs if missing ----
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()  # [B,1,F]
            else:
                raise ValueError("Decoder-only blockwise generation requires decoder_inputs.")

        # ---- prep rollout containers ----
        remaining = prediction_length
        collected_point_blocks: List[torch.Tensor] = []      # list of [B, k, F]
        collected_quant_blocks: List[torch.Tensor] = []       # list of [B, k, F, Q]
        quant_list_enabled = quantile_levels is not None
        quantile_levels_norm = self._normalize_levels(quantile_levels)

        # We'll also accumulate params across blocks (for bundle.params)
        params_tensor_acc: Optional[torch.Tensor] = None
        params_dict_acc_tensors: Optional[Dict[str, List[torch.Tensor]]] = None
        params_dict_acc_meta: Optional[Dict[str, Any]] = None

        # helper to accumulate dict params
        def _accum_dict_step(
            acc_tensors: Optional[Dict[str, List[torch.Tensor]]],
            acc_meta: Optional[Dict[str, Any]],
            block_dict: Dict[str, Any],
        ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]:
            if acc_tensors is None:
                acc_tensors = {}
            if acc_meta is None:
                acc_meta = {}
            for k, v in block_dict.items():
                if torch.is_tensor(v):
                    # ensure [B, T, ...]
                    vv = v
                    # if it's missing time dim, unsqueeze
                    if vv.ndim == 2:
                        vv = vv.unsqueeze(1)  # [B,1,...]
                    acc_tensors.setdefault(k, []).append(vv)
                else:
                    acc_meta.setdefault(k, v)
            return acc_tensors, acc_meta

        # ---- rollout loop ----
        running_seq = decoder_inputs  # grows each block: [B, T_running, F]

        while remaining > 0:
            # fake targets: only length matters
            fake_targets = torch.zeros(
                (running_seq.size(0), block_len, getattr(self.config, "feature_size", running_seq.size(-1))),
                device=device,
                dtype=dtype,
            )

            # run the model forward ONCE to get a block_len forecast
            out = self.forward(
                encoder_inputs=encoder_inputs,
                decoder_inputs=running_seq,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                targets=fake_targets,          # <-- tells forward how many future steps to project
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            # out.logits: [B, block_len, F?] (point-style or param-style)

            block_params = out.logits        # may be Tensor OR dict-like head output per step
            k = min(remaining, block_len)    # how many steps from this block we actually still need

            primary_head = self._get_primary_head()
            Fsize = getattr(self.config, "feature_size", block_params.shape[-1] if torch.is_tensor(block_params) else 1)

            # Turn block params into point forecast for this block
            if isinstance(block_params, dict):
                block_params = self._ensure_components_present(block_params, primary_head)
                # compute point and optional quantiles from dict-like params
                point_block_full = self._head_predict_point(primary_head, block_params)           # [B, block_len, F]
                quant_block_full = None
                if quant_list_enabled and quantile_levels_norm:
                    quant_block_full = self._head_quantiles(
                        primary_head,
                        block_params,
                        quantile_levels_norm,
                        feature_size=Fsize,
                    )  # [B, block_len, F, Q] or None

                # accumulate params dict for bundle.params
                params_dict_acc_tensors, params_dict_acc_meta = _accum_dict_step(
                    params_dict_acc_tensors,
                    params_dict_acc_meta,
                    block_params,
                )

            elif isinstance(block_params, torch.Tensor):
                # block_params is typically [B, block_len, something]
                point_block_full = self._head_predict_point(primary_head, block_params)           # [B, block_len, F]
                quant_block_full = None
                if quant_list_enabled and quantile_levels_norm:
                    quant_block_full = self._head_quantiles(
                        primary_head,
                        block_params,
                        quantile_levels_norm,
                        feature_size=Fsize,
                    )

                # accumulate tensor params across blocks for bundle.params
                # ensure [B, T_block, ...]
                bp_for_stack = block_params
                if bp_for_stack.ndim == 2:
                    bp_for_stack = bp_for_stack.unsqueeze(1)  # [B,1,...]
                if params_tensor_acc is None:
                    params_tensor_acc = bp_for_stack
                else:
                    params_tensor_acc = torch.cat([params_tensor_acc, bp_for_stack], dim=1)

            else:
                raise TypeError(
                    "Unsupported block_params type from model.forward(); "
                    "expected Tensor or Dict[str,Tensor]."
                )

            # slice k steps we actually need now
            point_block_slice = point_block_full[:, :k, :]  # [B,k,F]
            collected_point_blocks.append(point_block_slice)

            if quant_list_enabled and quantile_levels_norm and quant_block_full is not None:
                quant_block_slice = quant_block_full[:, :k, ...]  # [B,k,F,Q]
                collected_quant_blocks.append(quant_block_slice)

            # feedback ENTIRE predicted block to running_seq, not just k
            running_seq = torch.cat([running_seq, point_block_full], dim=1)  # [B, T_running+block_len, F]

            remaining -= k

        # ---- stitch all collected slices ----
        point_all = torch.cat(collected_point_blocks, dim=1)  # [B, prediction_length, F]

        quant_all = None
        if collected_quant_blocks:
            quant_all = torch.cat(collected_quant_blocks, dim=1)  # [B, prediction_length, F, Q]

        # stitch dict params if needed
        params_stacked: Union[torch.Tensor, Dict[str, torch.Tensor], None] = None
        if params_tensor_acc is not None:
            params_stacked = params_tensor_acc  # [B, sum_blocks, ...]
        elif params_dict_acc_tensors is not None:
            # concatenate dict-of-lists along time for each tensor field
            final_dict: Dict[str, Any] = {}
            for k_param, vs in params_dict_acc_tensors.items():
                final_dict[k_param] = torch.cat(vs, dim=1)  # [B, sum_blocks, ...]
            if params_dict_acc_meta:
                final_dict.update(params_dict_acc_meta)
            params_stacked = final_dict

        # Optional denormalization: point forecasts only
        if denormalize and not return_raw and hasattr(self.preprocessor, "denormalize"):
            if isinstance(point_all, torch.Tensor):
                point_all = self.preprocessor.denormalize(point_all)

        # Build ForecastBundle-like output
        extras_dict: Dict[str, Any] = {}
        try:
            extras_obj = HeadExtras(**extras_dict)
        except Exception:
            extras_obj = extras_dict  # fallback if HeadExtras signature changes

        bundle = ForecastBundle(
            point=point_all,            # [B, T, F]
            quantiles=quant_all,        # [B, T, F, Q] or None
            params=params_stacked,      # Tensor OR Dict[str,Tensor] OR None
            extras=extras_obj,
        )

        if return_bundle:
            return bundle
        if isinstance(quant_all, torch.Tensor):
            return quant_all
        return point_all


    # ---------------------------------------------------------------------
    # Convenience wrapper, symmetric with AutoregressiveStepwiseMixin.forecast
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        *,
        block_len: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
        validate_shapes: bool = True,
        verbose: bool = False,
    ) -> Union[
        ForecastBundle,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Convenience wrapper around `generate_blockwise`, just like
        AutoregressiveStepwiseMixin.forecast wraps its .generate.

        Usage:
            model.forecast_blockwise(
                inputs=batch_history,           # [B, T_hist, F]
                prediction_length=96,
                block_len=32,
                quantiles=[0.1,0.5,0.9],
                return_bundle=True,
            )

        Behavior:
            - If the model has an encoder (encoder-decoder arch), we pass `inputs`
              as `encoder_inputs` and seed the decoder with the last encoder step.
            - If the model is decoder-only, we pass `inputs` as `decoder_inputs`.

        Parameters
        ----------
        inputs : Tensor
            [B, T_hist, F] context window.
        prediction_length : int
            Total horizon to forecast.
        block_len : int | None
            How many steps to emit per block call. Defaults to config.prediction_length.
        quantiles : List[float] | None
            Quantile levels like [0.1,0.5,0.9]. If provided and supported,
            we'll stitch quantiles across blocks into [B, T, F, Q].
        attention_mask : Tensor | None
            Optional encoder attention mask [B, T_hist].
        decoder_attention_mask : Tensor | None
            Optional decoder attention mask [B, T_hist]. Usually same as attention_mask
            in decoder-only setups.
        denormalize : bool
            If True and model.preprocessor has .denormalize(), apply it to final point.
        return_bundle : bool
            If True, return ForecastBundle(point, quantiles, params, extras).
        return_raw : bool
            If False and denormalize=True, we denormalize point_all before returning.
        validate_shapes : bool
        verbose : bool

        Returns
        -------
        Same return contract as generate_blockwise():
          * ForecastBundle(...) if return_bundle=True
          * else quantile tensor [B,T,F,Q] if available
          * else point tensor [B,T,F]
        """

        if hasattr(self, "encoder") and self.encoder is not None:
            # encoder-decoder style: inputs go to encoder_inputs
            return self.generate_blockwise(
                encoder_inputs=inputs,
                decoder_inputs=None,  # we'll seed decoder from the last encoder step
                prediction_length=prediction_length,
                block_len=block_len,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                quantile_levels=quantiles,
                validate_shapes=validate_shapes,
                verbose=verbose,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
            )
        else:
            # decoder-only style: inputs go directly to decoder_inputs
            return self.generate_blockwise(
                encoder_inputs=None,
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                block_len=block_len,
                attention_mask=None,
                decoder_attention_mask=decoder_attention_mask if decoder_attention_mask is not None else attention_mask,
                quantile_levels=quantiles,
                validate_shapes=validate_shapes,
                verbose=verbose,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
            )
