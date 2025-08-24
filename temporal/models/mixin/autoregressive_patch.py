import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressivePatchMixin:
    """
    Patch-based autoregressive generation that returns a consistent ForecastBundle.

    This mixin is designed for models that autoregress in a *patch latent space*
    (decoder predicts the next patch embedding), then reconstruct the native
    time-domain sequence from the generated patch embeddings, and finally apply
    probabilistic output heads (e.g., Gaussian / StudentT / MDN / DistPred).

    Assumptions about the host model
    --------------------------------
    • self.config
        - feature_size: int          (# target features/channels F)
        - d_model: int               (head input size after reconstruction)
        - prediction_length: Optional[int]
        - num_attention_heads: Optional[int]
    • self.preprocessor
        - patch_size: int
        - process(input_values, attention_mask, is_causal, ...)
        - _prepare_decoder_inputs_for_generation(patch_embeds, attention_mask, past_key_values_length, is_causal)
        - denormalize(tensor)   # optional; applied to point only
    • self.encoder (optional) and self.decoder: HuggingFace-style
        - return_dict=True objects with .last_hidden_state and .past_key_values
    • self.output_patch_reconstructor(patch_embeds) -> Tensor
        - Typically maps [B, T_patch, D_latent] -> [B, T_patch, d_model]
          which we then reshape into [B, T_native, d_model]
    • self.output_heads: nn.Module or nn.ModuleList
        - forward(x_last) returns tensor OR dict of params
        - predict(params, method=...) -> [B, T, F]
        - sample(params, **kwargs)    -> [B, T, F]   (vectorized heads only; see notes)
        - sample_quantiles(params, quantile_levels) -> [B, T, 1/F?, Q]  (normalized here)
    • optional: self.head_aggregator(list_of_head_outputs) -> head_output

    Key guarantees
    --------------
    * bundle.point is always a Tensor [B, T, F]
    * bundle.quantiles is either None or a Tensor [B, T, F, Q]
    * bundle.params is the stacked raw *primary head* outputs:
        - Tensor for Gaussian/StudentT/QuantileRegression/etc.
        - Dict[str, Tensor] for Mixture (MDN) / DistPred
    * No dicts are ever placed into bundle.quantiles (prevents `.shape` errors)

    Notes on sampling with patch generation
    ---------------------------------------
    In this patch paradigm the decoder’s AR loop happens in *latent patch space*.
    Output heads are applied *after* the full horizon is reconstructed.
    Therefore, head.sample(...) does **not** influence the decoder’s next state.
    We still allow sampling to produce a sample path as the returned `point`
    (when the head supports *vectorized* sampling on [B, T, ...] params, e.g.,
    Gaussian/StudentT). For heads that implement only one-step sampling
    (e.g., MDN sample() that returns [B,1,1]), we fall back to predict(...).
    """

    # ---------------------------------------------------------------------
    # Basic utilities (parity with stepwise mixin)
    # ---------------------------------------------------------------------
    def enable_dropout(self) -> None:
        """Enable dropout layers (for MC dropout)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_primary_head(self) -> nn.Module:
        """Return the first (primary) head."""
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    @staticmethod
    def _get_cache_length(past_key_values) -> int:
        """
        Infer current KV cache length L from a HuggingFace-style past_key_values
        structure by looking at the first layer's key tensor.

        Returns
        -------
        int
        """
        if past_key_values is None:
            return 0

        first_layer = past_key_values[0]
        # Common shapes: tuple(list) of (k, v, ...) or dict with "k"
        if isinstance(first_layer, (tuple, list)):
            key_tensor = first_layer[0]
        elif isinstance(first_layer, dict):
            key_tensor = first_layer.get("k", None)
        else:
            key_tensor = None

        if key_tensor is None:
            raise ValueError(f"Cannot locate key tensor in past_key_values[0]: {type(first_layer)}")

        # (B, H, L, D) or (B, L, H, D) or (B, L, D)
        if key_tensor.ndim == 4:
            # Try to detect H dimension from config, else pick the larger of dims 1/2
            nh = getattr(getattr(self, "config", object), "num_attention_heads", None)
            if nh is not None:
                if key_tensor.shape[1] == nh:
                    return int(key_tensor.shape[2])
                if key_tensor.shape[2] == nh:
                    return int(key_tensor.shape[1])
            return int(max(key_tensor.shape[1], key_tensor.shape[2]))
        if key_tensor.ndim == 3:
            return int(key_tensor.shape[1])

        raise ValueError(f"Unexpected KV shape: {tuple(key_tensor.shape)}")

    @staticmethod
    def _get_scalar_value(value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
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

    @staticmethod
    def _normalize_levels(quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
        """Validate and sort requested quantile levels."""
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    @staticmethod
    def _ensure_b1f(x: torch.Tensor, feature_size: int) -> torch.Tensor:
        """
        Ensure feedback shape [B, 1, F] from common variants (not used by patch AR loop,
        but kept for parity and potential re-use).
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

    @staticmethod
    def _ensure_components_present(params: Dict[str, Any], head: nn.Module) -> Dict[str, Any]:
        """
        Ensure a MDN/DistPred-like dict has a 'components' key when the head exposes it.
        """
        if "components" not in params and hasattr(head, "components"):
            try:
                params = dict(params)  # shallow copy
                params["components"] = list(getattr(head, "components"))
            except Exception:
                pass
        return params

    @staticmethod
    def _normalize_quantile_shape(q: torch.Tensor, *, feature_size: int, Q: int) -> torch.Tensor:
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

    # ---------------------------------------------------------------------
    # Post-quantiles & point extraction (parity with stepwise mixin)
    # ---------------------------------------------------------------------
    def _post_quantiles_any(
        self,
        params_or_preds: Union[torch.Tensor, Dict[str, Any], List[Any]],
        head: nn.Module,
        quantile_levels: List[float],
    ) -> Union[torch.Tensor, Dict[str, Any], List[Any]]:
        """
        Compute quantiles once on stacked params for tensor OR dict OR list-of-heads.

        Returns a tensor [B, T, F, Q] when the head supports quantiles. For ModuleList
        setups, you can adapt to per-head if needed. By design we bundle *primary head*
        quantiles.
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
                return out  # if a head returns a non-tensor (we don't place into bundle.quantiles)
            except Exception as e:
                logger.warning(f"Post-quantiles on primary head failed; returning raw params. Error: {e}")
                return params_or_preds

        return params_or_preds

    def _compute_point_from_params(
        self,
        source: Union[torch.Tensor, Dict[str, Any], List[Any], None],
        head: nn.Module,
        *,
        quantiles_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute a point forecast either from a given quantile tensor (median slice) or
        by calling head.predict(...).
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
    # Generation (patch latent AR, then heads on full horizon)
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
        decoder_start_token_id: Optional[Any] = None,   # compatibility only
        eos_token_id: Optional[Any] = None,             # unused in patch AR; kept for parity
        early_stopping: bool = False,                   # unused in patch AR; kept for parity
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        prediction_strategy: Optional[Union[str, float, int]] = None,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        *,
        sampling: bool = False,                         # sampling affects returned point only (see class doc)
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        store_sampled: bool = False,                    # no-op in patch AR; kept for API parity
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
        Patch-based autoregressive forecast.

        Steps
        -----
        1) Encode context (if encoder exists) and seed the decoder with the last
           context patch (enc-dec) or the provided decoder_inputs (dec-only).
        2) Autoregressively predict *patch embeddings* for ceil(pred_len / patch_size) steps.
        3) Reconstruct native-time embeddings and reshape to [B, T_native, d_model].
        4) Apply primary output head on the full horizon to get params (tensor/dict).
        5) Optionally compute quantiles once on the stacked params (MDN-safe).
        6) Compute a point forecast via:
              • sampling (vectorized heads only) OR
              • predict(method=prediction_strategy or 'median') OR
              • median from computed quantiles
        7) Optionally denormalize the point.
        8) Return ForecastBundle or (quantiles if tensor) else point.

        Important
        ---------
        • sampling=True does NOT influence the decoder’s AR loop here. It only
          changes how we produce the final `point` from the stacked head params.
        """
        self.eval()
        if enable_mc_dropout:
            self.enable_dropout()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        if prediction_length is None:
            prediction_length = getattr(self.config, "prediction_length", 0)

        ref = decoder_inputs if encoder_inputs is None else encoder_inputs
        B, device = ref.shape[0], ref.device

        if prediction_length == 0:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        patch_size = int(getattr(self.preprocessor, "patch_size", 1))
        num_patch_steps = (prediction_length + patch_size - 1) // patch_size  # ceil-div

        # --------- Encoder path (if present) to get context patches ---------
        encoder_hidden_states = None
        if hasattr(self, "encoder") and self.encoder is not None and encoder_inputs is not None:
            enc_proc = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            context_patches = enc_proc["hidden_states"]  # [B, T_ctx_patch, D_latent]
            enc_out = self.encoder(
                hidden_states=context_patches,
                attention_mask=enc_proc["attention_mask"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state
            # Seed decoder with last context patch
            decoder_patch_seq = context_patches[:, -1:, :]  # [B, 1, D_latent]
        elif decoder_inputs is not None:
            dec_proc0 = self.preprocessor.process(
                input_values=decoder_inputs,
                attention_mask=attention_mask,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            decoder_patch_seq = dec_proc0["hidden_states"]  # [B, T0_patch, D_latent]
        else:
            raise ValueError("Could not determine architecture: provide encoder_inputs (for enc-dec) or decoder_inputs (for dec-only).")

        # --------- AR loop in patch space ---------
        generated_patches: List[torch.Tensor] = []
        past_key_values = None

        for _ in range(num_patch_steps):
            step_in = decoder_patch_seq[:, -1:, :] if (use_cache and past_key_values is not None) else decoder_patch_seq
            past_len = self._get_cache_length(past_key_values)

            dec_proc = self.preprocessor._prepare_decoder_inputs_for_generation(
                patch_embeds=step_in,
                attention_mask=(None if (use_cache and past_key_values is not None) else decoder_attention_mask),
                past_key_values_length=past_len,
                is_causal=True,
            )
            dec_out = self.decoder(
                hidden_states=dec_proc["hidden_states"],
                attention_mask=dec_proc["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            next_patch = dec_out.last_hidden_state[:, -1:, :]  # [B,1,D_latent]
            generated_patches.append(next_patch)
            decoder_patch_seq = torch.cat([decoder_patch_seq, next_patch], dim=1)

            if use_cache:
                past_key_values = dec_out.past_key_values

        if not generated_patches:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        # --------- Reconstruct native-time embeddings ---------
        all_patches = torch.cat(generated_patches, dim=1)  # [B, T_patch_out, D_latent]
        recon = self.output_patch_reconstructor(all_patches)  # expected [B, T_patch_out, d_model]

        if recon.ndim != 3:
            raise RuntimeError(f"output_patch_reconstructor must return [B,T_patch,d_model], got {tuple(recon.shape)}")

        B2, T_patch_out, Dm = recon.shape
        if B2 != B:
            raise RuntimeError(f"Reconstructor batch mismatch: expected {B}, got {B2}")

        # Turn patches into per-time-step hidden states [B, T_native, d_model]
        T_native = T_patch_out * patch_size
        try:
            # View assumes each patch expands to `patch_size` time-steps with the same Dm
            head_inputs = recon.view(B, T_native, Dm)
        except Exception as e:
            raise RuntimeError(
                f"Failed to reshape reconstructed patches to [B,{T_native},{Dm}] "
                f"using patch_size={patch_size}. Ensure your reconstructor outputs contiguous "
                f"[B, T_patch, d_model] and that T_native=T_patch*patch_size. Error: {e}"
            )

        # Trim to requested horizon
        head_inputs = head_inputs[:, :prediction_length, :]  # [B, T, d_model]

        # --------- Apply primary head on full horizon (get params) ---------
        primary_head = self._get_primary_head()
        params_stacked = primary_head(head_inputs)  # tensor OR dict, usually [B, T, ...] / dict of [B, T, ...]
        if isinstance(params_stacked, dict):
            params_stacked = self._ensure_components_present(params_stacked, primary_head)

        # --------- Quantiles (once; MDN-safe) ---------
        levels = self._normalize_levels(quantile_levels)
        q_tensor: Optional[torch.Tensor] = None
        if levels is not None and post_quantiles:
            out_q = self._post_quantiles_any(params_stacked, primary_head, levels)
            if torch.is_tensor(out_q):
                # normalize to [B, T, F, Q]
                q_tensor = self._normalize_quantile_shape(out_q, feature_size=getattr(self.config, "feature_size", 1), Q=len(levels))
            else:
                # If a head returns non-tensor (shouldn't happen), skip bundling in quantiles
                q_tensor = None

        # --------- Point forecast ---------
        # Prefer sampling when requested and supported *vectorized*; else predict; else median from q_tensor.
        point: torch.Tensor
        if sampling and hasattr(primary_head, "sample"):
            try:
                # Vectorized sampling over [B, T, ...] params (e.g., Gaussian/StudentT heads)
                sampled = primary_head.sample(params_stacked, **(sampling_kwargs or {}))
                if not torch.is_tensor(sampled):
                    raise TypeError("head.sample(...) did not return a Tensor.")
                point = sampled  # expected [B, T, F]
            except Exception as e:
                logger.warning(f"Vectorized sampling failed for primary head; falling back to predict(...). Error: {e}")
                point = self._compute_point_from_params(params_stacked, primary_head, quantiles_tensor=q_tensor)
        else:
            point = self._compute_point_from_params(params_stacked, primary_head, quantiles_tensor=q_tensor)

        # --------- Optional denormalization (point only) ---------
        if hasattr(self.preprocessor, "denormalize") and not return_raw:
            if isinstance(point, torch.Tensor) and point.size(-1) == getattr(self.config, "feature_size", point.size(-1)):
                try:
                    point = self.preprocessor.denormalize(point)
                except Exception as e:
                    logger.warning(f"Denormalization failed; returning raw point. Error: {e}")

        # --------- Bundle ---------
        extras_dict: Dict[str, Any] = {}
        try:
            extras_obj = HeadExtras(**extras_dict)  # type: ignore[arg-type]
        except Exception:
            extras_obj = extras_dict  # type: ignore[assignment]

        bundle = ForecastBundle(
            point=point,                 # [B, T, F]
            quantiles=q_tensor,          # [B, T, F, Q] or None
            params=params_stacked,       # Tensor for Gaussian/StudentT/QR; Dict for MDN/DistPred
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
            Input sequence [B, T, F] (for encoder-decoder: historical context,
            for decoder-only: prompt).
        prediction_length : int
            Number of steps to forecast in native time steps.
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
