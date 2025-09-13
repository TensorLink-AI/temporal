import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveLatentMixin:
    """
    Stepwise autoregressive generation in *latent space* for non-patch models.

    This approach is highly efficient as it avoids repeated projections to the
    output space and back during the generation loop.

    Pipeline
    --------
    1) Build initial latent sequence from context (encoder+preprocessor OR decoder_inputs+preprocessor).
    2) Warm the decoder cache by running the *full* context's latent sequence through it once.
    3) Iterate `prediction_length` times:
        - Take the previous step's *decoder last hidden state* (a single latent vector).
        - Prepare it as a single-token input (adding position info, etc.).
        - Decode one step using the KV cache to get the next latent vector.
        - Collect the new latent vector.
    4) After the loop, the full sequence of generated latent vectors is collected.
    5) Run the output head **once** on the full sequence of generated latents.
    6) Compute final point forecasts and optional quantiles from the head's output.
    7) Optionally denormalize the results.

    Requirements on host model
    --------------------------
    self.config:
      - d_model: int
      - feature_size: int
      - prediction_length: Optional[int]
      - num_attention_heads: Optional[int]
    self.preprocessor:
      - process(input_values, attention_mask, is_causal, ...)
      - _prepare_decoder_inputs_for_generation(hidden_states, attention_mask, past_key_values_length, is_causal)
      - denormalize(tensor)
    self.encoder / self.decoder: HuggingFace-style returning .last_hidden_state, .past_key_values
    self.output_heads: nn.Module or nn.ModuleList with:
        - forward(x) -> tensor or dict
        - predict(params, method=...) -> [B, T, F]
        - sample(params, **kwargs) -> [B, T, F] (optional vectorized)
        - sample_quantiles(params, quantile_levels) -> [B, T, F?, Q] (tensor preferred)
    """

    # ---------------------------------------------------------------------
    # Utilities (mostly identical to the patch version)
    # ---------------------------------------------------------------------
    def enable_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_primary_head(self) -> nn.Module:
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    def _get_cache_length(self, past_key_values) -> int:
        """Infer cached length L from various KV layouts."""
        if past_key_values is None:
            return 0
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)):
            key_tensor = first_layer[0]
        elif isinstance(first_layer, dict):
            key_tensor = first_layer.get("k") or first_layer.get("key")
            if key_tensor is None:
                try:
                    key_tensor = next(iter(first_layer.values()))
                except Exception:
                    key_tensor = None
        else:
            key_tensor = first_layer
        if key_tensor is None:
            return 0

        ndim = key_tensor.ndim
        nh = getattr(getattr(self, "config", object), "num_attention_heads", None)
        if ndim == 4:
            _, a1, a2, _ = key_tensor.shape  # (B,H,L,D) or (B,L,H,D)
            if nh is not None:
                if a1 == nh and a2 != nh: return int(a2)
                if a2 == nh and a1 != nh: return int(a1)
            return int(max(a1, a2))
        if ndim == 3:
            _, a1, a2 = key_tensor.shape      # (B,L,D) or (B,D,L)
            d_model = getattr(getattr(self, "config", object), "d_model", None)
            if d_model is not None:
                if a2 == d_model: return int(a1)
                if a1 == d_model: return int(a2)
            return int(max(a1, a2))
        return 0

    @staticmethod
    def _normalize_levels(quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    @staticmethod
    def _ensure_components_present(params: Dict[str, Any], head: nn.Module) -> Dict[str, Any]:
        if "components" not in params and hasattr(head, "components"):
            try:
                params = dict(params)
                params["components"] = list(getattr(head, "components"))
            except Exception:
                pass
        return params

    @staticmethod
    def _normalize_quantile_shape(q: torch.Tensor, *, feature_size: int, Q: int) -> torch.Tensor:
        """
        -> [B, T, F, Q]
        accepts [B,T,Q], [B,T,F,Q], [B,T,Q,F], [B,Q]
        """
        if q.ndim == 4:
            if q.shape[-1] == Q: return q
            if q.shape[-2] == Q: return q.permute(0, 1, 3, 2)
            return q
        if q.ndim == 3 and q.shape[-1] == Q:
            return q.unsqueeze(-2)      # [B,T,Q] -> [B,T,1,Q]
        if q.ndim == 2 and q.shape[-1] == Q:
            return q.unsqueeze(1).unsqueeze(2)  # [B,Q] -> [B,1,1,Q]
        raise ValueError(f"Cannot normalize quantile tensor of shape {tuple(q.shape)} to [B,T,F,Q].")

    def _post_quantiles_any(
        self,
        params_or_preds: Union[torch.Tensor, Dict[str, Any], List[Any]],
        head: nn.Module,
        quantile_levels: List[float],
    ) -> Union[torch.Tensor, Dict[str, Any], List[Any]]:
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
                logger.warning(f"Post-quantiles failed; returning raw params. Error: {e}")
                return params_or_preds
        return params_or_preds

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
            return point if point.ndim == 3 else point.unsqueeze(-1)
        if source is not None and hasattr(head, "predict"):
            src = source
            if isinstance(src, dict):
                src = self._ensure_components_present(src, head)
            try:
                pv = head.predict(src, method="median")
            except Exception:
                pv = head.predict(src, method="mean")
            return pv if pv.ndim == 3 else pv.unsqueeze(-1)
        raise TypeError("Cannot compute point forecast; head needs predict() or quantiles_tensor must be provided.")

    # ---------------------------------------------------------------------
    # Latent Stepwise AR
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
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        *,
        sampling: bool = False,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        return_raw: bool = False,
        return_bundle: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], ForecastBundle, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Latent stepwise AR with proper cache warm-up + single-token stepping.
        """
        self.eval()
        if kwargs.get("enable_mc_dropout", False):
            self.enable_dropout()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("Provide either 'encoder_inputs' (enc-dec) or 'decoder_inputs' (dec-only).")

        if prediction_length is None:
            prediction_length = getattr(self.config, "prediction_length", 0)

        ref = decoder_inputs if encoder_inputs is None else encoder_inputs
        B, device = ref.shape[0], ref.device

        if prediction_length == 0:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        # --------- 1. Build initial latent sequence from context ---------
        encoder_hidden_states = None
        if hasattr(self, "encoder") and self.encoder is not None and encoder_inputs is not None:
            enc_proc = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            initial_latents = enc_proc["hidden_states"]  # [B, T_ctx, D_latent]
            enc_out = self.encoder(
                hidden_states=initial_latents,
                attention_mask=enc_proc["attention_mask"],
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state
        elif decoder_inputs is not None:
            dec_proc0 = self.preprocessor.process(
                input_values=decoder_inputs,
                attention_mask=decoder_attention_mask,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            initial_latents = dec_proc0["hidden_states"]  # [B, T_ctx, D_latent]
        else:
            raise ValueError("Could not determine architecture or missing inputs.")

        # --------- 2. Warm the decoder cache ONCE with the full context ---------
        warm_out = self.decoder(
            hidden_states=initial_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )
        past_key_values = warm_out.past_key_values if use_cache else None
        next_hidden_state = warm_out.last_hidden_state[:, -1:, :]  # Seed for the loop [B, 1, D_latent]

        # --------- 3. Single-token AR loop in latent space ---------
        generated_latents: List[torch.Tensor] = []
        for _ in range(prediction_length):
            past_len = self._get_cache_length(past_key_values)
            
            # The preprocessor adds positional encodings for the new step
            step_input = self.preprocessor._prepare_decoder_inputs_for_generation(
                hidden_states=next_hidden_state,
                past_key_values_length=past_len,
            )

            dec_out = self.decoder(
                hidden_states=step_input["hidden_states"],
                attention_mask=step_input["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            
            next_hidden_state = dec_out.last_hidden_state  # [B, 1, D_latent]
            generated_latents.append(next_hidden_state)
            if use_cache:
                past_key_values = dec_out.past_key_values

        if not generated_latents:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        # --------- 4. Assemble latent sequence and run Head ONCE ---------
        head_inputs = torch.cat(generated_latents, dim=1)  # [B, prediction_length, D_latent]
        
        primary_head = self._get_primary_head()
        params_stacked = primary_head(head_inputs)
        if isinstance(params_stacked, dict):
            params_stacked = self._ensure_components_present(params_stacked, primary_head)

        # --------- 5. Compute quantiles and point forecast from params ---------
        levels = self._normalize_levels(quantile_levels)
        q_tensor: Optional[torch.Tensor] = None
        if levels:
            out_q = self._post_quantiles_any(params_stacked, primary_head, levels)
            if torch.is_tensor(out_q):
                q_tensor = self._normalize_quantile_shape(
                    out_q,
                    feature_size=getattr(self.config, "feature_size", 1),
                    Q=len(levels),
                )
        
        if sampling and hasattr(primary_head, "sample"):
            try:
                point = primary_head.sample(params_stacked, **(sampling_kwargs or {}))
                if not torch.is_tensor(point):
                    raise TypeError("head.sample(...) did not return a Tensor.")
            except Exception as e:
                logger.warning(f"Vectorized sampling failed; falling back to predict(...). Error: {e}")
                point = self._compute_point_from_params(params_stacked, primary_head, quantiles_tensor=q_tensor)
        else:
            point = self._compute_point_from_params(params_stacked, primary_head, quantiles_tensor=q_tensor)

        if point.ndim == 2:
            point = point.unsqueeze(-1)  # [B,T] -> [B,T,1]

        # --------- 6. Optional denormalization ---------
        if hasattr(self.preprocessor, "denormalize") and not return_raw:
            try:
                point = self.preprocessor.denormalize(point)
            except Exception as e:
                logger.warning(f"Denormalization failed; returning raw point. Error: {e}")
            if isinstance(q_tensor, torch.Tensor):
                try:
                    q_tensor = self.preprocessor.denormalize(q_tensor)
                except Exception as e:
                    logger.warning(f"Quantile denormalization failed; returning raw quantiles. Error: {e}")
        
        # --------- 7. Bundle and Return ---------
        bundle = ForecastBundle(
            point=point,
            quantiles=q_tensor,
            params=params_stacked,
            extras=HeadExtras(),
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
        Convenience wrapper around `generate`.
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
