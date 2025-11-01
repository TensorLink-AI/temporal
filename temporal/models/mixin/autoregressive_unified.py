# temporal/models/mixin/autoregressive_unified.py

import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveUnifiedMixin:
    """
    Unified autoregressive mixin that can do:
      • NON-PATCH blockwise rollout      ("blockwise")
      • PATCH AR in latent patch space   ("patch_ar")
      • PATCH-BLOCKWISE zero-shot style  ("patch_blockwise")

    Selection logic:
      1. If user passes mode=... -> we do that.
      2. Else if preprocessor looks patched -> default to "patch_blockwise"
         (because that's how you said you trained: zero-shot on patches).
      3. Else -> plain "blockwise".

    All modes try to return the same contract:
      - if return_bundle=True -> ForecastBundle(point, quantiles, params, extras)
      - else, if quantiles tensor is available -> that
      - else -> point tensor [B, T, F]
    """

    # ------------------------------------------------------------------
    # ----------- shared small helpers (pulled from both) --------------
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_levels(quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    def _get_primary_head(self) -> nn.Module:
        if not hasattr(self, "output_heads"):
            raise AttributeError("Model is missing output_heads.")
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

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
    def _normalize_quantile_shape(
        q: torch.Tensor,
        *,
        feature_size: int,
        Q: int,
    ) -> torch.Tensor:
        # normalize to [B, T, F, Q]
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
                raise RuntimeError(f"head.predict() failed for all methods {method_priority}: {last_err}")
        else:
            raise TypeError("Output head must implement .predict(...) to get a point forecast.")

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
        if not quantile_levels:
            return None

        if hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
            src = params_block
            if isinstance(src, dict):
                src = self._ensure_components_present(src, head)
            q_raw = head.sample_quantiles(src, quantile_levels)
            if not torch.is_tensor(q_raw):
                return None
            return self._normalize_quantile_shape(
                q_raw,
                feature_size=feature_size,
                Q=len(quantile_levels),
            )

        if torch.is_tensor(params_block):
            if params_block.shape[-1] == len(quantile_levels):
                return self._normalize_quantile_shape(
                    params_block,
                    feature_size=feature_size,
                    Q=len(quantile_levels),
                )
        return None

    # ------------------------------------------------------------------
    # -------------------- NON-PATCH BLOCKWISE ------------------------
    # ------------------------------------------------------------------
    def _generate_blockwise_nonpatch(
        self,
        *,
        encoder_inputs: Optional[torch.Tensor],
        decoder_inputs: Optional[torch.Tensor],
        prediction_length: int,
        block_len: int,
        attention_mask: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        quantile_levels: Optional[List[float]],
        validate_shapes: bool,
        verbose: bool,
        denormalize: bool,
        return_bundle: bool,
        return_raw: bool,
    ):
        ref = decoder_inputs if decoder_inputs is not None else encoder_inputs
        if ref is None:
            raise ValueError("You must provide either decoder_inputs or encoder_inputs.")

        B, device, dtype = ref.shape[0], ref.device, ref.dtype

        # 1) one-shot (fast path)
        if prediction_length <= block_len:
            fake_targets = torch.zeros(
                (B, prediction_length, getattr(self.config, "feature_size", ref.size(-1))),
                device=device,
                dtype=dtype,
            )
            out = self.forward(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                targets=fake_targets,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            block_params = out.logits
            primary_head = self._get_primary_head()
            Fsize = getattr(self.config, "feature_size", block_params.shape[-1] if torch.is_tensor(block_params) else 1)

            point_block = self._head_predict_point(primary_head, block_params)
            quant_block = None
            if quantile_levels:
                levels = self._normalize_levels(quantile_levels)
                quant_block = self._head_quantiles(primary_head, block_params, levels, feature_size=Fsize)

            if denormalize and not return_raw and hasattr(self.preprocessor, "denormalize"):
                point_block = self.preprocessor.denormalize(point_block)

            bundle = ForecastBundle(
                point=point_block,
                quantiles=quant_block,
                params=block_params,
                extras=HeadExtras(**{}) if "HeadExtras" in globals() else {},
            )
            if return_bundle:
                return bundle
            if quant_block is not None:
                return quant_block
            return point_block

        # 2) multi-block (slow path)
        remaining = prediction_length
        running_seq = decoder_inputs if decoder_inputs is not None else encoder_inputs[:, -1:, :].clone()

        collected_point: List[torch.Tensor] = []
        collected_quant: List[torch.Tensor] = []
        quant_norm = self._normalize_levels(quantile_levels) if quantile_levels else None

        params_tensor_acc: Optional[torch.Tensor] = None
        params_dict_acc_tensors: Optional[Dict[str, List[torch.Tensor]]] = None
        params_dict_acc_meta: Optional[Dict[str, Any]] = None

        def _accum_dict_step(acc_t, acc_m, block_d):
            if acc_t is None:
                acc_t = {}
            if acc_m is None:
                acc_m = {}
            for k, v in block_d.items():
                if torch.is_tensor(v):
                    vv = v
                    if vv.ndim == 2:
                        vv = vv.unsqueeze(1)
                    acc_t.setdefault(k, []).append(vv)
                else:
                    acc_m.setdefault(k, v)
            return acc_t, acc_m

        while remaining > 0:
            this_block = min(remaining, block_len)
            fake_targets = torch.zeros(
                (B, this_block, getattr(self.config, "feature_size", running_seq.size(-1))),
                device=device,
                dtype=dtype,
            )

            out = self.forward(
                encoder_inputs=encoder_inputs,
                decoder_inputs=running_seq,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                targets=fake_targets,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            block_params = out.logits
            primary_head = self._get_primary_head()
            Fsize = getattr(self.config, "feature_size", block_params.shape[-1] if torch.is_tensor(block_params) else 1)

            point_block = self._head_predict_point(primary_head, block_params)
            quant_block = None
            if quant_norm:
                quant_block = self._head_quantiles(primary_head, block_params, quant_norm, feature_size=Fsize)

            collected_point.append(point_block)
            if quant_block is not None:
                collected_quant.append(quant_block)

            # params accumulation
            if isinstance(block_params, dict):
                params_dict_acc_tensors, params_dict_acc_meta = _accum_dict_step(
                    params_dict_acc_tensors, params_dict_acc_meta, block_params
                )
            else:
                bp = block_params
                if bp.ndim == 2:
                    bp = bp.unsqueeze(1)
                if params_tensor_acc is None:
                    params_tensor_acc = bp
                else:
                    params_tensor_acc = torch.cat([params_tensor_acc, bp], dim=1)

            # feedback
            running_seq = torch.cat([running_seq, point_block], dim=1)
            remaining -= this_block

        point_all = torch.cat(collected_point, dim=1)
        quant_all = torch.cat(collected_quant, dim=1) if collected_quant else None

        # params stitching
        if params_tensor_acc is not None:
            params_stacked = params_tensor_acc
        elif params_dict_acc_tensors is not None:
            final_dict: Dict[str, Any] = {}
            for k, vs in params_dict_acc_tensors.items():
                final_dict[k] = torch.cat(vs, dim=1)
            if params_dict_acc_meta:
                final_dict.update(params_dict_acc_meta)
            params_stacked = final_dict
        else:
            params_stacked = None

        if denormalize and not return_raw and hasattr(self.preprocessor, "denormalize"):
            point_all = self.preprocessor.denormalize(point_all)

        bundle = ForecastBundle(
            point=point_all,
            quantiles=quant_all,
            params=params_stacked,
            extras=HeadExtras(**{}) if "HeadExtras" in globals() else {},
        )
        if return_bundle:
            return bundle
        if quant_all is not None:
            return quant_all
        return point_all

    # ------------------------------------------------------------------
    # -------------------- PATCH-BLOCKWISE (new) -----------------------
    # ------------------------------------------------------------------
    def _generate_patch_blockwise(
        self,
        *,
        decoder_inputs: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor],
        prediction_length: int,
        block_len: int,
        validate_shapes: bool,
        verbose: bool,
        denormalize: bool,
        return_bundle: bool,
        return_raw: bool,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_inputs: Optional[torch.Tensor] = None,
    ):
        """
        Repeated *training-style* patched forward calls:
          running_seq --(forward, target=this_block)--> predicted_block
          append -> repeat.
        Exactly what you said: "i trained to zero shot all patches - so generate
        must do that too, but in blocks".
        """
        B, device, dtype = decoder_inputs.shape[0], decoder_inputs.device, decoder_inputs.dtype
        F = getattr(self.config, "feature_size", decoder_inputs.size(-1))

        remaining = prediction_length
        running_seq = decoder_inputs
        collected: List[torch.Tensor] = []

        primary_head = self._get_primary_head()

        while remaining > 0:
            this_block = min(remaining, block_len)
            fake_targets = torch.zeros((B, this_block, F), device=device, dtype=dtype)

            out = self.forward(
                encoder_inputs=encoder_inputs,
                decoder_inputs=running_seq,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                targets=fake_targets,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            block_params = out.logits

            # turn params into point block
            if isinstance(block_params, dict):
                block_point = primary_head.predict(self._ensure_components_present(block_params, primary_head), method="median")
            else:
                # could already be [B, this_block, F]
                if hasattr(primary_head, "predict"):
                    block_point = primary_head.predict(block_params, method="median")
                else:
                    block_point = block_params

            if block_point.ndim == 2:
                block_point = block_point.unsqueeze(-1)
            if block_point.size(1) != this_block:
                block_point = block_point[:, -this_block:, :]

            if denormalize and not return_raw and hasattr(self.preprocessor, "denormalize"):
                try:
                    block_point = self.preprocessor.denormalize(block_point)
                except Exception:
                    pass

            collected.append(block_point)

            # feedback
            running_seq = torch.cat([running_seq, block_point], dim=1)
            if decoder_attention_mask is not None:
                block_mask = torch.ones(B, this_block, device=device, dtype=decoder_attention_mask.dtype)
                decoder_attention_mask = torch.cat([decoder_attention_mask, block_mask], dim=1)

            remaining -= this_block

        forecast = torch.cat(collected, dim=1)

        bundle = ForecastBundle(
            point=forecast,
            quantiles=None,
            params=None,
            extras=HeadExtras(**{}) if "HeadExtras" in globals() else {},
        )
        if return_bundle:
            return bundle
        return forecast

    # ------------------------------------------------------------------
    # --------------------- PATCH (old style) --------------------------
    # ------------------------------------------------------------------
    def _generate_patch_ar_latent(
        self,
        *,
        encoder_inputs: Optional[torch.Tensor],
        decoder_inputs: Optional[torch.Tensor],
        prediction_length: int,
        attention_mask: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        quantile_levels: Optional[List[float]],
        validate_shapes: bool,
        verbose: bool,
        sampling: bool,
        sampling_kwargs: Optional[Dict[str, Any]],
        return_raw: bool,
        return_bundle: bool,
    ):
        """
        This is basically your current AutoregressivePatchMixin.generate(...)
        but kept inside the unified mixin.
        """
        self.eval()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        ref = decoder_inputs if encoder_inputs is None else encoder_inputs
        B, device = ref.shape[0], ref.device

        if prediction_length == 0:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        patch_size = int(getattr(self.preprocessor, "patch_size", 1))
        num_patch_steps = (prediction_length + patch_size - 1) // patch_size

        # encoder path
        encoder_hidden_states = None
        if hasattr(self, "encoder") and self.encoder is not None and encoder_inputs is not None:
            enc_proc = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            context_patches = enc_proc["hidden_states"]
            enc_out = self.encoder(
                hidden_states=context_patches,
                attention_mask=enc_proc["attention_mask"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state
            decoder_patch_seq = context_patches[:, -1:, :]
        elif decoder_inputs is not None:
            dec_proc0 = self.preprocessor.process(
                input_values=decoder_inputs,
                attention_mask=attention_mask,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            decoder_patch_seq = dec_proc0["hidden_states"]
        else:
            raise ValueError("Could not determine architecture for patch AR.")

        generated_patches: List[torch.Tensor] = []
        past_key_values = None

        for _ in range(num_patch_steps):
            step_in = decoder_patch_seq[:, -1:, :] if (use_cache and past_key_values is not None) else decoder_patch_seq
            past_len = 0
            if past_key_values is not None:
                # simple len from first layer
                first_layer = past_key_values[0]
                key_tensor = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer
                past_len = key_tensor.shape[-2] if key_tensor.ndim == 4 else key_tensor.shape[1]

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
            next_patch = dec_out.last_hidden_state[:, -1:, :]
            generated_patches.append(next_patch)
            decoder_patch_seq = torch.cat([decoder_patch_seq, next_patch], dim=1)

            if use_cache:
                past_key_values = dec_out.past_key_values

        if not generated_patches:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        all_patches = torch.cat(generated_patches, dim=1)
        recon = self.output_patch_reconstructor(all_patches)

        if recon.ndim != 3:
            raise RuntimeError(f"output_patch_reconstructor must return [B,T_patch,d_model] or [B,T_patch,output_patch_size*d_model], got {tuple(recon.shape)}")

        B2, T_patch_out, proj = recon.shape
        if B2 != B:
            raise RuntimeError(f"Reconstructor batch mismatch: expected {B}, got {B2}")

        output_patch_size = getattr(getattr(self.preprocessor, "value_embedding", object), "output_patch_size", patch_size)
        d_model_cfg = getattr(self.config, "d_model", None)

        if d_model_cfg is not None and proj == output_patch_size * d_model_cfg:
            d_model = d_model_cfg
            head_inputs = recon.contiguous().view(B, T_patch_out * output_patch_size, d_model)
        elif d_model_cfg is not None and proj == d_model_cfg:
            d_model = d_model_cfg
            head_inputs = recon.unsqueeze(2).expand(B, T_patch_out, output_patch_size, d_model).reshape(
                B, T_patch_out * output_patch_size, d_model
            )
        else:
            if proj % output_patch_size != 0:
                raise RuntimeError(
                    f"Cannot infer d_model: proj={proj} not divisible by output_patch_size={output_patch_size}."
                )
            d_model = proj // output_patch_size
            head_inputs = recon.contiguous().view(B, T_patch_out * output_patch_size, d_model)

        head_inputs = head_inputs[:, :prediction_length, :]

        primary_head = self._get_primary_head()
        params_stacked = primary_head(head_inputs)
        if isinstance(params_stacked, dict):
            params_stacked = self._ensure_components_present(params_stacked, primary_head)

        levels = self._normalize_levels(quantile_levels)
        q_tensor: Optional[torch.Tensor] = None
        if levels is not None:
            out_q = None
            if hasattr(primary_head, "sample_quantiles"):
                try:
                    out_q = primary_head.sample_quantiles(params_stacked, levels)
                except Exception:
                    out_q = None
            if torch.is_tensor(out_q):
                q_tensor = self._normalize_quantile_shape(out_q, feature_size=getattr(self.config, "feature_size", 1), Q=len(levels))

        # point
        if sampling and hasattr(primary_head, "sample"):
            try:
                point = primary_head.sample(params_stacked, **(sampling_kwargs or {}))
            except Exception:
                point = self._head_predict_point(primary_head, params_stacked)
        else:
            if q_tensor is not None:
                mid = q_tensor.shape[-1] // 2
                point = q_tensor[..., mid]
                if point.ndim == 2:
                    point = point.unsqueeze(-1)
            else:
                point = self._head_predict_point(primary_head, params_stacked)

        if hasattr(self.preprocessor, "denormalize") and not return_raw:
            try:
                point = self.preprocessor.denormalize(point)
            except Exception:
                pass

        bundle = ForecastBundle(
            point=point,
            quantiles=q_tensor,
            params=params_stacked,
            extras=HeadExtras(**{}) if "HeadExtras" in globals() else {},
        )

        if return_bundle:
            return bundle
        if isinstance(q_tensor, torch.Tensor):
            return q_tensor
        return point

    # ------------------------------------------------------------------
    # ------------------------- PUBLIC API -----------------------------
    # ------------------------------------------------------------------
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
        # patch-only args:
        mode: Optional[str] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        sampling: bool = False,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Unified .generate().

        mode:
          - None               -> auto-detect
          - "blockwise"        -> non-patch blockwise (what we had)
          - "patch_ar"         -> old-style patch AR (latent AR loop)
          - "patch_blockwise"  -> repeated training-style patched blocks (new)
        """
        self.eval()

        if prediction_length is None:
            prediction_length = getattr(self.config, "prediction_length", None)
            if prediction_length is None:
                raise ValueError("prediction_length must be provided or defined in config.prediction_length")

        # guess if model is patched
        is_patched = hasattr(self, "preprocessor") and hasattr(self.preprocessor, "patch_size")

        if block_len is None:
            # for both patched and non-patched default to train horizon
            block_len = getattr(self.config, "prediction_length", prediction_length)

        if mode is None:
            if is_patched:
                # you said: "i trained my patch model to zero-shot all patches"
                # so default to the patched blockwise strategy
                mode = "patch_blockwise"
            else:
                mode = "blockwise"

        if mode == "blockwise":
            return self._generate_blockwise_nonpatch(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                prediction_length=prediction_length,
                block_len=block_len,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                quantile_levels=quantile_levels,
                validate_shapes=validate_shapes,
                verbose=verbose,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
            )

        if mode == "patch_blockwise":
            # we need a native sequence to start from
            if decoder_inputs is None:
                if encoder_inputs is not None:
                    decoder_inputs = encoder_inputs[:, -1:, :].clone()
                else:
                    raise ValueError("patch_blockwise mode needs decoder_inputs or encoder_inputs to seed from")
            return self._generate_patch_blockwise(
                decoder_inputs=decoder_inputs,
                decoder_attention_mask=decoder_attention_mask,
                prediction_length=prediction_length,
                block_len=block_len,
                validate_shapes=validate_shapes,
                verbose=verbose,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                attention_mask=attention_mask,
                encoder_inputs=encoder_inputs,
            )

        if mode == "patch_ar":
            return self._generate_patch_ar_latent(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                prediction_length=prediction_length,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                quantile_levels=quantile_levels,
                validate_shapes=validate_shapes,
                verbose=verbose,
                sampling=sampling,
                sampling_kwargs=sampling_kwargs,
                return_raw=return_raw,
                return_bundle=return_bundle,
            )

        raise ValueError(f"Unknown generation mode: {mode}")

    # keep symmetry
    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        *,
        mode: Optional[str] = None,
        block_len: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
        validate_shapes: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        if hasattr(self, "encoder") and self.encoder is not None:
            return self.generate(
                encoder_inputs=inputs,
                decoder_inputs=None,
                prediction_length=prediction_length,
                mode=mode,
                block_len=block_len,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                quantile_levels=quantiles,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                validate_shapes=validate_shapes,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.generate(
                encoder_inputs=None,
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                mode=mode,
                block_len=block_len,
                attention_mask=None,
                decoder_attention_mask=decoder_attention_mask if decoder_attention_mask is not None else attention_mask,
                quantile_levels=quantiles,
                denormalize=denormalize,
                return_bundle=return_bundle,
                return_raw=return_raw,
                validate_shapes=validate_shapes,
                verbose=verbose,
                **kwargs,
            )
