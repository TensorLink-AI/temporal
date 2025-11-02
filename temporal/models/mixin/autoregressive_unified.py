import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveUnifiedMixin:
    """
    One mixin to handle:
      - non-patch stepwise AR
      - non-patch blockwise AR
      - patch one-shot (your current patch generate)
      - patch blockwise / rollout (patch but longer than train horizon)

    Use:
        model.forecast(x, prediction_length=256, mode="patch_blockwise")
    """

    # ------------------------------------------------------------------
    #  Common helpers
    # ------------------------------------------------------------------
    def _is_patch_model(self) -> bool:
        return (
            hasattr(self, "preprocessor")
            and hasattr(self.preprocessor, "value_embedding")
            and hasattr(self.preprocessor.value_embedding, "patch_size")
        )

    def _normalize_levels(self, quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
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

    def _ensure_components_present(self, params: Dict[str, Any], head: nn.Module) -> Dict[str, Any]:
        if "components" not in params and hasattr(head, "components"):
            try:
                params = dict(params)
                params["components"] = list(getattr(head, "components"))
            except Exception:
                pass
        return params

    # ==================================================================
    #  1) NON-PATCH BLOCKWISE  (your existing logic, kept intact)
    # ==================================================================
    @torch.no_grad()
    def _generate_blockwise(
        self,
        *,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: int,
        block_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        quantile_levels: Optional[List[float]] = None,
        validate_shapes: bool = True,
        verbose: bool = False,
        denormalize: bool = False,
        return_bundle: bool = False,
        return_raw: bool = False,
    ):
        """
        This is basically your AutoregressiveBlockwiseMixin.generate(...)
        but inlined so the unified mixin can call it.
        """
        self.eval()

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
                is_causal=False,
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
            encoder_hidden_states = enc_out.last_hidden_state

        # ---- seed decoder_inputs if missing ----
        if decoder_inputs is None:
            if encoder_inputs is not None:
                decoder_inputs = encoder_inputs[:, -1:, :].clone()
            else:
                raise ValueError("Decoder-only generation requires decoder_inputs")

        # ---- FAST PATH: one-shot parallel forecast ----
        if prediction_length <= block_len:
            fake_targets = torch.zeros(
                (decoder_inputs.size(0), prediction_length,
                 getattr(self.config, "feature_size", decoder_inputs.size(-1))),
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

            point_block_full = self._head_predict_point(primary_head, block_params)
            quant_all = None
            if quantile_levels:
                q_levels = self._normalize_levels(quantile_levels)
                quant_all = self._head_quantiles(primary_head, block_params, q_levels, feature_size=Fsize)

            if denormalize and not return_raw and hasattr(self.preprocessor, "denormalize"):
                point_block_full = self.preprocessor.denormalize(point_block_full)

            extras_obj = HeadExtras(**{}) if "HeadExtras" in globals() else {}
            bundle = ForecastBundle(
                point=point_block_full,
                quantiles=quant_all,
                params=block_params,
                extras=extras_obj,
            )
            if return_bundle:
                return bundle
            if quant_all is not None and torch.is_tensor(quant_all):
                return quant_all
            return point_block_full

        # ---- SLOW PATH: coarse AR over blocks ----
        remaining = prediction_length
        collected_point_blocks: List[torch.Tensor] = []
        collected_quant_blocks: List[torch.Tensor] = []
        quantile_levels_norm = self._normalize_levels(quantile_levels) if quantile_levels else None

        params_tensor_acc: Optional[torch.Tensor] = None
        params_dict_acc_tensors: Optional[Dict[str, List[torch.Tensor]]] = None
        params_dict_acc_meta: Optional[Dict[str, Any]] = None

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
                    vv = v
                    if vv.ndim == 2:
                        vv = vv.unsqueeze(1)
                    acc_tensors.setdefault(k, []).append(vv)
                else:
                    acc_meta.setdefault(k, v)
            return acc_tensors, acc_meta

        running_seq = decoder_inputs

        while remaining > 0:
            this_block = min(remaining, block_len)
            fake_targets = torch.zeros(
                (running_seq.size(0), this_block,
                 getattr(self.config, "feature_size", running_seq.size(-1))),
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
            Fsize = getattr(self.config, "feature_size",
                            block_params.shape[-1] if torch.is_tensor(block_params) else 1)

            point_block_full = self._head_predict_point(primary_head, block_params)
            if quantile_levels_norm:
                quant_block_full = self._head_quantiles(primary_head, block_params, quantile_levels_norm, feature_size=Fsize)
            else:
                quant_block_full = None

            collected_point_blocks.append(point_block_full)
            if quant_block_full is not None:
                collected_quant_blocks.append(quant_block_full)

            if isinstance(block_params, dict):
                params_dict_acc_tensors, params_dict_acc_meta = _accum_dict_step(
                    params_dict_acc_tensors,
                    params_dict_acc_meta,
                    block_params,
                )
            else:
                bp_for_stack = block_params
                if bp_for_stack.ndim == 2:
                    bp_for_stack = bp_for_stack.unsqueeze(1)
                if params_tensor_acc is None:
                    params_tensor_acc = bp_for_stack
                else:
                    params_tensor_acc = torch.cat([params_tensor_acc, bp_for_stack], dim=1)

            running_seq = torch.cat([running_seq, point_block_full], dim=1)
            remaining -= this_block

        point_all = torch.cat(collected_point_blocks, dim=1)
        quant_all = None
        if collected_quant_blocks:
            quant_all = torch.cat(collected_quant_blocks, dim=1)

        params_stacked: Union[torch.Tensor, Dict[str, torch.Tensor], None] = None
        if params_tensor_acc is not None:
            params_stacked = params_tensor_acc
        elif params_dict_acc_tensors is not None:
            final_dict: Dict[str, Any] = {}
            for k_param, vs in params_dict_acc_tensors.items():
                final_dict[k_param] = torch.cat(vs, dim=1)
            if params_dict_acc_meta:
                final_dict.update(params_dict_acc_meta)
            params_stacked = final_dict

        if denormalize and not return_raw and hasattr(self.preprocessor, "denormalize"):
            point_all = self.preprocessor.denormalize(point_all)

        extras_obj = HeadExtras(**{}) if "HeadExtras" in globals() else {}
        bundle = ForecastBundle(
            point=point_all,
            quantiles=quant_all,
            params=params_stacked,
            extras=extras_obj,
        )
        if return_bundle:
            return bundle
        if quant_all is not None and torch.is_tensor(quant_all):
            return quant_all
        return point_all

    # -- helpers used above ------------------------------------
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
                raise RuntimeError(f"head.predict() failed: {last_err}")
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
            return self._normalize_quantile_shape(q_raw, feature_size=feature_size, Q=len(quantile_levels))
        if torch.is_tensor(params_block) and params_block.shape[-1] == len(quantile_levels):
            return self._normalize_quantile_shape(params_block, feature_size=feature_size, Q=len(quantile_levels))
        return None

    def _normalize_quantile_shape(self, q: torch.Tensor, *, feature_size: int, Q: int) -> torch.Tensor:
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

    # ==================================================================
    #  2) PATCH ONE-SHOT  (your current AutoregressivePatchMixin.generate)
    # ==================================================================
    @torch.no_grad()
    def _generate_patch_one_shot(
        self,
        *,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        quantile_levels: Optional[List[float]] = None,
        sampling: bool = False,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        return_raw: bool = False,
        post_quantiles: bool = True,
        return_bundle: bool = False,
        enable_mc_dropout: bool = False,
    ):
        # This is the same as your "unupdated patch version" but trimmed a bit
        self.eval()
        if enable_mc_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        if encoder_inputs is None and decoder_inputs is None:
            raise ValueError("You must provide either 'encoder_inputs' or 'decoder_inputs'.")

        ref = decoder_inputs if encoder_inputs is None else encoder_inputs
        B, device = ref.shape[0], ref.device

        if prediction_length == 0:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        patch_size = int(getattr(self.preprocessor, "patch_size", 1))
        num_patch_steps = (prediction_length + patch_size - 1) // patch_size

        # --- encoder path ---
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
                output_attentions=False,
                output_hidden_states=False,
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
            raise ValueError("Could not determine architecture for patch one-shot.")

        generated_patches: List[torch.Tensor] = []
        past_key_values = None

        for _ in range(num_patch_steps):
            step_in = decoder_patch_seq[:, -1:, :] if (past_key_values is not None) else decoder_patch_seq
            past_len = 0
            if past_key_values is not None:
                # could infer if needed; to keep simple here we pass 0
                past_len = 0

            dec_proc = self.preprocessor._prepare_decoder_inputs_for_generation(
                patch_embeds=step_in,
                attention_mask=(None if (past_key_values is not None) else decoder_attention_mask),
                past_key_values_length=past_len,
                is_causal=True,
            )
            dec_out = self.decoder(
                hidden_states=dec_proc["hidden_states"],
                attention_mask=dec_proc["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            next_patch = dec_out.last_hidden_state[:, -1:, :]
            generated_patches.append(next_patch)
            decoder_patch_seq = torch.cat([decoder_patch_seq, next_patch], dim=1)
            past_key_values = dec_out.past_key_values

        if not generated_patches:
            empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=ref.dtype)
            return ForecastBundle(point=empty) if return_bundle else empty

        all_patches = torch.cat(generated_patches, dim=1)
        recon = self.output_patch_reconstructor(all_patches)

        if recon.ndim != 3:
            raise RuntimeError(f"output_patch_reconstructor must return 3D Tensor, got {tuple(recon.shape)}")

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
                    f"Cannot infer d_model: recon.shape[-1]={proj} not divisible by output_patch_size={output_patch_size}."
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
        if levels is not None and post_quantiles and hasattr(primary_head, "sample_quantiles"):
            try:
                q_raw = primary_head.sample_quantiles(params_stacked if not isinstance(params_stacked, dict)
                                                      else self._ensure_components_present(params_stacked, primary_head),
                                                      levels)
                if torch.is_tensor(q_raw):
                    q_tensor = self._normalize_quantile_shape(
                        q_raw,
                        feature_size=getattr(self.config, "feature_size", 1),
                        Q=len(levels),
                    )
            except Exception as e:
                logger.warning(f"Patch one-shot quantiles failed: {e}")

        if sampling and hasattr(primary_head, "sample"):
            try:
                point = primary_head.sample(params_stacked, **(sampling_kwargs or {}))
            except Exception:
                point = self._compute_point_from_params__patch(params_stacked, primary_head, q_tensor)
        else:
            point = self._compute_point_from_params__patch(params_stacked, primary_head, q_tensor)

        if hasattr(self.preprocessor, "denormalize") and not return_raw:
            if point.size(-1) == getattr(self.config, "feature_size", point.size(-1)):
                try:
                    point = self.preprocessor.denormalize(point)
                except Exception:
                    pass

        extras = HeadExtras(**{}) if "HeadExtras" in globals() else {}
        bundle = ForecastBundle(
            point=point,
            quantiles=q_tensor,
            params=params_stacked,
            extras=extras,
        )
        if return_bundle:
            return bundle
        if isinstance(q_tensor, torch.Tensor):
            return q_tensor
        return point

    def _compute_point_from_params__patch(
        self,
        source: Union[torch.Tensor, Dict[str, Any], None],
        head: nn.Module,
        quantiles_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        raise TypeError("Cannot compute point from params for patch model")

    # ==================================================================
    #  3) PATCH BLOCKWISE / ROLLOUT
    # ==================================================================
    @torch.no_grad()
    def _generate_patch_blockwise(
        self,
        *,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        prediction_length: int,
        block_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **shared_kwargs,
    ):
        """
        Do patch generation in *multiple* shots, feeding back the native-time
        predictions. This is the patch analogue of the non-patch blockwise.

        Strategy:
            - If prediction_length <= block_len: just do patch one-shot.
            - Else:
                running_seq = initial inputs
                while remaining > 0:
                    this_block = min(remaining, block_len)
                    block_bundle = _generate_patch_one_shot(..., prediction_length=this_block, return_bundle=True)
                    append point to running_seq
        """
        if block_len is None:
            block_len = getattr(self.config, "prediction_length", None)
            if block_len is None:
                raise ValueError("block_len must be provided or set in config.prediction_length")

        # fast path
        if prediction_length <= block_len:
            return self._generate_patch_one_shot(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                prediction_length=prediction_length,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **shared_kwargs,
            )

        # multi-block
        ref = decoder_inputs if decoder_inputs is not None else encoder_inputs
        if ref is None:
            raise ValueError("You must provide at least encoder_inputs or decoder_inputs for patch_blockwise.")
        device = ref.device
        dtype = ref.dtype

        # we'll keep native-time seq here
        if decoder_inputs is not None:
            running_seq = decoder_inputs
            enc_inputs_static = None
        else:
            # enc-dec: first time we use encoder_inputs, then we just decode on the growing decoder seq
            running_seq = encoder_inputs[:, -1:, :].clone()
            enc_inputs_static = encoder_inputs

        remaining = prediction_length
        collected_points: List[torch.Tensor] = []
        collected_quants: List[torch.Tensor] = []
        params_tensor_acc: Optional[torch.Tensor] = None
        params_dict_acc: Optional[Dict[str, List[torch.Tensor]]] = None
        params_dict_meta: Optional[Dict[str, Any]] = None

        def _acc_dict(acc, meta, cur: Dict[str, Any]):
            if acc is None:
                acc = {}
            if meta is None:
                meta = {}
            for k, v in cur.items():
                if torch.is_tensor(v):
                    vv = v
                    if vv.ndim == 2:
                        vv = vv.unsqueeze(1)
                    acc.setdefault(k, []).append(vv)
                else:
                    meta.setdefault(k, v)
            return acc, meta

        while remaining > 0:
            this_block = min(remaining, block_len)
            # IMPORTANT: for subsequent steps we decode-only on the *running native seq*
            bundle = self._generate_patch_one_shot(
                encoder_inputs=enc_inputs_static if collected_points == [] else None,
                decoder_inputs=running_seq,
                prediction_length=this_block,
                attention_mask=None if collected_points else attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                return_bundle=True,
                **shared_kwargs,
            )
            point_b = bundle.point           # [B, this_block, F]
            quant_b = bundle.quantiles       # maybe None
            params_b = bundle.params

            collected_points.append(point_b)
            if isinstance(quant_b, torch.Tensor):
                collected_quants.append(quant_b)

            if isinstance(params_b, dict):
                params_dict_acc, params_dict_meta = _acc_dict(params_dict_acc, params_dict_meta, params_b)
            elif torch.is_tensor(params_b):
                if params_tensor_acc is None:
                    params_tensor_acc = params_b
                else:
                    if params_b.ndim == 2:
                        params_b = params_b.unsqueeze(1)
                    params_tensor_acc = torch.cat([params_tensor_acc, params_b], dim=1)

            # feedback
            running_seq = torch.cat([running_seq, point_b], dim=1)
            remaining -= this_block

        point_all = torch.cat(collected_points, dim=1)
        quant_all = None
        if collected_quants:
            quant_all = torch.cat(collected_quants, dim=1)

        params_final: Union[torch.Tensor, Dict[str, torch.Tensor], None] = None
        if params_tensor_acc is not None:
            params_final = params_tensor_acc
        elif params_dict_acc is not None:
            final_dict = {k: torch.cat(vs, dim=1) for k, vs in params_dict_acc.items()}
            if params_dict_meta:
                final_dict.update(params_dict_meta)
            params_final = final_dict

        # denorm (if requested in shared_kwargs)
        if shared_kwargs.get("denormalize", False) and hasattr(self.preprocessor, "denormalize"):
            point_all = self.preprocessor.denormalize(point_all)

        extras = HeadExtras(**{}) if "HeadExtras" in globals() else {}
        bundle_final = ForecastBundle(
            point=point_all,
            quantiles=quant_all,
            params=params_final,
            extras=extras,
        )
        if shared_kwargs.get("return_bundle", False):
            return bundle_final
        if quant_all is not None:
            return quant_all
        return point_all

    # ==================================================================
    #  PUBLIC ENTRYPOINTS
    # ==================================================================
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Unified entrypoint.
        kwargs:
            mode: "step" | "block" | "patch" | "patch_blockwise"
        """
        mode = kwargs.pop("mode", None)
        prediction_length = kwargs.get("prediction_length", None)
        block_len = kwargs.get("block_len", None)

        # auto-detect
        if mode is None:
            if self._is_patch_model():
                # if we're asking longer than train horizon, auto-upgrade to patch_blockwise
                train_h = getattr(self.config, "prediction_length", None)
                if train_h is not None and prediction_length is not None and prediction_length > train_h:
                    mode = "patch_blockwise"
                else:
                    mode = "patch"
            else:
                # non-patch: if user set a block_len or wants longer than trained horizon → block
                train_h = getattr(self.config, "prediction_length", None)
                if block_len is not None:
                    mode = "block"
                elif train_h is not None and prediction_length is not None and prediction_length > train_h:
                    mode = "block"
                else:
                    mode = "step"

        if mode in ("patch", "patch_one_shot"):
            return self._generate_patch_one_shot(**kwargs)
        if mode in ("patch_blockwise", "patch_ar", "patch_rollout"):
            return self._generate_patch_blockwise(**kwargs)
        if mode in ("block", "blockwise"):
            if block_len is None:
                block_len = getattr(self.config, "prediction_length", None)
            return self._generate_blockwise(block_len=block_len, **kwargs)

        # STEPWISE fallback — you can plug your real stepwise mixin here
        raise NotImplementedError("Stepwise mode not wired here; plug your AutoregressiveStepwiseMixin if needed.")

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        *,
        mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Forecast wrapper — just forwards to generate with encoder/decoder selection.
        """
        if hasattr(self, "encoder") and self.encoder is not None:
            return self.generate(
                encoder_inputs=inputs,
                prediction_length=prediction_length,
                mode=mode,
                **kwargs,
            )
        else:
            return self.generate(
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                mode=mode,
                **kwargs,
            )
