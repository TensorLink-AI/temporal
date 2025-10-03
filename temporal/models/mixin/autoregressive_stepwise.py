import logging
from typing import Optional, Union, Any, Dict, List, Tuple

import torch
import torch.nn as nn

from temporal.modules.heads.forecast_types import ForecastBundle, HeadExtras

logger = logging.getLogger(__name__)


class AutoregressiveStepwiseMixin:
    """
    Stepwise autoregressive generation that returns a consistent :class:`ForecastBundle`.

    This mixin runs the decoder **one native time step at a time**, threading
    past-key-values (KV cache) for efficiency. It supports differentiable generation
    (for RL-style training), post-hoc quantiles, per-step sampling for AR feedback,
    and multiple output-head configurations.

    Assumptions about the host model
    --------------------------------
    - ``self.config`` exposes:
        * ``feature_size`` (int) — number of target channels F
        * ``hidden_size`` (int)  — decoder hidden size
        * ``prediction_length`` (Optional[int])
        * ``num_attention_heads`` (Optional[int])
    - ``self.preprocessor`` implements:
        * ``process(input_values, attention_mask, is_causal, validate_shapes, verbose)``
        * ``_prepare_decoder_inputs_for_generation(patch_embeds, past_key_values_length)``
        * ``value_embedding(x)`` — embed a [B,1,F] step for the decoder
        * ``denormalize(t)`` (optional) — applied to point/quantiles if ``return_raw=False``
    - ``self.decoder`` (and optional ``self.encoder``) are HuggingFace-style:
        * return an object with ``.last_hidden_state`` and (optionally) ``.past_key_values``
    - ``self.output_heads``:
        * a ``nn.Module`` or ``nn.ModuleList`` whose ``forward(last_hidden)`` returns
          either a *tensor of params* or a *dict of params* (e.g., MDN/DistPred)
        * heads should implement:
            - ``predict(params, method=...)``  → [B,T,F] or [B,T]
            - ``sample(params, **kwargs)``     → [B,1,F] per step (for AR feedback)
            - ``sample_quantiles(params, quantile_levels)`` → [B,T,F,Q] (post-hoc)
    - Optional:
        * ``self.head_aggregator(list_of_per_head_outputs) -> Any`` — to combine multi-head outputs

    Key guarantees on return values
    -------------------------------
    * ``bundle.point`` is always a Tensor of shape ``[B, T, F]``.
    * ``bundle.quantiles`` is either ``None`` or a Tensor ``[B, T, F, Q]``.
    * ``bundle.params`` is the stacked *primary head* outputs:
        - Tensor for Gaussian/StudentT/QuantileRegression-style heads
        - Dict[str, Tensor] for MDN/DistPred
    * Dicts are **never** placed into ``bundle.quantiles`` to avoid shape errors.
    """

    # ---------------------------------------------------------------------
    # Basic utilities
    # ---------------------------------------------------------------------
    def enable_dropout(self) -> None:
        """
        Enable dropout layers regardless of global train/eval mode.

        Useful for Monte Carlo dropout at inference time. This method walks the
        module tree and puts every ``nn.Dropout`` in ``train()`` mode so it samples.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_cache_length(self, past_key_values) -> int:
        """
        Infer the current KV cache length ``L`` from a HuggingFace-style ``past_key_values``.

        The first layer's *key* tensor is inspected; common layouts include:
        ``(B,H,L,D)``, ``(B,L,H,D)``, and ``(B,L,D)``. If heads/d_model are available,
        dimensions are disambiguated using ``self.config.num_attention_heads`` or
        ``self.config.hidden_size``.

        Parameters
        ----------
        past_key_values : Any
            The structure returned by a decoder with ``use_cache=True``.

        Returns
        -------
        int
            The cached sequence length ``L``. Returns 0 if it cannot be inferred.
        """
        if past_key_values is None:
            return 0

        first_layer = past_key_values[0]
        # Extract key tensor from common layouts
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
            # (B,H,L,D) or (B,L,H,D)
            _, a1, a2, _ = key_tensor.shape
            if nh is not None:
                if a1 == nh and a2 != nh:
                    return int(a2)
                if a2 == nh and a1 != nh:
                    return int(a1)
            return int(max(a1, a2))

        if ndim == 3:
            # (B,L,D) or (B,D,L)
            _, a1, a2 = key_tensor.shape
            d_model = getattr(getattr(self, "config", object), "hidden_size", None)
            if d_model is not None:
                if a2 == d_model:
                    return int(a1)
                if a1 == d_model:
                    return int(a2)
            return int(max(a1, a2))

        return 0

    @staticmethod
    def _get_scalar_value(value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """
        Convert a value to a Python float if possible. Tensors are reduced to scalars.

        Parameters
        ----------
        value : Tensor | float | int | Any
            The value to convert.
        name : str
            Name used in error messages.

        Returns
        -------
        Optional[float]
            Converted float or ``None`` if the input was ``None``.

        Raises
        ------
        ValueError
            If a tensor cannot be reduced to a scalar.
        TypeError
            If conversion to float fails.
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

        Parameters
        ----------
        last_hidden : Tensor
            The decoder's last hidden state slice for the current step, typically
            ``[B, 1, hidden_size]``.

        Returns
        -------
        Tensor | Dict[str, Tensor] | List[Tensor|Dict]
            The raw output from the head(s). For a single head this is commonly a
            parameter tensor; for MDN/DistPred it is a dict; for multi-head setups
            it is a list aligned with ``self.output_heads``.
        """
        if not hasattr(self, "output_heads"):
            raise AttributeError("Model is missing output_heads, required for autoregressive generation.")
        if isinstance(self.output_heads, nn.ModuleList):
            return [head(last_hidden) for head in self.output_heads]
        return self.output_heads(last_hidden)

    @staticmethod
    def _normalize_levels(quantile_levels: Optional[List[float]]) -> Optional[List[float]]:
        """
        Validate and sort requested quantile levels.

        Parameters
        ----------
        quantile_levels : list[float] | None
            Requested quantiles in (0,1).

        Returns
        -------
        list[float] | None
            Sorted levels or ``None`` if no quantiles were requested.

        Raises
        ------
        ValueError
            If any quantile lies outside (0,1).
        """
        if quantile_levels is None:
            return None
        qs = [float(q) for q in quantile_levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    @staticmethod
    def _ensure_b1f(x: torch.Tensor, feature_size: int) -> torch.Tensor:
        """
        Ensure a feedback tensor has shape ``[B, 1, F]`` from common variants.

        Accepted inputs
        ---------------
        * ``[B,1,F,K]`` → mean over last dim → ``[B,1,F]``
        * ``[B,1,F]``   → as-is
        * ``[B,F]``     → unsqueeze to ``[B,1,F]``
        * ``[B,1,1]``   → broadcast to ``[B,1,F]`` if ``F>1``
        * ``[B]``       → reshape to ``[B,1,1]`` then broadcast if ``F>1``

        Parameters
        ----------
        x : Tensor
            Candidate feedback tensor.
        feature_size : int
            The target feature/channel dimension ``F``.

        Returns
        -------
        Tensor
            A tensor with shape ``[B, 1, F]``.
        """
        if x.ndim == 4:
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
        """
        Return the first (primary) output head.

        Returns
        -------
        nn.Module
            The primary head, which is used for AR feedback and bundling.
        """
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
        Call ``output_head.sample(...)`` with optional state threading.

        If the head's ``sample`` signature includes a ``state`` argument, a
        persistent per-generation dictionary (``self._gen_state``) is passed and
        updated across steps.

        Parameters
        ----------
        output_head : nn.Module
            The primary output head.
        head_output : Tensor | Dict | List
            Raw head output for the current step.
        sampling_kwargs : dict | None
            Extra keyword arguments forwarded to ``sample(...)`` (e.g., temperature).

        Returns
        -------
        (Tensor, dict|None)
            * ``y`` — next feedback values normalized to ``[B,1,F]``
            * ``new_state`` — any returned state dict (also merged into ``self._gen_state``)
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
        Decide what to **store** per step in the trajectory accumulator.

        Preference order (per head)
        ---------------------------
        1) If ``prediction_strategy`` is provided and the head implements
           ``predict(...)``, store that deterministic prediction.
        2) Else, if ``quantile_levels`` requested and head implements
           ``sample_quantiles(...)`` and we're **not** deferring quantiles,
           store the quantile tensor for this step.
        3) Otherwise, store the raw head output (tensor/dict).

        Multi-head behavior
        -------------------
        For ``nn.ModuleList`` heads, this is applied per head and the results
        are returned as a list. If a ``head_aggregator`` is defined, it is
        invoked on that list to return an aggregated value.

        Parameters
        ----------
        raw_head_output : Tensor | List[Any] | Dict[str, Any]
            Output of ``self.output_heads(last_hidden_step)``.
        prediction_strategy : str | float | int | None
            Strategy forwarded to ``head.predict`` when available:
            - "mean" | "median"
            - float in (0,1) for quantile
            - int for component index (head-specific)
        quantile_levels : list[float] | None
            Requested quantiles in (0,1).
        output_head : nn.Module
            Primary head (or per-head inside a list).
        defer_quantiles : bool
            When ``True``, skip per-step quantiles (favor post-hoc once on stacked params).

        Returns
        -------
        Tensor | Dict[str, Tensor] | List[Any]
            The item stored for this step.
        """
        levels = self._normalize_levels(quantile_levels)

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

        # Single head
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
        Choose the **feedback** value fed to the next AR step (normalized space).

        Preference order
        ----------------
        1) ``head.sample(...)`` if ``use_sampling`` is True.
        2) ``head.predict(..., method=prediction_strategy or "median")``.
        3) ``head.sample_quantiles(..., [0.5])`` as a fallback if implemented.
        4) Raw tensor (reduced/broadcast) if none of the above are available.

        Multi-head notes
        ----------------
        If multiple heads are present and no ``head_aggregator`` is defined, the
        first head's output is used for feedback and a warning is logged.

        Parameters
        ----------
        raw_head_output : Tensor | List[Any] | Dict[str, Any]
            Output from the head(s) for the current step.
        prediction_strategy : str | float | int | None
            Strategy for deterministic predictions (see also
            :meth:`_compute_prediction_to_store`).
        output_head : nn.Module
            Primary head.
        use_sampling : bool
            Whether to attempt ``head.sample(...)`` for feedback.
        sampling_kwargs : dict | None
            Extra kwargs forwarded to ``sample(...)``.

        Returns
        -------
        Tensor
            Feedback tensor shaped as ``[B, 1, F]``.

        Raises
        ------
        TypeError
            If the head returns a dict but supports neither ``sample`` nor ``predict``.
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
            if feedback_source.ndim == 4:  # e.g., [B,1,F,Q]
                feedback_source = feedback_source.mean(dim=-1)
            return self._ensure_b1f(feedback_source, self.config.feature_size)

        raise TypeError("Output head returning a dict must implement .sample() or .predict() to provide a tensor for AR feedback.")

    # ---------------------------------------------------------------------
    # Param stacking and post-quantiles
    # ---------------------------------------------------------------------
    @staticmethod
    def _accum_params_dict_step(
        acc_tensors: Optional[Dict[str, List[torch.Tensor]]],
        acc_meta: Optional[Dict[str, Any]],
        step_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]:
        """
        Accumulate dict-based head outputs over time (e.g., MDN/DistPred).

        Tensors are appended with ``T==1`` enforced; non-tensors (like
        a ``components`` list) are stored once in ``acc_meta``.

        Parameters
        ----------
        acc_tensors : dict[str, list[Tensor]] | None
            Accumulated per-key tensors.
        acc_meta : dict[str, Any] | None
            Stored non-tensor meta entries.
        step_dict : dict[str, Any]
            Dict produced by the head at a single step.

        Returns
        -------
        (dict[str, list[Tensor]], dict[str, Any])
            Updated accumulators for tensors and meta.
        """
        if acc_tensors is None:
            acc_tensors = {}
        if acc_meta is None:
            acc_meta = {}

        for k, v in step_dict.items():
            if torch.is_tensor(v):
                vv = v
                if vv.ndim >= 2 and vv.shape[1] != 1:
                    vv = vv[:, -1:, ...]  # keep the last step only
                acc_tensors.setdefault(k, []).append(vv)
            else:
                acc_meta.setdefault(k, v)
        return acc_tensors, acc_meta

    @staticmethod
    def _stack_params_dict(
        acc_tensors: Dict[str, List[torch.Tensor]],
        acc_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Concatenate dict-of-lists along time and reattach meta keys.

        Parameters
        ----------
        acc_tensors : dict[str, list[Tensor]]
            Per-key lists of step tensors (each with T=1).
        acc_meta : dict[str, Any] | None
            Non-tensor metadata to reattach.

        Returns
        -------
        dict[str, Any]
            Dict with tensors stacked into T: ``[B, T, ...]``.
        """
        out: Dict[str, Any] = {}
        for k, vs in acc_tensors.items():
            out[k] = torch.cat(vs, dim=1)  # [B, T, ...]
        if acc_meta:
            out.update(acc_meta)
        return out

    @staticmethod
    def _ensure_components_present(params: Dict[str, Any], head: nn.Module) -> Dict[str, Any]:
        """
        Ensure an MDN/DistPred-like dict has a 'components' key when the head exposes it.

        Parameters
        ----------
        params : dict[str, Any]
            Head parameter dictionary.
        head : nn.Module
            Output head which may define ``components``.

        Returns
        -------
        dict[str, Any]
            The (possibly) augmented parameter dictionary.
        """
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
        Normalize various quantile shapes to ``[B, T, F, Q]``.

        Accepted shapes
        ---------------
        * ``[B, T, Q]``    → ``[B, T, 1, Q]``
        * ``[B, T, F, Q]`` → as-is
        * ``[B, T, Q, F]`` → permute to ``[B, T, F, Q]``
        * ``[B, Q]``       → ``[B, 1, 1, Q]``

        Parameters
        ----------
        q : Tensor
            Quantile tensor.
        feature_size : int
            Target feature dimension ``F``.
        Q : int
            Number of quantile levels.

        Returns
        -------
        Tensor
            Quantiles normalized to ``[B, T, F, Q]``.

        Raises
        ------
        ValueError
            If the input shape cannot be normalized.
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
        Compute quantiles **once after the AR loop** for tensor/dict/list-of-heads.

        For the primary head (and per head in a ModuleList), calls
        ``sample_quantiles(...)`` when available and normalizes shapes.
        Returns the original object if quantiles are not supported.

        Parameters
        ----------
        params_or_preds : Tensor | Dict[str, Any] | List[Any]
            Stacked head outputs (e.g., params over T) or predictions.
        head : nn.Module
            The primary head (or per-head in a list).
        quantile_levels : list[float]
            Sorted quantile levels in (0,1).

        Returns
        -------
        Tensor | Dict[str, Any] | List[Any]
            Quantiles tensor ``[B,T,F,Q]`` for the primary head (if supported),
            otherwise the original input (or per-head list with substitutions).
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
        Produce the tuple ``(point, quantiles, params_for_bundle)`` in a head-agnostic way.

        *Point*: Prefer ``head.predict(..., method="median")``, fall back to ``"mean"``.
        *Quantiles*: If ``quantile_levels`` were requested and the head implements
        ``sample_quantiles(...)``, compute and normalize to ``[B,T,F,Q]``. For pure
        quantile-regression heads where ``params_stacked`` is already the quantile
        grid (last dim == ``Q``), normalize directly.

        Parameters
        ----------
        head : nn.Module
            Primary head.
        params_stacked : Tensor | Dict[str, Tensor]
            Stacked per-step head outputs over time (T dimension).
        quantile_levels : list[float] | None
            Quantile levels to compute post-hoc.

        Returns
        -------
        (Tensor, Tensor|None, Tensor|Dict[str, Tensor])
            * ``point``     — ``[B,T,F]``
            * ``quantiles`` — ``[B,T,F,Q]`` or ``None``
            * ``params``    — raw stacked head params (tensor or dict)

        Raises
        ------
        TypeError
            If the head cannot provide a point forecast via ``predict``.
        """
        F = getattr(self.config, "feature_size", 1)

        # point
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

        # quantiles
        q_tensor: Optional[torch.Tensor] = None
        if quantile_levels:
            if hasattr(head, "sample_quantiles") and callable(getattr(head, "sample_quantiles")):
                q_raw = head.sample_quantiles(src_for_point, quantile_levels)
                if not torch.is_tensor(q_raw):
                    raise TypeError("sample_quantiles must return a Tensor for bundling.")
                q_tensor = self._normalize_quantile_shape(q_raw, feature_size=F, Q=len(quantile_levels))
            else:
                if torch.is_tensor(params_stacked) and params_stacked.shape[-1] == len(quantile_levels):
                    q_tensor = self._normalize_quantile_shape(params_stacked, feature_size=F, Q=len(quantile_levels))

        return point, q_tensor, params_stacked

    def _compute_point_from_params(
        self,
        source: Union[torch.Tensor, Dict[str, Any], List[Any], None],
        head: nn.Module,
        *,
        quantiles_tensor: Optional[torch.Tensor] = None,
        method: Optional[Union[str, float, int]] = None,
    ) -> torch.Tensor:
        """
        Compute a point forecast from either a quantile tensor or ``head.predict``.

        Parameters
        ----------
        source : Tensor | Dict | List | None
            Stacked params or predictions for ``head.predict``.
        head : nn.Module
            The primary output head.
        quantiles_tensor : Tensor | None
            If provided, the median slice is used as the point.
        method : str | float | int | None
            Method forwarded to ``head.predict`` ("median", "mean", quantile, etc.).

        Returns
        -------
        Tensor
            Point tensor shaped ``[B,T,F]``.

        Raises
        ------
        TypeError
            If neither quantiles nor ``predict`` are available.
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
            meth = method if method is not None else "median"
            try:
                pv = head.predict(src, method=meth)
            except Exception:
                pv = head.predict(src, method="mean")
            if pv.ndim == 2:
                pv = pv.unsqueeze(-1)
            return pv

        raise TypeError("Cannot compute point forecast from given params; add 'predict()' to this head.")

    # ---------------------------------------------------------------------
    # Generation API
    # ---------------------------------------------------------------------
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
        return_samples: Optional[bool] = None,
        enable_mc_dropout: bool = False,
        return_raw: bool = False,
        post_quantiles: bool = True,
        return_bundle: bool = False,
        return_params: bool = False,
        differentiable: bool = False,
        **kwargs,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
        ForecastBundle,
        List[Dict[str, Union[torch.Tensor, List[str]]]],
    ]:
        """
        Stepwise autoregressive forecast. Gradients flow only if ``differentiable=True``.

        This method:
        1) Optionally encodes the context (enc-dec models).
        2) Preprocesses the decoder seed/prompt (normalized space).
        3) Warms the decoder KV cache on the full context.
        4) Iterates for ``prediction_length`` steps:
            - Runs the head(s) for the current step.
            - Chooses a feedback value via sampling or deterministic prediction.
            - Embeds the feedback and advances the decoder by one step.
        5) Stacks head params over time and (optionally) computes quantiles once.
        6) Assembles the :class:`ForecastBundle` or returns point/quantiles directly.

        Parameters
        ----------
        encoder_inputs : Tensor | None
            Input sequence for the encoder path (if present), shape ``[B, Tctx, F]``.
        decoder_inputs : Tensor | None
            Initial decoder prompt values ``[B, T0, F]``. If ``None`` and encoder inputs
            are provided, the last encoder step is used as the seed.
        prediction_length : int | None
            Number of autoregressive steps to generate. Defaults to
            ``self.config.prediction_length`` when ``None``.
        attention_mask : Tensor | None
            Mask aligned with ``encoder_inputs``.
        decoder_attention_mask : Tensor | None
            Mask aligned with ``decoder_inputs``.
        use_cache : bool
            Whether to thread ``past_key_values`` through the decoder for speed.
        eos_token_id : Any | None
            If provided, enables an equality-based early stopping check (performed on
            *denormalized* values for interpretability).
        prediction_strategy : str | float | int | None
            Deterministic prediction method for heads that implement ``predict``:
            - "mean" | "median"
            - float in (0,1): quantile level
            - int: head-specific index/component
        quantile_levels : list[float] | None
            If provided, compute quantiles once after stacking params over time.
        sampling : bool
            If ``True`` and the head implements ``sample(...)``, use sampling for AR feedback.
        sampling_kwargs : dict | None
            Extra kwargs forwarded to ``sample(...)`` (e.g., temperature, top_p).
        store_sampled : bool
            If ``True``, the returned ``point`` is the actually sampled AR path. When
            ``denormalize`` is available, sampled steps are denormalized as they are stored
            to avoid double-denormalization.
        return_samples : bool | None
            Backward-compatible alias for ``store_sampled``; if provided, merged via OR.
        enable_mc_dropout : bool
            If ``True``, enables dropout during eval for MC sampling.
        return_raw : bool
            If ``False`` and the preprocessor implements ``denormalize``, apply it to
            the point (and quantiles) before returning.
        post_quantiles : bool
            If ``True``, compute quantiles **once** on the stacked parameters after the loop.
        return_bundle : bool
            If ``True``, return a :class:`ForecastBundle`; else return a tensor (point or quantiles).
        return_params : bool
            If ``True``, return the stacked head parameters (Tensor or Dict) instead of a point.
        differentiable : bool
            If ``True``, allow gradients to flow through the decode loop (useful for RL).

        Returns
        -------
        ForecastBundle | Tensor | Dict | list
            - If ``return_bundle=True`` → a :class:`ForecastBundle`.
            - Else, if quantiles were computed → quantiles ``[B,T,F,Q]`` tensor.
            - Else → point ``[B,T,F]`` tensor.
            - If ``return_params=True`` → stacked head params (Tensor or Dict).

        Notes
        -----
        * The method temporarily switches the module to ``eval()`` for stable behavior
          during generation, but **restores** the original train/eval mode afterward.
        * With ``differentiable=True``, ``torch.set_grad_enabled(True)`` is used only
          around the parts of the loop that must retain gradients.
        """
        was_training = self.training
        if return_samples is not None:
            store_sampled = bool(return_samples) or bool(store_sampled)

        try:
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
                    raise ValueError("Decoder-only generation needs a real prompt in 'decoder_inputs'.")

            # Process initial (normalized) context
            initial_processed = self.preprocessor.process(
                input_values=decoder_inputs,
                attention_mask=decoder_attention_mask,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            initial_hidden_states = initial_processed["hidden_states"]
            initial_attention_mask = initial_processed["attention_mask"]

            # Warm KV cache with full context
            with torch.set_grad_enabled(differentiable):
                dec_out = self.decoder(
                    hidden_states=initial_hidden_states,
                    attention_mask=initial_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cache=use_cache,
                    return_dict=True,
                )
                past_key_values = dec_out.past_key_values if use_cache else None
                next_step_hidden = dec_out.last_hidden_state[:, -1:, :]

            # Accumulators
            sampled_steps: List[torch.Tensor] = []
            params_acc_list: List[Union[torch.Tensor, Dict[str, Any]]] = []

            # AR loop
            for i in range(prediction_length):
                # Head forward (retain grads only if differentiable)
                with torch.set_grad_enabled(differentiable):
                    head_out_params = self._get_head_output(next_step_hidden)
                params_acc_list.append(head_out_params)

                # Next feedback value in normalized space
                next_val_normalized = self._compute_next_decoder_input_value(
                    head_out_params,
                    prediction_strategy,
                    primary_head,
                    use_sampling=sampling,
                    sampling_kwargs=sampling_kwargs,
                )

                # Optionally store sampled (denormalized now to avoid double-denorm)
                if store_sampled and hasattr(self.preprocessor, "denormalize"):
                    with torch.set_grad_enabled(False):
                        denorm_val = self.preprocessor.denormalize(next_val_normalized)
                    sampled_steps.append(denorm_val)

                # Optional early stopping on denormalized scalar
                if early_stopping and eos_token_id is not None and hasattr(self.preprocessor, "denormalize"):
                    eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")
                    with torch.set_grad_enabled(False):
                        check_val = self.preprocessor.denormalize(next_val_normalized)
                    target = torch.full_like(check_val, eos_value_scalar)
                    if torch.allclose(check_val, target, rtol=0.0, atol=1e-6):
                        logger.info(f"Early stopping triggered at step {i + 1}.")
                        break

                # Prepare single new token for next step
                next_embed = self.preprocessor.value_embedding(next_val_normalized)
                past_len = self._get_cache_length(past_key_values)
                processed_step = self.preprocessor._prepare_decoder_inputs_for_generation(
                    patch_embeds=next_embed,
                    past_key_values_length=past_len,
                )

                # One-step decode (retain grads only if differentiable)
                with torch.set_grad_enabled(differentiable):
                    dec_out = self.decoder(
                        hidden_states=processed_step["hidden_states"],
                        attention_mask=processed_step["attention_mask"],
                        encoder_hidden_states=encoder_hidden_states,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        return_dict=True,
                    )
                    past_key_values = dec_out.past_key_values
                    next_step_hidden = dec_out.last_hidden_state

            # Assemble outputs
            if not params_acc_list:
                empty = torch.empty((B, 0, getattr(self.config, "feature_size", 1)), device=device, dtype=dtype)
                return ForecastBundle(point=empty) if return_bundle else empty

            if torch.is_tensor(params_acc_list[0]):
                params_stacked: Union[torch.Tensor, Dict[str, Any]] = torch.cat(params_acc_list, dim=1)  # [B,T,...]
            elif isinstance(params_acc_list[0], dict):
                acc_tensors: Dict[str, List[torch.Tensor]] = {}
                acc_meta: Optional[Dict[str, Any]] = None
                for step_dict in params_acc_list:
                    acc_tensors, acc_meta = self._accum_params_dict_step(acc_tensors, acc_meta, step_dict)  # type: ignore[arg-type]
                params_stacked = self._stack_params_dict(acc_tensors, acc_meta)
            else:
                raise TypeError("Unsupported head output type in params_acc_list.")

            if return_params:
                return params_stacked

            levels = self._normalize_levels(quantile_levels)
            point, q_tensor, params_for_bundle = self._extract_bundle_parts(self._get_primary_head(), params_stacked, levels)

            # If returning the sampled AR path, use it as the point (already denormalized)
            if store_sampled and sampled_steps:
                point = torch.cat(sampled_steps, dim=1)
                return_raw = True  # sampled path already denormalized

            # Optional denormalization (point + quantiles)
            if hasattr(self.preprocessor, "denormalize") and not return_raw:
                with torch.set_grad_enabled(False):
                    try:
                        point = self.preprocessor.denormalize(point)
                    except Exception as e:
                        logger.warning(f"Denormalization failed for point; returning raw. Error: {e}")
                    if isinstance(q_tensor, torch.Tensor):
                        try:
                            q_tensor = self.preprocessor.denormalize(q_tensor)
                        except Exception as e:
                            logger.warning(f"Denormalization failed for quantiles; returning raw. Error: {e}")

            bundle = ForecastBundle(
                point=point,
                quantiles=q_tensor,
                params=params_for_bundle,
                extras=HeadExtras(),
            )

            if return_bundle:
                return bundle
            if isinstance(q_tensor, torch.Tensor):
                return q_tensor
            return point

        finally:
            # Always restore the original train/eval mode
            self.train(was_training)

    @torch.no_grad()
    def forecast(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Convenience wrapper around :meth:`generate` that defaults to inference/no-grad.

        Parameters
        ----------
        inputs : Tensor
            Input sequence ``[B, T, F]`` (context for enc-dec or prompt for dec-only).
        prediction_length : int
            Number of steps to forecast.
        quantiles : list[float] | None
            Quantiles to compute post-hoc (e.g., ``[0.1, 0.5, 0.9]``).

        Returns
        -------
        ForecastBundle | Tensor
            See :meth:`generate` for return conventions.
        """
        if hasattr(self, "encoder") and self.encoder is not None:
            return self.generate(
                encoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                differentiable=False,
                **kwargs,
            )
        else:
            return self.generate(
                decoder_inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                differentiable=False,
                **kwargs,
            )
