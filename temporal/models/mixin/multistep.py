import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MultiStepMixin:
    """
    Single-pass multi-step forecasting with fixed decoder context and chunked roll-forward.

    - Decoder input length is always `context_length` (default: config.context_length or inputs.shape[1]).
    - If prediction_length <= chunk_length: one decoder call (true single pass).
    - If prediction_length  > chunk_length: repeat:
        * feed last `context_length` inputs (auto-padded if shorter),
        * decode once,
        * take head on the LAST `steps` hidden states,
        * append median predictions to inputs (model input domain),
        * slide window by `steps`.
    - No per-step autoregression; exactly one decoder call per chunk.

    Requirements on the host model:
      - self.config.feature_size (and optionally .context_length, .prediction_length, .num_attention_heads)
      - self.preprocessor.process(...), optional self.preprocessor.denormalize(...)
      - self.decoder (and optional self.encoder)
      - self.output_heads (nn.Module or nn.ModuleList)
      - optional self.head_aggregator (for multi-head aggregation)
    """

    # --------------------------- small helpers --------------------------------

    def _get_head_output(
        self, hidden: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        """Apply output head(s) to time-major hidden states [B, T, D]."""
        if not hasattr(self, "output_heads"):
            raise AttributeError("Model is missing output_heads.")
        if isinstance(self.output_heads, nn.ModuleList):
            return [head(hidden) for head in self.output_heads]
        return self.output_heads(hidden)

    def _get_primary_head(self) -> nn.Module:
        return self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads

    def _normalize_levels(self, levels: Optional[List[float]]) -> Optional[List[float]]:
        if levels is None:
            return None
        qs = [float(q) for q in levels]
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"All quantiles must be in (0,1). Got {qs}")
        return sorted(qs)

    @staticmethod
    def _stack_distpred_chunks(chunks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Concatenate a list of full-chunk DistPred dicts along time.
        Each chunk dict must have:
          - 'paths': [B, T_chunk, F, K]
          - optional 'path_logits': [B, T_chunk, K]
        Returns:
          {'paths': [B, sum(T_chunk), F, K], 'path_logits': [B, sum(T_chunk), K]?}
        """
        if not chunks:
            raise ValueError("Empty chunks list for DistPred stacking.")
        paths = torch.cat([c["paths"] for c in chunks], dim=1)
        out = {"paths": paths}
        if "path_logits" in chunks[0] and chunks[0]["path_logits"] is not None:
            out["path_logits"] = torch.cat([c["path_logits"] for c in chunks], dim=1)
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
            if hasattr(self, "head_aggregator") and self.head_aggregator:
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

    def _build_padded_context(
        self,
        x: torch.Tensor,                       # [B, S_in, F] (model input domain)
        context_length: int,
        *,
        pad_context_mode: str = "left_zeros",  # {"left_zeros","left_repeat"}
        pad_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure fixed-length decoder context and mask.
          - If S_in >= context_length: take the last context_length; mask=1s
          - If S_in  < context_length: left-pad to length.
        Returns:
          ctx      : [B, context_length, F]
          ctx_mask : [B, context_length] (1 for real tokens, 0 for pad)
        """
        B, S_in, F = x.shape
        device, dtype = x.device, x.dtype

        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")

        if S_in >= context_length:
            ctx = x[:, -context_length:, :].contiguous()
            mask = torch.ones(B, context_length, device=device, dtype=torch.float32)
            return ctx, mask

        pad_len = context_length - S_in
        if pad_context_mode == "left_zeros":
            pad = torch.full((B, pad_len, F), pad_value, device=device, dtype=dtype)
        elif pad_context_mode == "left_repeat":
            if S_in > 0:
                first = x[:, :1, :]
                pad = first.expand(B, pad_len, F).contiguous()
            else:
                pad = torch.full((B, pad_len, F), pad_value, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown pad_context_mode: {pad_context_mode!r}")

        ctx = torch.cat([pad, x], dim=1)  # [B, context_length, F]
        mask = torch.cat(
            [
                torch.zeros(B, pad_len, device=device, dtype=torch.float32),
                torch.ones(B, S_in, device=device, dtype=torch.float32),
            ],
            dim=1,
        )
        return ctx, mask

    # --------------------------- main API ------------------------------------

    @torch.no_grad()
    def forecast_single_pass_chunked(
        self,
        inputs: torch.Tensor,                    # [B, S_in, F] (history/context in *model input domain*)
        prediction_length: int,
        *,
        quantiles: Optional[List[float]] = None,
        chunk_length: int = 256,                 # stride size (e.g., 256)
        context_length: Optional[int] = None,    # decoder input length (e.g., 1024)
        collect: str = "auto",                   # {'auto','none','distpred','per_chunk'}
        prediction_strategy: Optional[Union[str, float, int]] = None,
        validate_shapes: bool = True,
        verbose: bool = True,
        return_raw: bool = False,
        attention_mask: Optional[torch.Tensor] = None,  # encoder mask if you use an encoder
        pad_context_mode: str = "left_zeros",           # {"left_zeros","left_repeat"}
        pad_value: float = 0.0,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
        List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ]:
        """
        Single-pass multi-step with fixed-length decoder context and chunked roll-forward.

        Returns:
          - collect='none'     -> aggregated tensor when possible (predict/quantiles/raw), [B, T, ...]
          - collect='distpred' -> {'paths':[B,T,F,K], 'path_logits':[B,T,K]?}
          - collect='per_chunk'-> list of per-chunk outputs (each [B, steps, ...] or a DistPred dict)
          - collect='auto'     -> stacked DistPred dict if available, else tensor
        """
        self.eval()

        B, device, dtype = inputs.shape[0], inputs.device, inputs.dtype
        F = getattr(self.config, "feature_size", None)
        if F is None:
            raise ValueError("config.feature_size is required.")

        # Determine decoder context length
        if context_length is None:
            context_length = getattr(self.config, "context_length", None) or inputs.shape[1]
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")

        primary_head = self._get_primary_head()

        total_T = int(prediction_length)
        if total_T <= 0:
            return torch.empty((B, 0, F), device=device, dtype=dtype)

        # Chunk plan
        n_full = total_T // chunk_length
        rem = total_T % chunk_length
        chunk_sizes: List[int] = ([chunk_length] * n_full) + ([rem] if rem > 0 else [])

        # (1) Build initial decoder context (auto-pad if needed)
        ctx, ctx_mask = self._build_padded_context(
            inputs, context_length,
            pad_context_mode=pad_context_mode,
            pad_value=pad_value,
        )  # ctx: [B, Ctx, F], ctx_mask: [B, Ctx]

        # Optional encoder pass (if the model has one)
        encoder_hidden_states = None
        if hasattr(self, "encoder") and self.encoder is not None:
            enc = self.preprocessor.process(
                input_values=ctx,            # padded context fed to encoder as well
                attention_mask=ctx_mask,
                is_causal=False,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            enc_out = self.encoder(
                hidden_states=enc["hidden_states"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state  # [B, Ctx, D_enc]

        per_chunk_outs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []

        for steps in chunk_sizes:
            # (2) Decoder sees EXACTLY `context_length` each chunk
            dec_in = ctx                              # [B, Ctx, F]
            dec_mask = ctx_mask                       # [B, Ctx]

            dec_proc = self.preprocessor.process(
                input_values=dec_in,
                past_key_values_length=0,
                attention_mask=dec_mask,
                is_causal=True,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            dec_out = self.decoder(
                hidden_states=dec_proc["hidden_states"],   # [B, Ctx, D]
                attention_mask=dec_proc["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                return_dict=True,
            )
            h_all = dec_out.last_hidden_state              # [B, Ctx, D]

            # (3) Take LAST `steps` hidden positions and run the head (time-major for this chunk)
            tail_hidden = h_all[:, -steps:, :]             # [B, steps, D]
            head_out = self._get_head_output(tail_hidden)  # tensor | dict | list

            # (4) What to return/store for this chunk
            store_item = self._compute_prediction_to_store(
                head_out, prediction_strategy, quantiles, primary_head
            )
            per_chunk_outs.append(store_item)

            # (5) Roll: append median predictions (model input domain) to form next context
            if steps > 0:
                if isinstance(head_out, dict) and "paths" in head_out:
                    # DistPred: median across K
                    rolled = torch.quantile(head_out["paths"], q=0.5, dim=-1)  # [B, steps, F]
                elif torch.is_tensor(store_item):
                    # Tensor: [B, steps, F] or [B, steps, F, Q]
                    if store_item.ndim == 4:
                        rolled = torch.median(store_item, dim=-1).values       # [B, steps, F]
                    else:
                        rolled = store_item                                     # [B, steps, F]
                else:
                    # Multi-head or other type: fallback to primary_head.predict median
                    if hasattr(primary_head, "predict") and callable(getattr(primary_head, "predict")):
                        rolled = primary_head.predict(head_out, method="median")  # [B, steps, F]
                    else:
                        raise TypeError("Cannot roll forward: unsupported head output type for context building.")

                # Update context: keep exactly `context_length` and set mask to ones from now on
                ctx = torch.cat([ctx, rolled], dim=1)[:, -context_length:, :].contiguous()
                ctx_mask = torch.ones(B, context_length, device=device, dtype=torch.float32)

        # (6) Package across chunks
        first = per_chunk_outs[0]
        if collect == "per_chunk":
            final = per_chunk_outs
        elif (collect in ("auto", "distpred")) and isinstance(first, dict) and "paths" in first:
            final = self._stack_distpred_chunks(per_chunk_outs)  # {'paths':[B,T,F,K], 'path_logits':[B,T,K]?}
        elif torch.is_tensor(first):
            # concatenate along time
            try:
                final = torch.cat(per_chunk_outs, dim=1)         # [B, T, ...]
            except Exception as e:
                logger.warning(f"Concat failed for tensor chunks; returning per-chunk list. Error: {e}")
                final = per_chunk_outs
        else:
            final = per_chunk_outs

        # (7) Optional denorm if returning a plain tensor with last-dim == F
        if isinstance(final, torch.Tensor):
            last_dim = final.size(-1) if final.dim() >= 2 else None
            can_denorm = (last_dim == getattr(self.config, "feature_size", last_dim))
            if hasattr(self.preprocessor, "denormalize") and can_denorm and not return_raw:
                logger.info("Denormalizing single-pass chunked predictions.")
                final = self.preprocessor.denormalize(final)

        return final
