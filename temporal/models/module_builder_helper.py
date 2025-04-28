import inspect
from typing import Dict, Any
import torch.nn as nn
from temporal.registry.core import resolve

def _prepare_args(cls: type[nn.Module],
                  base: Dict[str, Any],
                  extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge `base` + `extra`, keep only ctor-accepted keys, add hidden-size aliases.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    args = {**base, **extra}

    # handle common hidden-size aliases
    hidden = args.get("hidden_size")
    if hidden is not None:
        for alias in ("d_model", "dim", "embedding_dim", "normalized_shape"): # Added normalized_shape here
            if alias in params and alias not in args:
                args[alias] = hidden

    # drop unsupported keys
    allowed = {p for p in params if p not in ("self", "args", "kwargs")}
    return {k: v for k, v in args.items() if k in allowed}


class ModuleBuilder:
    def __init__(self, config):   # unchanged
        self.config = config

    # ------------------------------------------------------------------
    # Generic resolver
    # ------------------------------------------------------------------
    def _build(self,
               kind: str,
               name: str,
               base_kwargs: Dict[str, Any] | None = None,
               user_kwargs: Dict[str, Any] | None = None):
        base_kwargs  = base_kwargs  or {}
        user_kwargs  = user_kwargs  or {}
        cls = resolve(kind, name)
        kwargs = _prepare_args(cls, base_kwargs, user_kwargs)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Specific helpers become one-liners
    # ------------------------------------------------------------------
    def build_attention(self, cfg):
        return self._build(
            "attention", cfg.attention_type,
            base_kwargs=dict(hidden_size=self.config.hidden_size,
                             num_heads=cfg.num_heads,
                             dropout=cfg.dropout),
            user_kwargs=cfg.kwargs,
        )

    def build_feedforward(self, cfg=None):
        cfg = cfg or self.config.feedforward_config
        return self._build(
            "feedforward", cfg.type,
            base_kwargs=dict(hidden_size=self.config.hidden_size,
                             intermediate_size=cfg.intermediate_size,
                             activation=cfg.activation,
                             dropout=cfg.dropout),
            user_kwargs=cfg.kwargs,
        )

    def build_value_embedding(self):
        cfg = self.config.value_embedding_config
        return self._build(
            "embedding", cfg.type,
            base_kwargs=dict(hidden_size=self.config.hidden_size,
                             feature_size=self.config.feature_size),
            user_kwargs=cfg.kwargs,
        )

    def build_positional_embedding(self):
        cfg = self.config.positional_embedding_config
        return self._build("embedding", cfg.type,
                           base_kwargs=dict(hidden_size=self.config.hidden_size),
                           user_kwargs=cfg.kwargs)

    def build_normalization(self):
        cfg = self.config.norm_config
        # Pass hidden_size directly to base_kwargs, which will be handled by _prepare_args
        # for classes like LayerNorm expecting 'normalized_shape'
        return self._build(
            "normalization", cfg.norm_type,
            base_kwargs=dict(hidden_size=self.config.hidden_size, eps=cfg.eps),
            user_kwargs=cfg.kwargs # Added user_kwargs here to pass elementwise_affine etc.
        )

    def build_head_aggregator(self):
        cfg = self.config.head_agg_config
        out_sz = self.config.output_head_config.output_size
        if out_sz is None:
            raise ValueError("output_head_config.output_size must be set.")

        return self._build(
            "head_agg", cfg.type,
            base_kwargs=dict(input_size=out_sz,
                             output_size=out_sz,
                             num_heads=self.config.output_token_lengths),
            user_kwargs=cfg.kwargs,
        )
