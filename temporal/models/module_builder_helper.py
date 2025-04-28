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
    hidden = args.get("hidden_size") # Keep this for other modules
    embed_dim_val = args.get("embed_dim") # Check if embed_dim was passed directly

    # Prioritize explicit embed_dim if passed, otherwise use hidden_size if needed
    target_dim = embed_dim_val if embed_dim_val is not None else hidden

    if target_dim is not None:
        # Ensure embed_dim is set if the constructor accepts it
        if 'embed_dim' in params and 'embed_dim' not in args:
             args['embed_dim'] = target_dim
        # Map to other aliases if they exist and aren't already set
        for alias in ("d_model", "dim", "embedding_dim", "normalized_shape"):
            if alias in params and alias not in args:
                args[alias] = target_dim

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
        # Debugging print: See what args are passed to the constructor
        # print(f"Building {kind}/{name} ({cls.__name__}) with args: {kwargs}")
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Specific helpers become one-liners
    # ------------------------------------------------------------------
    def build_attention(self, cfg):
        # cfg is an AttentionConfig instance
        # Pass hidden_size AS embed_dim directly into base_kwargs.
        # _prepare_args will then ensure it's used if the constructor needs 'embed_dim'.
        return self._build(
            "attention", cfg.attention_type,
            base_kwargs=dict(
                embed_dim=self.config.hidden_size, # Key change: pass as embed_dim
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                # Get bias from cfg if present, else default (True is common)
                bias=getattr(cfg, 'bias', True)
            ),
            user_kwargs=cfg.kwargs,
        )

    def build_feedforward(self, cfg=None):
        cfg = cfg or self.config.feedforward_config
        return self._build(
            "feedforward", cfg.type,
            base_kwargs=dict(hidden_size=self.config.hidden_size, # Keep as hidden_size here
                             intermediate_size=cfg.intermediate_size,
                             activation=cfg.activation,
                             dropout=cfg.dropout),
            user_kwargs=cfg.kwargs,
        )

    def build_value_embedding(self):
        cfg = self.config.value_embedding_config
        return self._build(
            "embedding", cfg.type,
            base_kwargs=dict(hidden_size=self.config.hidden_size, # Keep as hidden_size here
                             feature_size=self.config.feature_size),
            user_kwargs=cfg.kwargs,
        )

    def build_positional_embedding(self):
        cfg = self.config.positional_embedding_config
        return self._build("embedding", cfg.type,
                           base_kwargs=dict(hidden_size=self.config.hidden_size), # Keep as hidden_size here
                           user_kwargs=cfg.kwargs)

    def build_normalization(self):
        cfg = self.config.norm_config
        # Pass hidden_size directly to base_kwargs, which will be handled by _prepare_args
        # for classes like LayerNorm expecting 'normalized_shape'
        # Pass cfg.kwargs now that NormalizationConfig has it.
        return self._build(
            "normalization", cfg.norm_type,
            base_kwargs=dict(hidden_size=self.config.hidden_size, eps=cfg.eps), # Keep as hidden_size here
            user_kwargs=cfg.kwargs
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
