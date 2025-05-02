import inspect
from typing import Dict, Any
import torch.nn as nn
from temporal.registry.core import resolve

def _prepare_args(cls: type[nn.Module],
                  base: Dict[str, Any],
                  extra: Dict[str, Any],
                  builder_instance=None) -> Dict[str, Any]: # Added builder_instance
    """
    Merge `base` + `extra`, keep only ctor-accepted keys, add hidden-size aliases,
    and optionally add the builder instance if accepted.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    args = {**base, **extra}

    # handle common hidden-size aliases
    hidden = args.get("hidden_size")
    embed_dim_val = args.get("embed_dim")
    target_dim = embed_dim_val if embed_dim_val is not None else hidden

    if target_dim is not None:
        if 'embed_dim' in params and 'embed_dim' not in args:
             args['embed_dim'] = target_dim
        for alias in ("d_model", "dim", "embedding_dim", "normalized_shape"):
            if alias in params and alias not in args:
                args[alias] = target_dim

    # Check if constructor accepts 'builder' argument
    accepts_builder = 'builder' in params

    # Drop unsupported keys, but keep track if 'builder' was originally passed
    allowed = {p for p in params if p not in ("self", "args", "kwargs")}
    final_args = {k: v for k, v in args.items() if k in allowed and k != 'builder'}

    # Add the builder instance if the constructor accepts it and an instance was provided
    if accepts_builder and builder_instance is not None:
        final_args['builder'] = builder_instance
        
    # Handle original 'builder' argument if passed and constructor accepts it
    # This prevents accidentally overwriting a specifically passed builder arg 
    # if the calling code somehow passed one in base or extra kwargs.
    if 'builder' in args and 'builder' not in final_args and accepts_builder: 
        final_args['builder'] = args['builder']
        
    # print(f"_prepare_args for {cls.__name__}: final_args = {final_args.keys()}") # Debug print
    return final_args


class ModuleBuilder:
    def __init__(self, config):
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
        # === Pass self (the builder instance) to _prepare_args ===
        kwargs = _prepare_args(cls, base_kwargs, user_kwargs, builder_instance=self)
        # print(f"Building {kind}/{name} ({cls.__name__}) with args: {kwargs.keys()}") # Debug print keys
        try:
            return cls(**kwargs)
        except TypeError as e:
             raise TypeError(f"Error instantiating {kind}/{name} ({cls.__name__}) with args {list(kwargs.keys())}: {e}") from e

    # ------------------------------------------------------------------
    # Specific helpers remain unchanged, _build now handles builder injection
    # ------------------------------------------------------------------
    def build_attention(self, cfg):
        return self._build(
            "attention", cfg.attention_type,
            base_kwargs=dict(
                embed_dim=self.config.hidden_size,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                bias=getattr(cfg, 'bias', True)
            ),
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
        return self._build(
            "normalization", cfg.norm_type,
            base_kwargs=dict(hidden_size=self.config.hidden_size, eps=cfg.eps),
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
