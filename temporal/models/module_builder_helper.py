import inspect
from typing import Dict, Any
import torch.nn as nn
from temporal.registry.core import resolve

def _prepare_args(cls: type[nn.Module],
                  base: Dict[str, Any],
                  extra: Dict[str, Any],
                  builder_instance=None) -> Dict[str, Any]:
    """
    Merge `base` + `extra`, keep only ctor-accepted keys, add hidden-size aliases,
    and optionally add the builder instance if accepted.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    args = {**base, **extra}

    # --- Refined Alias Handling --- 
    # Determine the primary dimension size from args (d_model > hidden_size > embed_dim)
    target_dim = None
    if "d_model" in args:
        target_dim = args["d_model"]
    elif "hidden_size" in args:
        target_dim = args["hidden_size"]
    elif "embed_dim" in args:
        target_dim = args["embed_dim"]
        
    # If a dimension size was found, add missing aliases if the constructor accepts them
    if target_dim is not None:
        possible_aliases = {
            "d_model", "hidden_size", "embed_dim", "dim", 
            "embedding_dim", "normalized_shape" # Add any other common aliases
        }
        for alias in possible_aliases:
            if alias in params and alias not in args:
                args[alias] = target_dim
    # --- End Refined Alias Handling ---
    
    # Check if constructor accepts 'builder' argument
    accepts_builder = 'builder' in params

    # Drop unsupported keys, but keep track if 'builder' was originally passed
    allowed = {p for p in params if p not in ("self", "args", "kwargs")}
    final_args = {k: v for k, v in args.items() if k in allowed and k != 'builder'}

    # Add the builder instance if the constructor accepts it and an instance was provided
    if accepts_builder and builder_instance is not None:
        final_args['builder'] = builder_instance
        
    # Handle original 'builder' argument if passed and constructor accepts it
    if 'builder' in args and 'builder' not in final_args and accepts_builder: 
        final_args['builder'] = args['builder']
        
    # print(f"_prepare_args for {cls.__name__}: final_args = {final_args.keys()}") # Debug print
    return final_args


class ModuleBuilder:
    def __init__(self, config):
        self.config = config
        # Store d_model or hidden_size for convenience
        self._model_dim = getattr(config, 'd_model', getattr(config, 'hidden_size', None))
        if self._model_dim is None:
             raise ValueError("Config must have d_model or hidden_size")

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
        # Automatically add model dimension to base_kwargs for alias handling
        # Ensure it's added *before* calling _prepare_args
        if 'd_model' not in base_kwargs and 'hidden_size' not in base_kwargs:
             base_kwargs['d_model'] = self._model_dim # Use stored dim
             
        kwargs = _prepare_args(cls, base_kwargs, user_kwargs, builder_instance=self)
        # print(f"Building {kind}/{name} ({cls.__name__}) with args: {kwargs.keys()}") # Debug print keys
        try:
            return cls(**kwargs)
        except TypeError as e:
             # Improve error message to show which arguments were actually passed
             passed_args_str = ", ".join(f"{k}={v!r}" for k,v in kwargs.items()) # Show values too
             raise TypeError(f"Error instantiating {kind}/{name} ({cls.__name__}) with args [{passed_args_str}]: {e}") from e

    # ------------------------------------------------------------------
    # Specific helpers
    # ------------------------------------------------------------------
    def build_attention(self, cfg):
        # Base kwargs now mostly handled by _build and _prepare_args alias logic
        return self._build(
            "attention", cfg.attention_type,
            base_kwargs=dict(
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                bias=getattr(cfg, 'bias', True)
            ),
            user_kwargs=cfg.kwargs,
        )

    def build_feedforward(self, cfg=None):
        cfg = cfg or self.config.feedforward_config
        # --- Explicitly add hidden_size to base_kwargs --- # Corrected
        return self._build(
            "feedforward", cfg.type,
            base_kwargs=dict( 
                             hidden_size=self._model_dim, # Pass the model dim directly
                             intermediate_size=cfg.intermediate_size,
                             activation=cfg.activation,
                             dropout=cfg.dropout),
            user_kwargs=cfg.kwargs,
        )

    def build_value_embedding(self):
        cfg = self.config.value_embedding_config
        # Base kwargs (d_model/hidden_size, feature_size) handled by _prepare_args
        return self._build(
            "embedding", cfg.type,
            base_kwargs={ 
                "feature_size": getattr(self.config, 'feature_size', getattr(self.config, 'input_dim', 1))
                }, 
            user_kwargs=cfg.kwargs,
        )

    def build_positional_embedding(self):
        cfg = self.config.positional_embedding_config
        # Base kwargs (d_model/hidden_size) handled by _build/_prepare_args
        return self._build("embedding", cfg.type,
                           base_kwargs={},
                           user_kwargs=cfg.kwargs)

    def build_normalization(self):
        cfg = self.config.norm_config
         # Base kwargs (d_model/hidden_size/normalized_shape) handled by _prepare_args
        return self._build(
            "normalization", cfg.norm_type,
            base_kwargs=dict(eps=cfg.eps),
            user_kwargs=cfg.kwargs
        )

    def build_head_aggregator(self):
        cfg = self.config.head_agg_config
        # Ensure output_head_config exists and has output_size
        if not hasattr(self.config, 'output_head_config') or not hasattr(self.config.output_head_config, 'output_size'):
             raise ValueError("output_head_config with output_size must be set in main config.")
        out_sz = self.config.output_head_config.output_size
        if out_sz is None:
            raise ValueError("output_head_config.output_size cannot be None.")
            
        if not hasattr(self.config, 'output_token_lengths'):
             raise ValueError("config.output_token_lengths must be set for head aggregation.")

        return self._build(
            "head_agg", cfg.type,
            base_kwargs=dict(input_size=out_sz,
                             output_size=out_sz,
                             num_heads=self.config.output_token_lengths),
            user_kwargs=cfg.kwargs,
        )

    # --- build_loss method ---
    def build_loss(self):
        """Builds the loss function based on config.loss_config."""
        if not hasattr(self.config, 'loss_config'):
            raise ValueError("Config is missing 'loss_config' attribute.")
        
        cfg = self.config.loss_config
        loss_type = cfg.get("type")
        if not loss_type:
            raise ValueError("loss_config must contain a 'type' key.")
            
        # Prepare base kwargs (pass quantiles if available and loss needs it)
        base_kwargs = {}
        if hasattr(self.config, 'quantiles'):
             base_kwargs['quantiles'] = self.config.quantiles
             
        # User kwargs from the loss_config itself
        user_kwargs = cfg.get("kwargs", {})

        # Build the loss module
        return self._build(
            kind="loss",
            name=loss_type,
            base_kwargs=base_kwargs,
            user_kwargs=user_kwargs
        )
    # --- End build_loss method ---

