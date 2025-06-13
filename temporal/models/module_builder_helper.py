import inspect
from typing import Dict, Any, Type
import torch.nn as nn
from temporal.registry.core import resolve

def _prepare_args(
    cls: Type[nn.Module],
    base_kwargs: Dict[str, Any],
    user_kwargs: Dict[str, Any],
    builder_instance=None
) -> Dict[str, Any]:
    """Prepares the final keyword arguments for a module's constructor.

    This helper function intelligently merges base and user-provided arguments,
    handles common dimension aliases (e.g., `d_model`, `hidden_size`), and
    only keeps arguments that are actually accepted by the target class's
    `__init__` method. It can also inject the `ModuleBuilder` instance itself
    if the constructor accepts a `builder` argument.

    Args:
        cls (Type[nn.Module]): The module class to be instantiated.
        base_kwargs (Dict[str, Any]): Base arguments provided by the builder
            (e.g., `num_heads`, `dropout`).
        user_kwargs (Dict[str, Any]): Arguments provided by the user in the
            configuration's `kwargs` section.
        builder_instance (optional): An instance of `ModuleBuilder` to be
            injected if the constructor accepts it.

    Returns:
        Dict[str, Any]: A clean dictionary of keyword arguments ready to be
        passed to the class constructor.
    """
    signature = inspect.signature(cls.__init__)
    accepted_params = set(signature.parameters.keys())
    
    # Merge base and user kwargs, with user kwargs taking precedence.
    merged_args = {**base_kwargs, **user_kwargs}

    # Handle common aliases for the model's main hidden dimension.
    # Find the primary dimension size from the provided arguments.
    target_dim = merged_args.get("d_model") or merged_args.get("hidden_size") or merged_args.get("embed_dim")
        
    if target_dim is not None:
        possible_aliases = {
            "d_model", "hidden_size", "embed_dim", "dim",
            "embedding_dim", "normalized_shape"
        }
        for alias in possible_aliases:
            if alias in accepted_params and alias not in merged_args:
                merged_args[alias] = target_dim
    
    # Filter the merged arguments to include only those accepted by the constructor.
    final_args = {k: v for k, v in merged_args.items() if k in accepted_params}

    # Inject the builder instance if the constructor accepts it.
    if 'builder' in accepted_params and builder_instance is not None:
        final_args['builder'] = builder_instance
        
    return final_args


class ModuleBuilder:
    """A helper class to build all primitive modules of a time series model.

    This class reads from a main configuration object and uses the module
    registry to instantiate the various components needed for the model,
    such as attention layers, embeddings, normalization layers, and loss functions.
    It encapsulates the logic for preparing arguments and handling dependencies
    between different configuration sections.

    Attributes:
        config: The main `TransformerTimeSeriesConfig` object.
    """
    def __init__(self, config):
        """Initializes the ModuleBuilder.

        Args:
            config: The main configuration object for the model.

        Raises:
            ValueError: If the config does not contain `d_model` or `hidden_size`.
        """
        self.config = config
        self._model_dim = getattr(config, 'd_model', getattr(config, 'hidden_size', None))
        if self._model_dim is None:
             raise ValueError("The configuration must define 'd_model' or 'hidden_size'.")

    def _build(
        self,
        kind: str,
        name: str,
        base_kwargs: Dict[str, Any] = None,
        user_kwargs: Dict[str, Any] = None
    ) -> nn.Module:
        """The generic, core build method.

        This method resolves a class from the registry and instantiates it
        using a combination of base arguments and user-specified arguments.

        Args:
            kind (str): The category of the module (e.g., "attention").
            name (str): The specific type of the module (e.g., "full").
            base_kwargs (Dict[str, Any], optional): Default arguments for this kind.
            user_kwargs (Dict[str, Any], optional): User-defined arguments from config.

        Returns:
            nn.Module: An instantiated PyTorch module.
        """
        base_kwargs = base_kwargs or {}
        user_kwargs = user_kwargs or {}
        cls = resolve(kind, name)

        # Automatically add the model's hidden dimension to the base arguments.
        if 'd_model' not in base_kwargs and 'hidden_size' not in base_kwargs:
             base_kwargs['d_model'] = self._model_dim
             
        kwargs = _prepare_args(cls, base_kwargs, user_kwargs, builder_instance=self)
        
        try:
            return cls(**kwargs)
        except TypeError as e:
             passed_args_str = ", ".join(f"{k}={type(v).__name__}" for k,v in kwargs.items())
             raise TypeError(
                 f"Failed to instantiate '{name}' ({cls.__name__}) for kind '{kind}'."
                 f"  > Provided args: {{{passed_args_str}}}"
                 f"  > Original error: {e}"
             ) from e

    def build_attention(self, cfg) -> nn.Module:
        """Builds an attention module from an `AttentionConfig`."""
        return self._build(
            "attention", cfg.attention_type,
            base_kwargs={
                "num_heads": cfg.num_heads,
                "dropout": cfg.dropout,
                "bias": getattr(cfg, 'bias', True),
                "use_rope": getattr(cfg, 'use_rope', False),
                "rope_base": getattr(cfg, 'rope_base', 10000),
                "use_alibi": getattr(cfg, 'use_alibi', False),
                "max_position_embeddings": getattr(self.config, 'max_position_embeddings', 4096)
            },
            user_kwargs=cfg.kwargs,
        )

    def build_feedforward(self, cfg=None) -> nn.Module:
        """Builds a feed-forward network from a `FeedForwardConfig`."""
        cfg = cfg or self.config.feedforward_config
        return self._build(
            "feedforward", cfg.type,
            base_kwargs={
                "hidden_size": self._model_dim,
                "intermediate_size": cfg.intermediate_size,
                "activation": cfg.activation,
                "dropout": cfg.dropout
            },
            user_kwargs=cfg.kwargs,
        )

    def build_value_embedding(self) -> nn.Module:
        """Builds the primary value embedding module."""
        cfg = self.config.value_embedding_config
        return self._build(
            "embedding", cfg.type,
            base_kwargs={
                "feature_size": getattr(self.config, 'feature_size', getattr(self.config, 'input_dim', 1))
            }, 
            user_kwargs=cfg.kwargs,
        )

    def build_positional_embedding(self) -> nn.Module:
        """Builds the positional embedding module."""
        cfg = self.config.positional_embedding_config
        return self._build("embedding", cfg.type, user_kwargs=cfg.kwargs)

    def build_normalization(self) -> nn.Module:
        """Builds a normalization layer from a `NormalizationConfig`."""
        cfg = self.config.norm_config
        return self._build(
            "normalization", cfg.norm_type,
            base_kwargs={"eps": cfg.eps},
            user_kwargs=cfg.kwargs
        )

    def build_head_aggregator(self) -> nn.Module:
        """Builds a head aggregator module."""
        cfg = self.config.head_agg_config
        if not hasattr(self.config, 'output_head_config') or self.config.output_head_config.output_size is None:
             raise ValueError("output_head_config with a valid output_size must be set in the main config.")
        
        if not hasattr(self.config, 'output_token_lengths'):
             raise ValueError("config.output_token_lengths must be set for head aggregation.")

        output_size = self.config.output_head_config.output_size
        return self._build(
            "head_agg", cfg.type,
            base_kwargs={
                "input_size": output_size,
                "output_size": output_size,
                "num_heads": self.config.output_token_lengths
            },
            user_kwargs=cfg.kwargs,
        )

    def build_loss(self) -> nn.Module:
        """Builds the loss function from the `loss_config`."""
        if not hasattr(self.config, 'loss_config'):
            raise ValueError("Config is missing the 'loss_config' attribute.")
        
        cfg = self.config.loss_config
        loss_type = cfg.get("type")
        if not loss_type:
            raise ValueError("'loss_config' must contain a 'type' key.")
            
        base_kwargs = {}
        if hasattr(self.config, 'quantiles'):
             base_kwargs['quantiles'] = self.config.quantiles
             
        user_kwargs = cfg.get("kwargs", {})

        return self._build(
            kind="loss",
            name=loss_type,
            base_kwargs=base_kwargs,
            user_kwargs=user_kwargs
        )
