import inspect
from typing import Dict, Any, Type, Optional
import torch.nn as nn
from dataclasses import asdict, dataclass, field # Import dataclass and field here

from temporal.registry.core import resolve
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import FeedForwardConfig
from temporal.configs.embedding_config import EmbeddingConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.configs.head_aggregation_config import HeadAggregationConfig
from temporal.configs.loss_config import LossConfig
from temporal.configs.base_config import BaseConfig # Import BaseConfig


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
    def __init__(self, config: TransformerTimeSeriesConfig):
        """Initializes the ModuleBuilder.

        Args:
            config: The main configuration object for the model.

        Raises:
            ValueError: If the config does not contain `d_model` or `hidden_size`.
        """
        self.config = config
        self._model_dim = config.d_model # d_model is now guaranteed by TransformerTimeSeriesConfig __post_init__
        if self._model_dim is None:
             raise ValueError("The configuration must define 'd_model'.")

    def _build(
        self,
        kind: str,
        module_config: BaseConfig # Accepts a BaseConfig instance directly
    ) -> nn.Module:
        """The generic, core build method.

        This method resolves a class from the registry and instantiates it
        using arguments derived directly from the provided module_config dataclass.

        Args:
            kind (str): The category of the module (e.g., "attention").
            module_config (BaseConfig): The specific configuration dataclass instance
                                      for the module to be built.

        Returns:
            nn.Module: An instantiated PyTorch module.
        """
        cls = resolve(kind, module_config.type) # Use module_config.type as the name
        
        # Handle "block" kind specifically to pass the entire block_config as 'config'
        # and inject the builder
        if kind == "block":
            kwargs = {"config": module_config}
        else:
            # Convert dataclass to dict; `to_dict` handles nested configs appropriately
            kwargs = module_config.to_dict()
            
            # Remove the 'type' key as it's used for registry lookup, not module init
            kwargs.pop('type', None)

        # Automatically add the model's hidden dimension if the module constructor accepts it
        # and it's not already provided by the config itself (e.g., embedding_dim for embeddings)
        signature = inspect.signature(cls.__init__)
        accepted_params = set(signature.parameters.keys())

        if 'd_model' in accepted_params and 'd_model' not in kwargs:
             kwargs['d_model'] = self._model_dim
        elif 'embed_dim' in accepted_params and 'embed_dim' not in kwargs:
             kwargs['embed_dim'] = self._model_dim
        elif 'hidden_size' in accepted_params and 'hidden_size' not in kwargs:
             kwargs['hidden_size'] = self._model_dim

        # Special handling for normalization layers: inject normalized_shape
        if kind == "normalization" and 'normalized_shape' in accepted_params:
            if module_config.type == "revin" or module_config.type == "revin2d":
                # RevIN and RevIN2D use 'num_features'
                if 'num_features' not in kwargs:
                    kwargs['num_features'] = self._model_dim
            else:
                # Default LayerNorm, RMSNorm, ScaleNorm use 'normalized_shape'
                if 'normalized_shape' not in kwargs:
                    kwargs['normalized_shape'] = self._model_dim

        # Inject the builder instance if the constructor accepts it.
        # This is particularly relevant for block types which need the builder for sub-modules.
        if 'builder' in accepted_params:
            kwargs['builder'] = self
        
        try:
            return cls(**kwargs)
        except TypeError as e:
             passed_args_str = ", ".join(f"{k}={type(v).__name__}" for k,v in kwargs.items())
             raise TypeError(
                 f"Failed to instantiate '{module_config.type}' ({cls.__name__}) for kind '{kind}'. "
                 f" > Provided args: {{{passed_args_str}}}. "
                 f" > Original error: {e}"
             ) from e

    def build_attention(self, cfg: AttentionConfig) -> nn.Module:
        """Builds an attention module from an `AttentionConfig`."""
        return self._build("attention", cfg)

    def build_feedforward(self, cfg: FeedForwardConfig) -> nn.Module:
        """Builds a feed-forward network from a `FeedForwardConfig`."""
        return self._build("feedforward", cfg)

    def build_value_embedding(self, cfg: EmbeddingConfig) -> nn.Module:
        """Builds the primary value embedding module."""
        return self._build("embedding", cfg)

    def build_positional_embedding(self, cfg: EmbeddingConfig) -> nn.Module:
        """Builds the positional embedding module."""
        return self._build("embedding", cfg)

    def build_normalization(self, cfg: NormalizationConfig) -> nn.Module:
        """Builds a normalization layer from a `NormalizationConfig`."""
        return self._build("normalization", cfg)

    def build_head_aggregator(self, cfg: HeadAggregationConfig) -> nn.Module:
        """Builds a head aggregator module."""
        # Ensure necessary config attributes are available before building
        if self.config.output_head_config.output_size is None:
             raise ValueError("output_head_config with a valid output_size must be set in the main config.")
        if self.config.output_token_lengths is None:
             raise ValueError("config.output_token_lengths must be set for head aggregation.")

        # Add these to the kwargs passed to the aggregator, as they are not part of its config dataclass
        # but are model-level parameters it might need.
        # These should be passed via kwargs to the module's __init__.
        extra_kwargs = {
            "input_size": self.config.output_head_config.output_size,
            "output_size": self.config.output_head_config.output_size,
            "num_heads": self.config.output_token_lengths
        }
        
        # Temporarily create a mutable dict to merge kwargs
        mutable_cfg_dict = cfg.to_dict()
        mutable_cfg_dict.update(extra_kwargs)
        
        # Reconstruct a temporary BaseConfig for _build, or adapt _build to take dict directly
        # For simplicity and to maintain _build's signature, let's create a temporary config-like object
        @dataclass(frozen=True)
        class TempAggConfig(BaseConfig):
            type: str = cfg.type
            input_size: int = extra_kwargs["input_size"]
            output_size: int = extra_kwargs["output_size"]
            num_heads: int = extra_kwargs["num_heads"]
            kwargs: Dict[str, Any] = field(default_factory=dict)

            def __post_init__(self):
                object.__setattr__(self, "kwargs", cfg.kwargs)

        temp_cfg = TempAggConfig(type=cfg.type, **extra_kwargs, kwargs=cfg.kwargs)
        return self._build("head_agg", temp_cfg)

    def build_loss(self, cfg: LossConfig) -> nn.Module:
        """Builds the loss function from a `LossConfig`."""
        # Loss function might need quantiles from the main config
        if hasattr(self.config, 'quantiles') and self.config.quantiles:
            # Create a mutable copy of the config dict to add quantiles
            mutable_cfg_dict = cfg.to_dict()
            mutable_cfg_dict['quantiles'] = self.config.quantiles
            
            # Create a temporary config dataclass for _build method
            @dataclass(frozen=True)
            class TempLossConfig(LossConfig):
                quantiles: Optional[list[float]] = field(default_factory=list)
                kwargs: Dict[str, Any] = field(default_factory=dict)

                def __post_init__(self):
                    super().__post_init__()
                    # Ensure kwargs from original config are preserved
                    if 'kwargs' in mutable_cfg_dict:
                        object.__setattr__(self, 'kwargs', mutable_cfg_dict['kwargs'])
            
            temp_cfg = TempLossConfig(type=cfg.type, quantiles=mutable_cfg_dict['quantiles'], kwargs=mutable_cfg_dict.get('kwargs', {}))
            return self._build("loss", temp_cfg)
        
        return self._build("loss", cfg)
