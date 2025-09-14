import inspect
from typing import Dict, Any, Type, Optional
import torch.nn as nn
from dataclasses import asdict, dataclass, field, replace

from temporal.registry.core import resolve
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import FeedForwardConfig
from temporal.configs.embedding_config import EmbeddingConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.configs.head_aggregation_config import HeadAggregationConfig
from temporal.configs.loss_config import LossConfig
from temporal.configs.base_config import BaseConfig
from temporal.configs.transformer_block_config import TransformerBlockConfig # Import TransformerBlockConfig
from temporal.configs.quantizer_config import  QuantizerConfig

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
            ValueError: If the config does not contain `d_model`.
        """
        self.config = config
        self._model_dim = config.d_model
        self._feature_size = config.feature_size 

        if self._model_dim is None:
            raise ValueError("The configuration must define 'd_model'.")

    def _build(self, kind: str, module_config: BaseConfig) -> nn.Module:
        """The generic, core build method.

        This method resolves a class from the registry and instantiates it.
        It intelligently decides whether to pass the entire configuration object
        or to unpack it into keyword arguments based on the module's constructor signature.

        Args:
            kind (str): The category of the module (e.g., "attention").
            module_config (BaseConfig): The specific configuration dataclass instance
                                      for the module to be built.

        Returns:
            nn.Module: An instantiated PyTorch module.
        """
        cls = resolve(kind, module_config.type)
        signature = inspect.signature(cls.__init__)
        accepted_params = set(signature.parameters.keys())

        kwargs = {}
        # Special handling for 'block' kind to ensure 'config' is passed if expected
        if kind == "block" and ('config' in accepted_params or 'cfg' in accepted_params):
            if 'config' in accepted_params:
                kwargs['config'] = module_config
            else:
                kwargs['cfg'] = module_config
        else:
            # Decide how to pass arguments: as a single config object or unpacked.
            if 'config' in accepted_params:
                kwargs = {'config': module_config}
            elif 'cfg' in accepted_params:
                kwargs = {'cfg': module_config}
            else:
                # Fallback to unpacking the config into keyword arguments.
                kwargs = module_config.to_dict()
                if 'kwargs' in kwargs:
                    extra_kwargs = kwargs.pop('kwargs')
                    kwargs.update(extra_kwargs)
                kwargs.pop('type', None)

        # Automatically inject common model-wide parameters if the module needs them.
        if 'd_model' in accepted_params and 'd_model' not in kwargs:
            kwargs['d_model'] = self._model_dim
        elif 'embed_dim' in accepted_params and 'embed_dim' not in kwargs:
            kwargs['embed_dim'] = self._model_dim
        elif 'hidden_size' in accepted_params and 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = self._model_dim
        if 'feature_size' in accepted_params and 'feature_size' not in kwargs:
            kwargs['feature_size'] = self._feature_size
        # Special handling for normalization layers.
        if kind == "normalization" and 'normalized_shape' in accepted_params:
            if getattr(module_config, 'type', '') in ("revin", "revin2d"):
                if 'num_features' not in kwargs:
                    kwargs['num_features'] = self._model_dim
            else:
                if 'normalized_shape' not in kwargs:
                    kwargs['normalized_shape'] = self._model_dim

        # Inject the builder itself if requested.
        if 'builder' in accepted_params:
            kwargs['builder'] = self

        # Filter out any arguments that the constructor does not accept.
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}

        try:
            return cls(**filtered_kwargs)
        except TypeError as e:
            passed_args_str = ", ".join(f"{k}={type(v).__name__}" for k, v in filtered_kwargs.items())
            raise TypeError(
                f"Failed to instantiate '{module_config.type}' ({cls.__name__}) for kind '{kind}'.\n"
                f" > Provided args: {{{passed_args_str}}}.\n"
                f" > Accepted params: {accepted_params}.\n"
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

    def build_quantizer(self, cfg: QuantizerConfig) -> nn.Module:
        """Builds a quantizer module from its configuration."""
        cfg  = replace(cfg , d_model=self.config.d_model)
        return self._build("quantizer",cfg  )

    def build_normalization(self, cfg: NormalizationConfig) -> nn.Module:
        """Builds a normalization layer from a `NormalizationConfig`."""
        return self._build("normalization", cfg)

    def build_head_aggregator(self, cfg: HeadAggregationConfig) -> nn.Module:
        """Builds a head aggregator module."""
        if self.config.output_head_config.output_size is None:
             raise ValueError("output_head_config with a valid output_size must be set in the main config.")
        if self.config.output_token_lengths is None:
             raise ValueError("config.output_token_lengths must be set for head aggregation.")

        extra_kwargs = {
            "input_size": self.config.output_head_config.output_size,
            "output_size": self.config.output_head_config.output_size,
            "num_heads": self.config.output_token_lengths
        }
        
        mutable_cfg_dict = cfg.to_dict()
        mutable_cfg_dict.update(extra_kwargs)
        
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
        if hasattr(self.config, 'quantiles') and self.config.quantiles:
            mutable_cfg_dict = cfg.to_dict()
            mutable_cfg_dict['quantiles'] = self.config.quantiles
            
            @dataclass(frozen=True)
            class TempLossConfig(LossConfig):
                quantiles: Optional[list[float]] = field(default_factory=list)
                kwargs: Dict[str, Any] = field(default_factory=dict)

                def __post_init__(self):
                    super().__post_init__()
                    if 'kwargs' in mutable_cfg_dict:
                        object.__setattr__(self, 'kwargs', mutable_cfg_dict['kwargs'])
            
            temp_cfg = TempLossConfig(type=cfg.type, quantiles=mutable_cfg_dict['quantiles'], kwargs=mutable_cfg_dict.get('kwargs', {}))
            return self._build("loss", temp_cfg)
        
        return self._build("loss", cfg)
