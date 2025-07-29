from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np

from transformers import PretrainedConfig

from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.attention_config import AttentionConfig, attention_config_from_dict # Import helper
from temporal.configs.output_head_config import OutputHeadConfig, output_head_config_from_dict # Import helper
from temporal.configs.transformer_block_config import TransformerBlockConfig, transformer_block_config_from_dict # Import helper
from temporal.configs.embedding_config import EmbeddingConfig, embedding_config_from_dict # Import helper
from temporal.configs.head_aggregation_config import HeadAggregationConfig, head_aggregation_config_from_dict # Import helper
from temporal.configs.normalization_config import NormalizationConfig, normalization_config_from_dict # Import helper
from temporal.configs.quantizer_config import QuantizerConfig, quantizer_config_from_dict # Import helper
from temporal.configs.loss_config import LossConfig, loss_config_from_dict, PROBABILISTIC_LOSSES # Import helper and constant


@register_config_type("base_time_series_config")
@dataclass(frozen=True)
class BaseTimeSeriesConfig(BaseConfig, PretrainedConfig):
    """
    Configuration for a base time series forecasting model.

    Inherits from :class:`transformers.PretrainedConfig` to enable saving/loading
    via the Hugging Face ecosystem. This dataclass also integrates with the
    local `BaseConfig` for `to_dict` and `from_dict` consistency.
    """
    # Note: 'type' field is inherited from BaseConfig, can be overridden if needed
    type: str = "base_time_series_config"
    
    feature_size: int = 1
    context_length: int = 128
    prediction_length: int = 12
    # Use Optional for quantiles if it can be None, then handle default in __post_init__ or property
    quantiles: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    output_token_lengths: int = 1
    loss_config: LossConfig = field(default_factory=lambda: MSELossConfig()) # Use specific loss config
    use_dynamic_features: bool = False
    use_static_features: bool = False
    autoregressive: bool = True
    is_decoder: bool = False
    
    # Add a target_dim for explicit handling, if not derived
    target_dim: Optional[int] = None # This will be set by TransformerTimeSeriesConfig typically

    def __post_init__(self):
        # Call PretrainedConfig's __init__ using object.__setattr__ for frozen dataclass
        object.__setattr__(self, "_name_or_path", self.type)
        object.__setattr__(self, "model_type", self.type)

        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")
        
        # Validation for loss_config and quantiles
        if self.loss_config.type in PROBABILISTIC_LOSSES:
            if self.quantiles is None or len(self.quantiles) == 0:
                raise ValueError(f"Probabilistic loss '{self.loss_config.type}' requires a list of quantiles.")
            if not all(0 < q < 1 for q in self.quantiles):
                raise ValueError("All quantile values must be in the open interval (0, 1)")
        
        # If target_dim is not set, default it to feature_size
        if self.target_dim is None:
            object.__setattr__(self, "target_dim", self.feature_size)

    # Override to_dict to integrate with BaseConfig's recursive to_dict
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        # PretrainedConfig might add its own fields, filter them out if they duplicate
        # fields already handled by BaseConfig's asdict().
        # However, for consistency, let's explicitly build it from dataclass fields first.
        data = asdict(self)
        
        # Manually include PretrainedConfig's relevant fields if not already present
        # and ensure nested configs are dicts
        hf_specific_fields = {"_name_or_path": self._name_or_path, "model_type": self.model_type}
        
        # Recursively convert nested BaseConfig objects to dictionaries
        for f in fields(self):
            if isinstance(getattr(self, f.name), BaseConfig):
                data[f.name] = getattr(self, f.name).to_dict()
            elif isinstance(getattr(self, f.name), list):
                data[f.name] = [
                    item.to_dict() if isinstance(item, BaseConfig) else item
                    for item in getattr(self, f.name)
                ]
        
        # Merge, giving precedence to dataclass fields, then HF fields
        return {**data, **hf_specific_fields}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        # Ensure 'type' is set for BaseConfig's from_dict if not present in input
        if "type" not in data:
            data["type"] = cls.type # Default to the current class's type

        # Handle nested loss_config specifically before passing to super
        if "loss_config" in data and isinstance(data["loss_config"], dict):
            data["loss_config"] = loss_config_from_dict(data["loss_config"])
        elif "loss_type" in data and "loss_config" not in data: # Handle old loss_type
            data["loss_config"] = loss_config_from_dict({"type": data.pop("loss_type")})
        
        # If quantiles is directly set, ensure num_quantiles logic is handled
        num_quantiles = data.pop("num_quantiles", None) # Pop it so it's not passed to init directly
        if num_quantiles is not None and data.get("quantiles") is None:
            # Reconstruct quantiles if only num_quantiles was provided
            data["quantiles"] = np.linspace(
                0.5 / num_quantiles,
                1 - 0.5 / num_quantiles,
                num_quantiles
            ).tolist()
        elif (
            num_quantiles is not None
            and data.get("quantiles") is not None
            and num_quantiles != len(data["quantiles"])
        ):
            raise ValueError(
                f"num_quantiles ({num_quantiles}) does not match "
                f"len(quantiles) ({len(data['quantiles'])}). Set one or the other."
            )

        # PretrainedConfig expects specific arguments in its __init__ (via kwargs). 
        # We need to filter `data` for fields relevant to `BaseTimeSeriesConfig`.
        # Remaining kwargs will be handled by PretrainedConfig's __init__.
        
        # Get valid fields for the current dataclass (BaseTimeSeriesConfig)
        # Using `fields` from dataclasses to get direct fields
        current_cls_fields = {f.name for f in fields(cls) if f.init}

        # Separate arguments for BaseTimeSeriesConfig's __init__ from other kwargs
        init_kwargs = {k: v for k, v in data.items() if k in current_cls_fields}
        # All other keys are passed as **kwargs to PretrainedConfig
        remaining_kwargs = {k: v for k, v in data.items() if k not in current_cls_fields}

        # Instantiate the current class using filtered kwargs
        instance = cls(**init_kwargs, **remaining_kwargs)
        
        # After instantiation, manually set hidden PretrainedConfig attributes if needed
        # This ensures they are set after __post_init__ if not already by PretrainedConfig's init
        if "_name_or_path" in data:
            object.__setattr__(instance, "_name_or_path", data["_name_or_path"])
        if "model_type" in data:
            object.__setattr__(instance, "model_type", data["model_type"])
        
        return instance


@register_config_type("transformer_time_series_config")
@dataclass(frozen=True)
class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    """
    Configuration for a transformer-based time-series forecasting model.
    Extends ``BaseTimeSeriesConfig`` with transformer-specific options.
    """
    type: str = "transformer_time_series_config"

    # Override relevant fields or add new ones
    model_type: str = "transformer" # This is for internal model selection, not the config type
    d_model: int = 64
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 4096
    architecture: TransformerArchitectureConfig = field(default_factory=TransformerArchitectureConfig)
    value_embedding_config: EmbeddingConfig = field(default_factory=lambda: embedding_config_from_dict({"type": "value"}))
    positional_embedding_config: EmbeddingConfig = field(default_factory=lambda: embedding_config_from_dict({"type": "sinusoidal"}))
    encoder_blocks: Optional[List[TransformerBlockConfig]] = None
    decoder_blocks: Optional[List[TransformerBlockConfig]] = None
    output_head_config: OutputHeadConfig = field(default_factory=lambda: output_head_config_from_dict({"type": "linear"}))
    norm_config: NormalizationConfig = field(default_factory=lambda: normalization_config_from_dict({"type": "layer"}))
    head_agg_config: HeadAggregationConfig = field(default_factory=lambda: head_aggregation_config_from_dict({"type": "mean"}))
    quantizer_config: Optional[QuantizerConfig] = None
    vocab_size: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_teacher_forcing: bool = True
    aux_loss_weight: float = 0.01
    use_cache: bool = True

    # Deprecated fields (keep for from_dict compatibility but don't use in new code)
    attention_blocks: Optional[Any] = field(default=None, repr=False, compare=False)
    feedforward_config: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__() # Call parent dataclass validation first
        
        if self.attention_blocks is not None:
            print("Warning: `attention_blocks` config key is deprecated. Configure attention within `encoder_blocks`/`decoder_blocks`.")
        if self.feedforward_config is not None:
            print("Warning: `feedforward_config` key is deprecated. Configure FFN within `encoder_blocks`/`decoder_blocks`.")

        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if not (0.0 <= self.hidden_dropout_prob <= 1.0):
            raise ValueError("hidden_dropout_prob must be in [0, 1]")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0")

        # Ensure nested configs are instances of their dataclass types
        if not isinstance(self.architecture, TransformerArchitectureConfig):
            raise ValueError("architecture must be a TransformerArchitectureConfig instance.")
        if not isinstance(self.value_embedding_config, EmbeddingConfig):
            raise ValueError("value_embedding_config must be an EmbeddingConfig instance.")
        if not isinstance(self.positional_embedding_config, EmbeddingConfig):
            raise ValueError("positional_embedding_config must be an EmbeddingConfig instance.")
        if self.encoder_blocks is not None:
            if not all(isinstance(b, TransformerBlockConfig) for b in self.encoder_blocks):
                raise ValueError("All items in encoder_blocks must be TransformerBlockConfig instances.")
        if self.decoder_blocks is not None:
            if not all(isinstance(b, TransformerBlockConfig) for b in self.decoder_blocks):
                raise ValueError("All items in decoder_blocks must be TransformerBlockConfig instances.")
        if not isinstance(self.output_head_config, OutputHeadConfig):
            raise ValueError("output_head_config must be an OutputHeadConfig instance.")
        if not isinstance(self.norm_config, NormalizationConfig):
            raise ValueError("norm_config must be a NormalizationConfig instance.")
        if not isinstance(self.head_agg_config, HeadAggregationConfig):
            raise ValueError("head_agg_config must be a HeadAggregationConfig instance.")
        if self.quantizer_config is not None and not isinstance(self.quantizer_config, QuantizerConfig):
            raise ValueError("quantizer_config must be a QuantizerConfig instance or None.")
        if not isinstance(self.loss_config, LossConfig):
            raise ValueError("loss_config must be a LossConfig instance.")

        # Specific validation for blocks and attention heads (moved from old validate_config)
        all_blocks = (self.encoder_blocks or []) + (self.decoder_blocks or [])
        for i, block_config in enumerate(all_blocks):
            if block_config.attention_config:
                heads = block_config.attention_config.num_heads
                if self.d_model % heads != 0:
                    raise ValueError(
                        f"d_model ({self.d_model}) must be divisible by num_heads ({heads}) "
                        f"in self-attention of block {i}"
                    )
            if isinstance(block_config, DecoderBlockConfig) and block_config.cross_attention_config:
                cross_heads = block_config.cross_attention_config.num_heads
                if self.d_model % cross_heads != 0:
                    raise ValueError(
                        f"d_model ({self.d_model}) must be divisible by num_heads ({cross_heads}) "
                        f"in cross-attention of block {i}"
                    )

        if self.quantizer_config and not self.vocab_size:
            print("Warning: Quantizer configured but vocab_size not set.")
        if self.architecture.layout != "encoder-only" and self.decoder_start_token_id is None and self.vocab_size is not None:
            print("Warning: Decoder model needs decoder_start_token_id if tokenized.")
        # Quantiles validation is now in BaseTimeSeriesConfig's __post_init__


    # Override from_dict to handle nested dataclasses from dicts
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        # First, process the deprecated fields and aliases from the original config
        if "attention_blocks" in data:
            print("Warning: `attention_blocks` config key is deprecated. Configure attention within `encoder_blocks`/`decoder_blocks`.")
            data.pop("attention_blocks")
        if "feedforward_config" in data:
            print("Warning: `feedforward_config` key is deprecated. Configure FFN within `encoder_blocks`/`decoder_blocks`.")
            data.pop("feedforward_config")

        if "hidden_size" in data and "d_model" not in data:
            data["d_model"] = data.pop("hidden_size")

        # Handle feature_size/input_dim for backward compatibility
        if "input_dim" not in data and "feature_size" in data:
             data["input_dim"] = data.pop("feature_size")
        elif "feature_size" in data and "input_dim" in data and data["feature_size"] != data["input_dim"]:
            print(f"Warning: Both 'input_dim' ({data['input_dim']}) and 'feature_size' ({data['feature_size']}) found in config dict. "
                  f"Preferring 'input_dim'. 'feature_size' will be ignored.")
            data.pop("feature_size")
        elif "feature_size" in data: 
            data.pop("feature_size")

        # Handle nested configurations if they are dictionaries
        if "architecture" in data and isinstance(data["architecture"], dict):
            data["architecture"] = TransformerArchitectureConfig.from_dict(data["architecture"])

        if "value_embedding_config" in data and isinstance(data["value_embedding_config"], dict):
            data["value_embedding_config"] = embedding_config_from_dict(data["value_embedding_config"])
        if "positional_embedding_config" in data and isinstance(data["positional_embedding_config"], dict):
            data["positional_embedding_config"] = embedding_config_from_dict(data["positional_embedding_config"])
        
        if "encoder_blocks" in data and isinstance(data["encoder_blocks"], list):
            data["encoder_blocks"] = [transformer_block_config_from_dict(b) if isinstance(b, dict) else b for b in data["encoder_blocks"]]
        if "decoder_blocks" in data and isinstance(data["decoder_blocks"], list):
            data["decoder_blocks"] = [transformer_block_config_from_dict(b) if isinstance(b, dict) else b for b in data["decoder_blocks"]]

        if "output_head_config" in data and isinstance(data["output_head_config"], dict):
            data["output_head_config"] = output_head_config_from_dict(data["output_head_config"])
        if "norm_config" in data and isinstance(data["norm_config"], dict):
            data["norm_config"] = normalization_config_from_dict(data["norm_config"])
        if "head_agg_config" in data and isinstance(data["head_agg_config"], dict):
            data["head_agg_config"] = head_aggregation_config_from_dict(data["head_agg_config"])
        if "quantizer_config" in data and isinstance(data["quantizer_config"], dict):
            data["quantizer_config"] = quantizer_config_from_dict(data["quantizer_config"])
        if "loss_config" in data and isinstance(data["loss_config"], dict):
            data["loss_config"] = loss_config_from_dict(data["loss_config"])
        
        # Pass all remaining data to the superclass from_dict, which handles the creation
        # and further nested parsing via the CONFIG_REGISTRY.
        return super().from_dict(data)

