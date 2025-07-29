from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Sequence, Callable
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY


@dataclass(frozen=True, kw_only=True)
class EmbeddingConfig(BaseConfig):
    """
    Base configuration for embedding layers.
    Specific embedding types should inherit from this class.
    """
    dropout: float = field(default=0.1)
    embedding_dim: Optional[int] = field(default=None) # Will be d_model from overall config

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@register_config_type("value_embedding")
@dataclass(frozen=True, kw_only=True)
class TimeSeriesValueEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for a simple linear value embedding.
    """
    type: str = field(default="value") # Override type and make it kw_only
    feature_size: int = field(default=1)
    use_value_norm: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")


@register_config_type("flexible_value_embedding")
@dataclass(frozen=True, kw_only=True)
class FlexibleValueEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for a flexible value embedding.
    """
    input_dims: Union[int, Sequence[int]] # Required kw-only field

    type: str = field(default="flexible_value") # Override type and make it kw_only
    # proj_builder: Callable[[int, int, Dict[str, Any]], nn.Module] = field(default=None) # Cannot be dataclass field
    proj_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_layer_norm: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.input_dims, int) and self.input_dims <= 0:
            raise ValueError("input_dims must be a positive integer or sequence of positive integers.")
        if isinstance(self.input_dims, (list, tuple)):
            if not all(isinstance(dim, int) and dim > 0 for dim in self.input_dims):
                raise ValueError("All dimensions in input_dims sequence must be positive integers.")


@register_config_type("sinusoidal_positional_embedding")
@dataclass(frozen=True, kw_only=True)
class SinusoidalPositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for fixed sinusoidal positional embeddings.
    """
    type: str = field(default="sinusoidal")
    max_seq_len: int = field(default=2048)

    def __post_init__(self):
        super().__post_init__() # Use () for super().__post_init__
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")


@register_config_type("patch_embedding")
@dataclass(frozen=True, kw_only=True)
class TimeSeriesPatchEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for patch embeddings.
    """
    patch_size: int   # Required kw-only field
    feature_size: int # Required kw-only field

    type: str = field(default="patch")
    output_patch_size: Optional[int] = field(default=None)
    stride: Optional[int] = field(default=None)
    pad_value: float = field(default=0.0)
    use_mlp: bool = field(default=False)
    mlp_hidden_size: Optional[int] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")
        if self.output_patch_size is not None and self.output_patch_size <= 0:
            raise ValueError("output_patch_size must be a positive integer if specified.")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be a positive integer if specified.")
        if self.use_mlp and self.mlp_hidden_size is not None and self.mlp_hidden_size <= 0:
            raise ValueError("mlp_hidden_size must be a positive integer if use_mlp is True and specified.")


@register_config_type("global_embedding")
@dataclass(frozen=True, kw_only=True)
class TimeSeriesGlobalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for global embedding.
    """
    seq_len: int      # Required kw-only field
    feature_size: int # Required kw-only field

    type: str = field(default="global")

    def __post_init__(self):
        super().__post_init__()
        if self.seq_len <= 0:
            raise ValueError("seq_len must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")


@register_config_type("rotary_positional_embedding")
@dataclass(frozen=True, kw_only=True)
class RotaryPositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for Rotary Positional Embedding.
    """
    type: str = field(default="rotary")
    max_seq_len: int = field(default=2048)
    base: int = field(default=10000)

    def __post_init__(self):
        super().__post_init__()
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if self.base <= 0:
            raise ValueError("base must be a positive integer.")


@register_config_type("learned_absolute_positional_embedding")
@dataclass(frozen=True, kw_only=True)
class LearnedAbsolutePositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for learned absolute positional embeddings.
    """
    type: str = field(default="learned_abs")
    max_seq_len: int = field(default=2048)

    def __post_init__(self):
        super().__post_init__()
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")


@register_config_type("shaw_relative_positional_bias")
@dataclass(frozen=True, kw_only=True)
class ShawRelativePositionalBiasConfig(EmbeddingConfig):
    """
    Configuration for Shaw relative positional bias.
    """
    num_heads: int # Required kw-only field

    type: str = field(default="relative_shaw")
    max_distance: int = field(default=128)

    def __post_init__(self):
        super().__post_init__()
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be a positive integer.")


@register_config_type("fourier_feature_embedding")
@dataclass(frozen=True, kw_only=True)
class FourierFeatureEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for Fourier Feature Embedding.
    """
    type: str = field(default="fourier")
    num_features: int = field(default=16)

    def __post_init__(self):
        super().__post_init__()
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer.")


@register_config_type("time2vec_embedding")
@dataclass(frozen=True, kw_only=True)
class Time2VecEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for Time2Vec embedding.
    """
    type: str = field(default="time2vec")
    use_cos: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.embedding_dim is not None and self.embedding_dim < 2:
            raise ValueError("d_model (embedding_dim) must be >= 2 for Time2Vec.")


@register_config_type("alibi_positional_bias")
@dataclass(frozen=True, kw_only=True)
class ALiBiPositionalBiasConfig(EmbeddingConfig):
    """
    Configuration for ALiBi Positional Bias.
    """
    num_heads: int # Required kw-only field

    type: str = field(default="alibi")
    max_seq_len: int = field(default=2048)

    def __post_init__(self):
        super().__post_init__()
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")


@register_config_type("bucketed_relative_bias")
@dataclass(frozen=True, kw_only=True)
class BucketedRelativeBiasConfig(EmbeddingConfig):
    """
    Configuration for Bucketed Relative Bias.
    """
    num_heads: int # Required kw-only field

    type: str = field(default="bucketed")
    num_buckets: int = field(default=32)
    max_distance: int = field(default=128)

    def __post_init__(self):
        super().__post_init__()
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer.")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be a positive integer.")


@register_config_type("conv_pos_embedding")
@dataclass(frozen=True, kw_only=True)
class ConvolutionalPositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for Convolutional Positional Embedding.
    """
    type: str = field(default="conv_pos")
    kernel_size: int = field(default=3)
    max_seq_len: int = field(default=2048)

    def __post_init__(self):
        super().__post_init__()
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")


@register_config_type("time_delta_embedding")
@dataclass(frozen=True, kw_only=True)
class TimeDeltaEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for TimeDelta Embedding.
    """
    type: str = field(default="timedelta")
    hidden_dim: int = field(default=64)

    def __post_init__(self):
        super().__post_init__()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")


@register_config_type("stacked_embedding")
@dataclass(frozen=True, kw_only=True)
class StackedPositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for a Stacked Positional Embedding.
    """
    type: str = field(default="stacked_embedding")
    embedding_configs: List[EmbeddingConfig] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if not self.embedding_configs:
            raise ValueError("Stacked positional embedding must have at least one embedding_config.")
        for config in self.embedding_configs:
            if not isinstance(config, EmbeddingConfig):
                raise ValueError(f"All items in embedding_configs must be EmbeddingConfig instances, got {type(config).__name__}")


@register_config_type("none_embedding")
@dataclass(frozen=True, kw_only=True)
class NoneEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for a placeholder None Embedding.
    """
    type: str = field(default="none")


@register_config_type("s4_positional_embedding")
@dataclass(frozen=True, kw_only=True)
class S4PositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for S4 Positional Embedding.
    """
    type: str = field(default="s4")
    kernel_size: int = field(default=512)
    max_seq_len: int = field(default=4096)

    def __post_init__(self):
        super().__post_init__()
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")


@register_config_type("wavelet_positional_embedding")
@dataclass(frozen=True, kw_only=True)
class WaveletPositionalEmbeddingConfig(EmbeddingConfig):
    """
    Configuration for Wavelet Positional Embedding.
    """
    type: str = field(default="wavelet")
    wavelet: str = field(default="db4")
    level: int = field(default=3)
    max_seq_len: int = field(default=2048)

    def __post_init__(self):
        super().__post_init__()
        if self.level <= 0:
            raise ValueError("level must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")


# Helper function for polymorphic creation
def embedding_config_from_dict(data: Dict[str, Any]) -> EmbeddingConfig:
    embedding_type = data.get("type", "value") # Default to 'value' if type not specified
    # Map config type names to registry keys if they differ (e.g., "value" -> "value_embedding")
    type_to_registry_key = {
        "value": "value_embedding",
        "flexible_value": "flexible_value_embedding",
        "sinusoidal": "sinusoidal_positional_embedding",
        "patch": "patch_embedding",
        "global": "global_embedding",
        "rotary": "rotary_positional_embedding",
        "learned_abs": "learned_absolute_positional_embedding",
        "relative_shaw": "shaw_relative_positional_bias",
        "fourier": "fourier_feature_embedding",
        "time2vec": "time2vec_embedding",
        "alibi": "alibi_positional_bias",
        "bucketed": "bucketed_relative_bias",
        "conv_pos": "conv_pos_embedding",
        "timedelta": "time_delta_embedding",
        "stacked_embedding": "stacked_embedding",
        "none": "none_embedding",
        "s4": "s4_positional_embedding",
        "wavelet": "wavelet_positional_embedding",
    }
    registry_key = type_to_registry_key.get(embedding_type, embedding_type) # Fallback to type if not in map

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, EmbeddingConfig):
        raise ValueError(f"Unknown or invalid embedding_type: {embedding_type} (mapped to registry key: {registry_key})")
    
    # Special handling for 'embedding_configs' in StackedPositionalEmbeddingConfig
    if registry_key == "stacked_embedding" and "embedding_configs" in data:
        data["embedding_configs"] = [
            embedding_config_from_dict(cfg) if isinstance(cfg, dict) else cfg
            for cfg in data["embedding_configs"]
        ]

    return config_class.from_dict(data)
