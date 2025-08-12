from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

@register_config_type("normalization") # Generic type for NormalizationConfig if no specific type is given
@dataclass(frozen=True, kw_only=True)
class NormalizationConfig(BaseConfig):
    """
    Configuration for normalization layers.
    """
    type: str = field(default="layer") # Renamed from norm_type for consistency with 'type' field
    eps: float = field(default=1e-5)
    elementwise_affine: bool = field(default=True) # ADDED: Applies to LayerNorm, RMSNorm
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.eps <= 0:
            raise ValueError("eps (epsilon) must be a positive float.")
        # Add validation for different normalization types if needed


@register_config_type("revin_normalization")
@dataclass(frozen=True, kw_only=True)
class RevINConfig(NormalizationConfig):
    """
    Configuration for Reversible Instance Normalization (RevIN).
    """
    type: str = field(default="revin")
    num_features: int # Required: corresponds to d_model/feature_size
    affine: bool = field(default=True) # Matches RevIN module param
    subtract_last: bool = field(default=False) # Matches RevIN module param

    def __post_init__(self):
        super().__post_init__()
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer for RevIN.")

@register_config_type("dynamic_revin_normalization")
@dataclass(frozen=True, kw_only=True)
class DynamicRevINConfig(NormalizationConfig):
    """
    Configuration for Dynamic Reversible Instance Normalization (DynamicRevIN).
    """
    type: str = field(default="dynamic_revin")
    num_features: int
    affine_mode: Union[str, Dict[str, Any]] = field(default_factory=lambda: {"type": "fixed"})

    def __post_init__(self):
        super().__post_init__()
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer for DynamicRevIN.")

@register_config_type("revin2d_normalization")
@dataclass(frozen=True, kw_only=True)
class RevIN2dConfig(NormalizationConfig):
    """
    Configuration for 2D Reversible Instance Normalization (RevIN2d).
    """
    type: str = field(default="revin2d")
    num_features: int # Required
    affine: bool = field(default=True)
    subtract_last: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer for RevIN2d.")


# Helper function for polymorphic creation, although for now only 'layer' is defined.
def normalization_config_from_dict(data: Dict[str, Any]) -> NormalizationConfig:
    norm_type = data.get("type", "layer")
    # Map config type names to registry keys if they differ
    type_to_registry_key = {
        "layer": "normalization", # Default for the base NormalizationConfig
        "revin": "revin_normalization",
        "dynamic_revin": "dynamic_revin_normalization",
        "revin2d": "revin2d_normalization",
        "rms": "normalization", # RMSNorm uses the base NormalizationConfig type for now as its specific fields match
        "scale": "normalization", # ScaleNorm also uses base
    }
    registry_key = type_to_registry_key.get(norm_type, norm_type) # Fallback to type if not in map

    config_class = CONFIG_REGISTRY.get(registry_key) # Assuming type string directly maps to registry key
    if not config_class or not issubclass(config_class, NormalizationConfig):
        # Fallback to base if specific type not found or invalid
        return NormalizationConfig.from_dict(data) 
    return config_class.from_dict(data)
