from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type

@register_config_type("normalization") # Generic type for NormalizationConfig if no specific type is given
@dataclass(frozen=True)
class NormalizationConfig(BaseConfig):
    """
    Configuration for normalization layers.
    """
    type: str = "layer" # Renamed from norm_type for consistency with 'type' field
    eps: float = 1e-5
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.eps <= 0:
            raise ValueError("eps (epsilon) must be a positive float.")
        # Add validation for different normalization types if needed

# Helper function for polymorphic creation, although for now only 'layer' is defined.
def normalization_config_from_dict(data: Dict[str, Any]) -> NormalizationConfig:
    norm_type = data.get("type", "layer")
    config_class = CONFIG_REGISTRY.get(norm_type) # Assuming type string directly maps to registry key
    if not config_class or not issubclass(config_class, NormalizationConfig):
        return NormalizationConfig.from_dict(data) # Fallback to base if specific type not found or invalid
    return config_class.from_dict(data)
