from dataclasses import dataclass, field
from typing import Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY # Added CONFIG_REGISTRY

@register_config_type("quantizer") # Generic type for QuantizerConfig
@dataclass(frozen=True, kw_only=True)
class QuantizerConfig(BaseConfig):
    """
    Configuration for time series quantization.
    """
    type: str = field(default="mean_std_bins") # Renamed from quantization_type for consistency
    vocab_size: int = field(default=4096)
    num_features: int = field(default=1)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer.")
        # Add validation for different quantization types if needed

# Helper function for polymorphic creation, although for now only 'mean_std_bins' is defined.
def quantizer_config_from_dict(data: Dict[str, Any]) -> QuantizerConfig:
    quant_type = data.get("type", "mean_std_bins")
    config_class = CONFIG_REGISTRY.get(quant_type) # Assuming type string directly maps to registry key
    if not config_class or not issubclass(config_class, QuantizerConfig):
        return QuantizerConfig.from_dict(data) # Fallback to base if specific type not found or invalid
    return config_class.from_dict(data)
