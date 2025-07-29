from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type

@register_config_type("quantizer") # Generic type for QuantizerConfig
@dataclass(frozen=True)
class QuantizerConfig(BaseConfig):
    """
    Configuration for time series quantization.
    """
    type: str = "mean_std_bins" # Renamed from quantization_type for consistency
    vocab_size: int = 4096
    num_features: int = 1
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
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
