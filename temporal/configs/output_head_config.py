from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type

@register_config_type("output_head") # Generic type for OutputHeadConfig if no specific type is given
@dataclass(frozen=True)
class OutputHeadConfig(BaseConfig):
    """
    Configuration for the output head of the transformer model.
    """
    type: str = "linear"
    output_size: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.output_size is None or self.output_size <= 0:
            raise ValueError("output_size must be a positive integer.")

# Helper function for polymorphic creation, although for now only 'linear' is defined.
def output_head_config_from_dict(data: Dict[str, Any]) -> OutputHeadConfig:
    output_head_type = data.get("type", "linear")
    config_class = CONFIG_REGISTRY.get(output_head_type) # Assuming type string directly maps to registry key
    if not config_class or not issubclass(config_class, OutputHeadConfig):
        # Fallback to base if specific type not found or invalid
        return OutputHeadConfig.from_dict(data) 
    return config_class.from_dict(data)
