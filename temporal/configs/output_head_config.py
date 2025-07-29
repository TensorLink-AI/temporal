from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

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

@register_config_type("distpred_output_head")
@dataclass(frozen=True)
class DistPredOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a distributional prediction head.
    """
    # New non-default arguments must come first, before any inherited fields with defaults
    # or any new fields with defaults.
    num_outputs: int
    feature_size: int
    
    # Now, inherited fields (or new fields with defaults) can follow
    type: str = "distpred" # Overrides the default from OutputHeadConfig
    use_tanh: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")

# Helper function for polymorphic creation
def output_head_config_from_dict(data: Dict[str, Any]) -> OutputHeadConfig:
    output_head_type = data.get("type", "linear")
    # Map config type names to registry keys if they differ
    type_to_registry_key = {
        "linear": "output_head", # Default for the base OutputHeadConfig
        "distpred": "distpred_output_head",
    }
    registry_key = type_to_registry_key.get(output_head_type, output_head_type) # Fallback to type if not in map

    config_class = CONFIG_REGISTRY.get(registry_key)
    if not config_class or not issubclass(config_class, OutputHeadConfig):
        raise ValueError(f"Unknown or invalid output_head_type: {output_head_type} (mapped to registry key: {registry_key})")
    
    return config_class.from_dict(data)
