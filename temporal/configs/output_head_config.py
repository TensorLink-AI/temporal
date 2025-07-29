from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

@register_config_type("output_head") # Generic type for OutputHeadConfig if no specific type is given
@dataclass(frozen=True)
class OutputHeadConfig(BaseConfig):
    """
    Configuration for the output head of the transformer model.
    """
    # Make 'type' a keyword-only argument to allow non-default arguments in subclasses
    type: str = field(default="linear", kw_only=True)
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
    # New non-default arguments MUST come first
    num_outputs: int
    feature_size: int

    # Inherited fields with default values, or new fields with default values,
    # must come after all non-default fields (both new and inherited).
    # Since 'type' is now kw_only in the base, we don't strictly need to re-declare it
    # for positional order, but re-declaring ensures its default is applied correctly
    # for this specific subclass type string.
    type: str = field(default="distpred", kw_only=True)
    use_tanh: bool = False # New field with default

    def __post_init__(self):
        # Call parent's __post_init__ first to ensure base validations run
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
