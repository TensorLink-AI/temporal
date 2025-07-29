from dataclasses import dataclass, field
from typing import Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type

@register_config_type("head_aggregation") # Generic type for HeadAggregationConfig if no specific type is given
@dataclass(frozen=True, kw_only=True)
class HeadAggregationConfig(BaseConfig):
    """
    Configuration for combining multiple output heads.
    """
    type: str = field(default="mean")
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Add any validation for head aggregation types here
        pass

# Helper function for polymorphic creation, although for now only 'mean' is defined.
def head_aggregation_config_from_dict(data: Dict[str, Any]) -> HeadAggregationConfig:
    agg_type = data.get("type", "mean")
    config_class = CONFIG_REGISTRY.get(agg_type) # Assuming type string directly maps to registry key
    if not config_class or not issubclass(config_class, HeadAggregationConfig):
        return HeadAggregationConfig.from_dict(data) # Fallback to base if specific type not found or invalid
    return config_class.from_dict(data)
