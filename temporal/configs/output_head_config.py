from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

@register_config_type("output_head")
@dataclass(frozen=True, kw_only=True)
class OutputHeadConfig(BaseConfig):
    """
    Configuration for the output head of the transformer model.
    """
    # All fields are keyword-only to avoid dataclass ordering issues
    type: str = field(default="linear", kw_only=True)
    output_size: int
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.output_size is None or self.output_size <= 0:
            raise ValueError("output_size must be a positive integer.")

@register_config_type("distpred_output_head")
@dataclass(frozen=True, kw_only=True)
class DistPredOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a distributional prediction head.
    """
    # Required fields
    num_outputs: int
    feature_size: int

    # Override default type for this subclass
    type: str = field(default="distpred", kw_only=True)
    use_tanh: bool = False

    def __post_init__(self):
        # Ensure base validations run first
        super().__post_init__()
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")


def output_head_config_from_dict(data: Dict[str, Any]) -> OutputHeadConfig:
    """
    Construct the appropriate OutputHeadConfig subclass from a dict.
    """
    output_head_type = data.get("type", "linear")
    # Map user-facing types to registry keys
    type_to_registry_key = {
        "linear": "output_head",
        "distpred": "distpred_output_head",
    }
    registry_key = type_to_registry_key.get(output_head_type, output_head_type)

    config_class = CONFIG_REGISTRY.get(registry_key)
    if not config_class or not issubclass(config_class, OutputHeadConfig):
        raise ValueError(
            f"Unknown or invalid output_head_type: {output_head_type}"
        )

    return config_class.from_dict(data)
