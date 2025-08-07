from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

@register_config_type("output_head") # Generic type for OutputHeadConfig if no specific type is given
@dataclass(frozen=True, kw_only=True)
class OutputHeadConfig(BaseConfig):
    """
    Configuration for the output head of the transformer model.
    """
    type: str = field(default="linear")
    output_size: Optional[int] = field(default=None)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.output_size is None or self.output_size <= 0:
            raise ValueError("output_size must be a positive integer.")

@register_config_type("distpred_output_head")
@dataclass(frozen=True, kw_only=True)
class DistPredOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a distributional prediction head.
    """
    num_outputs: int
    feature_size: int
    type: str = field(default="distpred")
    use_tanh: bool = field(default=False)
    tanh_scale: float = field(default=10.0)

    def __post_init__(self):
        super().__post_init__()
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")
        if self.tanh_scale <= 0:
            raise ValueError("tanh_scale must be a positive float.")

@register_config_type("quantile_regression_output_head")
@dataclass(frozen=True, kw_only=True)
class QuantileRegressionOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a quantile regression output head.
    """
    num_quantiles: int
    feature_size: int = field(default=1)
    type: str = field(default="quantile_regression")

    def __post_init__(self):
        super().__post_init__()
        if self.num_quantiles <= 0:
            raise ValueError("num_quantiles must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")

@register_config_type("mixture_output_head")
@dataclass(frozen=True, kw_only=True)
class MixtureOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a Mixture Density Network (MDN) output head.
    """
    components: List[str] = field(default_factory=list)
    type: str = field(default="mixture")

    def __post_init__(self):
        super().__post_init__()
        if not self.components:
            raise ValueError("MixtureOutputHead requires at least one component.")

@register_config_type("timeflow_output_head")
@dataclass(frozen=True, kw_only=True)
class TimeFlowOutputHeadConfig(OutputHeadConfig):
    """Configuration for the TimeFlowOutputHead."""
    type: str = field(default="timeflow")
    target_channels: int
    cond_channels: int
    num_blocks: int
    model_channels: int
    num_sampling_steps: int = 10
    input_token_len: int = 1
    diffusion_batch_mul: int = 1

    def __post_init__(self):
        # We override the base post_init because this head's output size
        # is implicitly defined by target_channels and is not a direct parameter.
        super(OutputHeadConfig, self).__post_init__()


# Helper function for polymorphic creation
def output_head_config_from_dict(data: Dict[str, Any]) -> OutputHeadConfig:
    output_head_type = data.get("type", "linear")
    type_to_registry_key = {
        "linear": "output_head",
        "gaussian": "output_head",
        "distpred": "distpred_output_head",
        "quantile_regression": "quantile_regression_output_head",
        "mixture": "mixture_output_head",
        "timeflow": "timeflow_output_head",
    }
    registry_key = type_to_registry_key.get(output_head_type, output_head_type)

    config_class = CONFIG_REGISTRY.get(registry_key)
    if not config_class or not issubclass(config_class, OutputHeadConfig):
        raise ValueError(f"Unknown or invalid output_head_type: {output_head_type} (mapped to registry key: {registry_key})")
    
    return config_class.from_dict(data)
