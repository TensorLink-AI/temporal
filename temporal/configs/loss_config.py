from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal

from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

#: List of loss types treated as probabilistic. Each requires valid quantile values in (0, 1).
PROBABILISTIC_LOSSES = ["quantile", "mq", "crps"]

@dataclass(frozen=True, kw_only=True)
class LossConfig(BaseConfig):
    """
    Base configuration for the loss function.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # `type` field is inherited from BaseConfig and is kw_only.

    def __post_init__(self):
        super().__post_init__() # Call BaseConfig's post_init for its own validations
        pass # No common validation for all loss types here

@register_config_type("mse_loss")
@dataclass(frozen=True, kw_only=True)
class MSELossConfig(LossConfig):
    """
    Configuration for Mean Squared Error loss.
    """
    type: str = field(default="mse") # Override type field and make it kw_only

@register_config_type("crps_loss")
@dataclass(frozen=True, kw_only=True)
class CRPSLossConfig(LossConfig):
    """
    Configuration for Continuous Ranked Probability Score (CRPS) loss.
    """
    type: str = field(default="crps") # Override type field and make it kw_only
    reduction: str = field(default="mean")
    estimator: str = field(default="pinball") # Common estimator, change as needed
    spread_lambda: float = field(default=0.0)
    spread_penalty_type: str = field(default="log")
    spread_penalty_epsilon: float = field(default=0.0)
    spread_target_spread: float = field(default=0.0)
    scaling_type: str = field(default="none")
    scaling_dim: int = field(default=1)
    scaling_eps: float = field(default=1e-8)

    def __post_init__(self):
        super().__post_init__()
        if self.estimator not in ["pinball", "pwm", "nrg", "fair"]:
            raise ValueError(f"CRPS estimator must be 'pinball', 'pwm', 'nrg', or 'fair', got {self.estimator}")
        if not (0.0 <= self.spread_lambda <= 1.0):
            raise ValueError(f"spread_lambda must be in [0, 1], got {self.spread_lambda}")
        if self.spread_penalty_type not in ["log", "inverse", "symmetric_log", "none"]:
            raise ValueError(f"spread_penalty_type must be 'log', 'inverse', 'symmetric_log', or 'none', got {self.spread_penalty_type}")
        if self.spread_penalty_epsilon < 0:
            raise ValueError("spread_penalty_epsilon cannot be negative.")
        if self.scaling_type not in ["none", "std", "minmax"]:
            raise ValueError(f"scaling_type must be 'none', 'std', or 'minmax', got {self.scaling_type}")
        if self.scaling_dim <= 0:
            raise ValueError("scaling_dim must be a positive integer.")
        if self.scaling_eps < 0:
            raise ValueError("scaling_eps cannot be negative.")

@register_config_type("quantile_loss")
@dataclass(frozen=True, kw_only=True)
class QuantileLossConfig(LossConfig):
    """
    Configuration for Quantile Loss.
    """
    type: str = field(default="quantile") # Override type field and make it kw_only
    quantiles: List[float] = field(default_factory=list) # Required quantiles list

    def __post_init__(self):
        super().__post_init__()
        if not self.quantiles or not all(0 < q < 1 for q in self.quantiles):
            raise ValueError("Quantile loss requires a list of quantiles in the open interval (0, 1).")

@register_config_type("timeflow_loss")
@dataclass(frozen=True, kw_only=True)
class TimeFlowLossConfig(LossConfig):
    """
    Configuration for the TimeFlow loss module.
    """
    type: Literal["timeflow"] = "timeflow"
    target_channels: int
    cond_channels: int
    num_blocks: int
    model_channels: int
    num_sampling_steps: int = 10
    reduction: str = "mean"


@register_config_type("nll_loss")
@dataclass(frozen=True, kw_only=True)
class NLLLossConfig(LossConfig):
    """
    Configuration for Negative Log Likelihood loss.
    """
    type: str = field(default="nll")
    # All NLL-specific parameters should go into `kwargs` for module instantiation
    # `distribution_type` is required for NegativeLogLikelihoodLoss.__init__
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"distribution_type": "gaussian"}) # Default distribution type if not specified

    def __post_init__(self):
        super().__post_init__()
        # Ensure distribution_type is set in kwargs
        if "distribution_type" not in self.kwargs:
            raise ValueError("NLLLossConfig requires 'distribution_type' to be specified in kwargs.")
        if self.kwargs["distribution_type"] not in ["gaussian", "mixture"]:
            raise ValueError(f"Unsupported distribution_type for NLLLoss: {self.kwargs['distribution_type']}. Supported types are 'gaussian', 'mixture'.")


# Helper function for polymorphic creation
def loss_config_from_dict(data: Dict[str, Any]) -> LossConfig:
    loss_type = data.get("type", "mse") # Default to 'mse' if type not specified
    type_to_registry_key = {
        "mse": "mse_loss",
        "crps": "crps_loss",
        "quantile": "quantile_loss",
        "mq": "quantile_loss",
        "timeflow": "timeflow_loss",
        "nll": "nll_loss", # Map 'nll' to the new config type
    }
    registry_key = type_to_registry_key.get(loss_type, loss_type)

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, LossConfig):
        raise ValueError(f"Unknown or invalid loss_type: {loss_type} (mapped to registry key: {registry_key})")

    return config_class.from_dict(data)
