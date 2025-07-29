from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
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
    # CRPS might have specific kwargs, e.g., reduction, but for now just type
    reduction: str = field(default="mean")
    estimator: str = field(default="pinball") # Common estimator, change as needed
    spread_lambda: float = field(default=0.0)
    spread_penalty_type: str = field(default="log")
    spread_penalty_epsilon: float = field(default=0.0)
    spread_target_spread: float = field(default=0.0)

    def __post_init__(self):
        super().__post_init__()
        # Add CRPS-specific validation here
        if self.estimator not in ["pinball", "pwm"]:
            raise ValueError(f"CRPS estimator must be 'pinball' or 'pwm', got {self.estimator}")
        if not (0.0 <= self.spread_lambda <= 1.0):
            raise ValueError(f"spread_lambda must be in [0, 1], got {self.spread_lambda}")
        if self.spread_penalty_type not in ["log", "inverse", "symmetric_log", "none"]:
            raise ValueError(f"spread_penalty_type must be 'log', 'inverse', 'symmetric_log', or 'none', got {self.spread_penalty_type}")
        if self.spread_penalty_epsilon < 0:
            raise ValueError("spread_penalty_epsilon cannot be negative.")

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


# Helper function for polymorphic creation
def loss_config_from_dict(data: Dict[str, Any]) -> LossConfig:
    loss_type = data.get("type", "mse") # Default to 'mse' if type not specified
    # Map config type names to registry keys if they differ
    type_to_registry_key = {
        "mse": "mse_loss",
        "crps": "crps_loss",
        "quantile": "quantile_loss",
        "mq": "quantile_loss", # Multi-quantile loss usually uses the same config as quantile
    }
    registry_key = type_to_registry_key.get(loss_type, loss_type) # Fallback to type if not in map

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, LossConfig):
        raise ValueError(f"Unknown or invalid loss_type: {loss_type} (mapped to registry key: {registry_key})")

    return config_class.from_dict(data)
