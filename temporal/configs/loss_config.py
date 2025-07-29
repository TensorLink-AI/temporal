from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

#: List of loss types treated as probabilistic. Each requires valid quantile values in (0, 1).
PROBABILISTIC_LOSSES = ["quantile", "mq", "crps"]

@dataclass(frozen=True)
class LossConfig(BaseConfig):
    """
    Base configuration for the loss function.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass # No common validation for all loss types here

@register_config_type("mse_loss")
@dataclass(frozen=True)
class MSELossConfig(LossConfig):
    """
    Configuration for Mean Squared Error loss.
    """
    type: str = "mse"

@register_config_type("crps_loss")
@dataclass(frozen=True)
class CRPSLossConfig(LossConfig):
    """
    Configuration for Continuous Ranked Probability Score (CRPS) loss.
    """
    type: str = "crps"
    # CRPS might have specific kwargs, e.g., reduction, but for now just type
    def __post_init__(self):
        super().__post_init__()
        # Add CRPS-specific validation if needed

@register_config_type("quantile_loss")
@dataclass(frozen=True)
class QuantileLossConfig(LossConfig):
    """
    Configuration for Quantile Loss.
    """
    type: str = "quantile"
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

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
