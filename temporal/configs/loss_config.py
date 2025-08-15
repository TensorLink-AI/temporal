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
        super().__post_init__()
        pass

# FIX: Added a new config class for the 'timeseries_generic' loss type.
@register_config_type("timeseries_generic")
@dataclass(frozen=True, kw_only=True)
class TimeSeriesLossConfig(LossConfig):
    """
    Configuration for the generic TimeSeriesLoss, which can handle mse, mae, etc.
    """
    type: str = field(default="timeseries_generic")
    loss_type: str = field(default="mse") # The actual underlying loss (e.g., 'mse', 'mae')
    quantiles: Optional[List[float]] = field(default=None)


@register_config_type("mse_loss")
@dataclass(frozen=True, kw_only=True)
class MSELossConfig(LossConfig):
    """
    Configuration for Mean Squared Error loss.
    """
    type: str = field(default="mse")

@register_config_type("crps_loss")
@dataclass(frozen=True, kw_only=True)
class CRPSLossConfig(LossConfig):
    """
    Configuration for Continuous Ranked Probability Score (CRPS) loss.
    """
    type: str = field(default="crps")
    reduction: str = field(default="mean")
    estimator: str = field(default="pinball")
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

@register_config_type("quantile_loss")
@dataclass(frozen=True, kw_only=True)
class QuantileLossConfig(LossConfig):
    """
    Configuration for Quantile Loss.
    """
    type: str = field(default="quantile")
    quantiles: List[float] = field(default_factory=list)

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
    type: str = field(default="timeflow")
    reduction: str = field(default="mean")


@register_config_type("nll_loss")
@dataclass(frozen=True, kw_only=True)
class NLLLossConfig(LossConfig):
    """
    Configuration for Negative Log Likelihood loss.
    """
    type: str = field(default="nll")
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"distribution_type": "gaussian"})

    def __post_init__(self):
        super().__post_init__()
        if "distribution_type" not in self.kwargs:
            raise ValueError("NLLLossConfig requires 'distribution_type' to be specified in kwargs.")
        if self.kwargs["distribution_type"] not in ["gaussian", "mixture"]:
            raise ValueError(f"Unsupported distribution_type for NLLLoss: {self.kwargs['distribution_type']}. Supported types are 'gaussian', 'mixture'.")


# Helper function for polymorphic creation
def loss_config_from_dict(data: Dict[str, Any]) -> LossConfig:
    loss_type = data.get("type", "mse")
    type_to_registry_key = {
        # FIX: Added 'timeseries_generic' to the map.
        "timeseries_generic": "timeseries_generic",
        "mse": "mse_loss",
        "crps": "crps_loss",
        "quantile": "quantile_loss",
        "mq": "quantile_loss",
        "timeflow": "timeflow_loss",
        "nll": "nll_loss",
    }
    registry_key = type_to_registry_key.get(loss_type, loss_type)

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, LossConfig):
        raise ValueError(f"Unknown or invalid loss_type: {loss_type} (mapped to registry key: {registry_key})")

    return config_class.from_dict(data)