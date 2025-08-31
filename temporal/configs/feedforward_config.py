from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY

@dataclass(frozen=True, kw_only=True)
class FeedForwardConfig(BaseConfig):
    """
    Base configuration for the feed-forward network sublayer.
    Specific FFN types should inherit from this class.
    """
    activation: str = field(default="gelu")
    dropout: float = field(default=0.1)
    bias: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@register_config_type("standard_feedforward")
@dataclass(frozen=True, kw_only=True)
class StandardFeedForwardConfig(FeedForwardConfig):
    """
    Configuration for a standard feed-forward network.
    """
    type: str = field(default="standard") # Overriding type and making it kw_only
    intermediate_size: int = 128 # Now a required kw-only field

    def __post_init__(self):
        super().__post_init__()
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0")


@register_config_type("moe_feedforward")
@dataclass(frozen=True, kw_only=True)
class MoEFeedForwardConfig(FeedForwardConfig):
    """
    Configuration for a Mixture-of-Experts (MoE) feed-forward network.
    """
    num_experts: int = field(default=8) # Default added for convenience
    top_k: int = field(default=2)       # Default added for convenience
    
    type: str = field(default="moe") # Overriding type and making it kw_only
    expert_intermediate_size: Optional[int] = field(default=None) # Optional kw-only
    load_balancing_coef: float = field(default=0.01) # Defaulted kw-only
    gate_dropout: Optional[float] = field(default=None) # ADDED: For gate dropout

    def __post_init__(self):
        super().__post_init__()
        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0 for MoE")
        if not (0 < self.top_k <= self.num_experts):
            raise ValueError("top_k must be > 0 and <= num_experts for MoE")
        if self.expert_intermediate_size is not None and self.expert_intermediate_size <= 0:
            raise ValueError("expert_intermediate_size must be > 0 if specified.")
        if not (0.0 <= self.load_balancing_coef <= 1.0):
            raise ValueError(f"load_balancing_coef must be in [0, 1], got {self.load_balancing_coef}")
        if self.gate_dropout is not None and not (0.0 <= self.gate_dropout <= 1.0):
            raise ValueError(f"gate_dropout must be in [0, 1] or None, got {self.gate_dropout}")

# Helper function for polymorphic creation
def feedforward_config_from_dict(data: Dict[str, Any]) -> FeedForwardConfig:
    ffn_type = data.get("type", "standard")
    config_class = CONFIG_REGISTRY.get(ffn_type + "_feedforward")
    if not config_class or not issubclass(config_class, FeedForwardConfig):
        raise ValueError(f"Unknown or invalid feedforward_type: {ffn_type}")
    return config_class.from_dict(data)
