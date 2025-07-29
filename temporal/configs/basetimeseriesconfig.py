from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Type, TypeVar
import numpy as np

from transformers import PretrainedConfig

from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY
from temporal.configs.loss_config import LossConfig, MSELossConfig, loss_config_from_dict, PROBABILISTIC_LOSSES

T = TypeVar('T', bound='BaseTimeSeriesConfig')

@register_config_type("base_time_series_config")
@dataclass(frozen=True)
class BaseTimeSeriesConfig(BaseConfig, PretrainedConfig):
    """
    Configuration for a base time series forecasting model.

    Inherits from :class:`transformers.PretrainedConfig` to enable saving/loading
    via the Hugging Face ecosystem. This dataclass also integrates with the
    local `BaseConfig` for `to_dict` and `from_dict` consistency.
    """
    # Note: 'type' field is inherited from BaseConfig, can be overridden if needed
    type: str = "base_time_series_config"
    
    feature_size: int = 1
    context_length: int = 128
    prediction_length: int = 12
    # Use Optional for quantiles if it can be None, then handle default in __post_init__ or property
    quantiles: Optional[List[float]] = field(default_factory=list)
    output_token_lengths: int = 1
    loss_config: LossConfig = field(default_factory=MSELossConfig) # Use specific loss config
    use_dynamic_features: bool = False
    use_static_features: bool = False
    autoregressive: bool = True
    is_decoder: bool = False
    
    # Add a target_dim for explicit handling, if not derived
    target_dim: Optional[int] = None # This will be set by TransformerTimeSeriesConfig typically

    def __post_init__(self):
        # BaseConfig's __post_init__ is called automatically by dataclass MRO if it exists.
        # PretrainedConfig does not have __post_init__.
        # We explicitly set PretrainedConfig attributes using object.__setattr__
        # because the dataclass is frozen.
        object.__setattr__(self, "_name_or_path", self.type)
        object.__setattr__(self, "model_type", self.type)

        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")
        
        # Validation for loss_config and quantiles
        if self.loss_config.type in PROBABILISTIC_LOSSES:
            if not self.quantiles:
                raise ValueError(f"Probabilistic loss '{self.loss_config.type}' requires a list of quantiles.")
            if not all(0 < q < 1 for q in self.quantiles):
                raise ValueError("All quantile values must be in the open interval (0, 1)")
        
        # If target_dim is not set, default it to feature_size
        if self.target_dim is None:
            object.__setattr__(self, "target_dim", self.feature_size)

    # Override to_dict to integrate with BaseConfig's recursive to_dict
    def to_dict(self) -> Dict[str, Any]:
        # Use BaseConfig's to_dict to get dataclass fields serialized
        data = super().to_dict()
        
        # Add PretrainedConfig's specific fields if they are not already there
        if not "_name_or_path" in data: data["_name_or_path"] = self._name_or_path
        if not "model_type" in data: data["model_type"] = self.model_type
        
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        # Ensure 'type' is set for BaseConfig's from_dict if not present in input
        if "type" not in data:
            data["type"] = cls.type # Default to the current class's type

        # Handle nested loss_config specifically before passing to super
        if "loss_config" in data and isinstance(data["loss_config"], dict):
            data["loss_config"] = loss_config_from_dict(data["loss_config"])
        elif "loss_type" in data and "loss_config" not in data: # Handle old loss_type
            data["loss_config"] = loss_config_from_dict({"type": data.pop("loss_type")})
        
        # If num_quantiles is directly set, reconstruct quantiles if needed
        num_quantiles = data.pop("num_quantiles", None) 
        if num_quantiles is not None and not data.get("quantiles"):
            data["quantiles"] = np.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles).tolist()
        elif num_quantiles is not None and data.get("quantiles") and num_quantiles != len(data["quantiles"]):
             raise ValueError(
                        f'num_quantiles ({num_quantiles}) does not match len(quantiles) '
                        f'({len(data["quantiles"])}) — set one or the other.'
                    )

        # The super().from_dict (from BaseConfig) will handle filtering valid keys and recursive parsing.
        # It will correctly instantiate the dataclass using its __init__ and then call __post_init__.
        instance = super().from_dict(data)
        
        return instance
