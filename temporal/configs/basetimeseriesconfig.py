from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Dict, Any, Type, TypeVar
import numpy as np

from transformers import PretrainedConfig

from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY
from temporal.configs.loss_config import LossConfig, MSELossConfig, loss_config_from_dict, PROBABILISTIC_LOSSES

T = TypeVar('T', bound='BaseTimeSeriesConfig')

@register_config_type("base_time_series_config")
@dataclass(frozen=True, kw_only=True) # All fields are keyword-only now
class BaseTimeSeriesConfig(BaseConfig, PretrainedConfig):
    """
    Configuration for a base time series forecasting model.

    Inherits from :class:`transformers.PretrainedConfig` to enable saving/loading
    via the Hugging Face ecosystem. This dataclass also integrates with the
    local `BaseConfig` for `to_dict` and `from_dict` consistency.
    """
    # All fields explicitly defined as keyword-only using field(default=...)
    type: str = field(default="base_time_series_config")
    
    feature_size: int = field(default=1)
    context_length: int = field(default=128)
    prediction_length: int = field(default=12)
    quantiles: Optional[List[float]] = field(default_factory=list)
    output_token_lengths: int = field(default=1)
    loss_config: LossConfig = field(default_factory=MSELossConfig)
    use_dynamic_features: bool = field(default=False)
    use_static_features: bool = field(default=False)
    autoregressive: bool = field(default=True)
    is_decoder: bool = field(default=False)
    target_dim: Optional[int] = field(default=None)

    def __post_init__(self):
        super().__post_init__() # Call BaseConfig's post_init for its own validations
        # No need to call PretrainedConfig's __init__; it's handled by its __new__ or meta-class logic
        # and we manually set attributes for frozen dataclass compatibility.
        object.__setattr__(self, "_name_or_path", self.type)
        object.__setattr__(self, "model_type", self.type)
        if not hasattr(self, "_attn_implementation_internal"):
            object.__setattr__(self, "_attn_implementation_internal", "eager")
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

    def to_dict(self) -> Dict[str, Any]:
        # Use BaseConfig's to_dict to get dataclass fields serialized
        data = super().to_dict()
        
        # Add PretrainedConfig's specific fields if they are not already there
        if "_name_or_path" not in data: data["_name_or_path"] = self._name_or_path
        if "model_type" not in data: data["model_type"] = self.model_type
        
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Construct a config from a Python dict, normalizing legacy keys and
        filling sane defaults. Does not mutate the input `data`.
        """
        d: Dict[str, Any] = dict(data or {})  # shallow copy to avoid side-effects

        # Ensure required 'type' (used by BaseConfig serialization)
        d.setdefault("type", getattr(cls, "type", cls.__name__.lower()))

        # ---- Normalize loss config ----
        loss_cfg = d.get("loss_config")
        if isinstance(loss_cfg, dict):
            d["loss_config"] = loss_config_from_dict(loss_cfg)
        elif "loss_type" in d and "loss_config" not in d:
            d["loss_config"] = loss_config_from_dict({"type": d.pop("loss_type")})

        # ---- Quantiles handling ----
        num_quantiles = d.pop("num_quantiles", None)
        if num_quantiles is not None:
            if not isinstance(num_quantiles, int) or num_quantiles <= 0:
                raise ValueError("num_quantiles must be a positive integer")
            if "quantiles" not in d or d["quantiles"] in (None, []):
                d["quantiles"] = np.linspace(
                    0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles
                ).tolist()
            elif len(d["quantiles"]) != num_quantiles:
                raise ValueError(
                    f"num_quantiles ({num_quantiles}) does not match "
                    f"len(quantiles) ({len(d['quantiles'])}). Set one or the other."
                )

        # Create instance via parent
        instance: T = super().from_dict(d)

        # ---- HF compatibility: attention implementation flag ----
        attn_impl = d.get("_attn_implementation") or d.get("_attn_implementation_internal")
        if attn_impl is not None:
            object.__setattr__(instance, "_attn_implementation_internal", attn_impl)

        return instance
