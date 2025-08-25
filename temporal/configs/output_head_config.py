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

@register_config_type("gaussian_output_head")
@dataclass(frozen=True, kw_only=True)
class GaussianOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a Gaussian (Normal) distribution output head.
    """
    # This type string is the user-facing identifier
    type: str = field(default="gaussian")

    # Parameters specific to the GaussianHead, moved from kwargs
    min_log_sigma: float = -7.0
    max_log_sigma: float = 5.0
    sigma_floor: float = 1e-4
    init_log_sigma: Optional[float] = None

@register_config_type("distpred_output_head")
@dataclass(frozen=True, kw_only=True)
class DistPredOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a distributional prediction head.
    """
    # New required fields (kw_only means order doesn't matter for init, but good practice)
    num_outputs: int
    feature_size: int

    # Override type field from base, and new default fields
    type: str = field(default="distpred")
    use_tanh: bool = field(default=False)
    tanh_scale: float = field(default=10.0) # ADDED: For DistPredHead

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
    num_quantiles: int # ADDED: For QuantileRegressionOutputHead
    feature_size: int = field(default=1) # ADDED: For QuantileRegressionOutputHead
    type: str = field(default="quantile_regression")

    def __post_init__(self):
        super().__post_init__()
        if self.num_quantiles <= 0:
            raise ValueError("num_quantiles must be a positive integer.")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")

from temporal.configs.output_head_config import OutputHeadConfig  # whatever your base is

@register_config_type("mixture_output_head")
@dataclass(frozen=True, kw_only=True)
class MixtureOutputHeadConfig(OutputHeadConfig):
    """
    Config for MixtureOutputHead (univariate MDN).
    - `components`: list of component names; aliases are accepted and canonicalized.
    - `feature_size`: kept for API parity; MixtureOutputHead is univariate and will
      assert feature_size == 1.
    """
    components: List[str] = field(default_factory=list)
    feature_size: int = 1
    type: str = field(default="mixture")

    # ---- alias / validation tables (kept in sync with the head) ----
    _ALIASES: Dict[str, str] = field(default_factory=lambda: {
        # Gaussian
        "gaussian": "normal", "gauss": "normal", "normal": "normal",
        # Log-normal
        "lognormal": "log_normal", "log_norm": "log_normal", "log-normal": "log_normal",
        # Student-t
        "studentt": "student_t", "student_t": "student_t", "student-t": "student_t", "t": "student_t",
        # Negative Binomial
        "negativebinomial": "neg_binomial", "negative_binomial": "neg_binomial",
        "neg-binomial": "neg_binomial", "nb": "neg_binomial",
        # Fixed Normal
        "fixednormal": "fixed_normal", "fixed_normal": "fixed_normal", "fixed-normal": "fixed_normal",
    }, init=False, repr=False)

    _DIST_PARAM_COUNTS: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "normal":       {"mu": 1, "sigma": 1},
        "fixed_normal": {"mu": 1},
        "student_t":    {"df": 1, "mu": 1, "scale": 1},
        "log_normal":   {"mu": 1, "sigma": 1},
        "neg_binomial": {"r": 1, "p": 1},
    }, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not self.components:
            raise ValueError("MixtureOutputHead requires at least one component.")

        # Canonicalize & validate
        canon = []
        for c in self.components:
            key = c.strip().lower().replace(" ", "").replace("-", "_")
            c2 = self._ALIASES.get(key, key)
            if c2 not in self._DIST_PARAM_COUNTS:
                known = ", ".join(sorted(self._DIST_PARAM_COUNTS.keys()))
                raise ValueError(
                    f"Unknown mixture component '{c}'. After normalization -> '{c2}'. "
                    f"Known components: [{known}]."
                )
            canon.append(c2)

        # write back canonical list (dataclass is frozen)
        object.__setattr__(self, "components", canon)

        if self.feature_size != 1:
            raise NotImplementedError("MixtureOutputHead is currently univariate (feature_size must be 1).")

    @property
    def derived_output_size(self) -> int:
        """
        Total projection width the head needs:
          sum(params_per_component) + num_components (for mixture logits).
        """
        total = self.num_components  # mixture logits
        for c in self.components:
            total += sum(self._DIST_PARAM_COUNTS[c].values())
        return total

    @property
    def num_components(self) -> int:
        return len(self.components)



@register_config_type("timeflow_output_head")
@dataclass(frozen=True, kw_only=True)
class TimeFlowOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a TimeFlow output head.
    """
    target_channels: int
    cond_channels: int
    num_blocks: int
    model_channels: int
    num_sampling_steps: int = field(default=10)
    type: str = field(default="timeflow")

    def __post_init__(self):
        super().__post_init__()
        if self.target_channels <= 0:
            raise ValueError("target_channels must be a positive integer.")
        if self.cond_channels <= 0:
            raise ValueError("cond_channels must be a positive integer.")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer.")
        if self.model_channels <= 0:
            raise ValueError("model_channels must be a positive integer.")

@register_config_type("student_t_output_head")
@dataclass(frozen=True, kw_only=True)
class StudentTOutputHeadConfig(OutputHeadConfig):
    """
    Configuration for a Student's T-Distribution output head.
    """
    feature_size: int
    num_outputs: int = field(default=1)
    type: str = field(default="student_t")

    def __post_init__(self):
        super().__post_init__()
        if self.feature_size <= 0:
            raise ValueError("feature_size must be a positive integer.")
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be a positive integer.")

# Helper function for polymorphic creation
def output_head_config_from_dict(data: Dict[str, Any]) -> OutputHeadConfig:
    output_head_type = data.get("type", "linear")
    # Map config type names to registry keys if they differ
    type_to_registry_key = {
        "linear": "output_head", # Default for the base OutputHeadConfig
        "gaussian": "gaussian_output_head", # <-- UPDATE THIS LINE
        "distpred": "distpred_output_head",
        "quantile_regression": "quantile_regression_output_head", # ADDED
        "mixture": "mixture_output_head", # ADDED
        "timeflow": "timeflow_output_head",
        "student_t": "student_t_output_head",
    }
    registry_key = type_to_registry_key.get(output_head_type, output_head_type) # Fallback to type if not in map

    config_class = CONFIG_REGISTRY.get(registry_key)
    if not config_class or not issubclass(config_class, OutputHeadConfig):
        raise ValueError(f"Unknown or invalid output_head_type: {output_head_type} (mapped to registry key: {registry_key})")
    
    return config_class.from_dict(data)
