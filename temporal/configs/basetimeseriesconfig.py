from transformers import PretrainedConfig
from typing import List, Optional

#: List of loss types treated as probabilistic. Each requires valid quantile values in (0, 1).
PROBABILISTIC_LOSSES = ["quantile", "mq"]


class BaseTimeSeriesConfig(PretrainedConfig):
    """Configuration for a base time series forecasting model.

    Inherits from :class:`transformers.PretrainedConfig` to enable saving/loading
    via the Hugging Face ecosystem.

    Core attributes (shared by all time-series model variants):

        feature_size (int): Number of input features per time step.
        context_length (int): Length of the historical context window.
        prediction_length (int): Forecast horizon length.
        quantiles (List[float]): Quantile levels for probabilistic forecasting.
        output_token_lengths (int): Number of output tokens produced per forecast.
        loss_type (str): Loss function type; one of PROBABILISTIC_LOSSES or a point-wise loss.
        use_dynamic_features (bool): Whether to include dynamic covariates.
        use_static_features (bool): Whether to include static covariates.
        autoregressive (bool): If True, model predicts one step at a time.
        is_decoder (bool): If True, configures model for decoder-only architectures.
    """

    model_type    = "base_time_series"
    # start with HF's built-in mapping, then we'll auto-extend it below
    attribute_map = dict(PretrainedConfig.attribute_map)

    def __init__(
        self,
        feature_size: int = 1,
        context_length: int = 128,
        prediction_length: int = 12,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        output_token_lengths: int = 1,
        loss_type: str = "quantile",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        autoregressive: bool = True,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # core settings
        self.feature_size         = feature_size
        self.context_length       = context_length
        self.prediction_length    = prediction_length

        # probabilistic forecasting
        self.quantiles            = quantiles
        self.output_token_lengths = output_token_lengths
        self.loss_type            = loss_type

        # covariate flags
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features  = use_static_features

        # sequence mode
        self.autoregressive       = autoregressive
        self.is_decoder           = is_decoder

        self.validate_config()

    def validate_config(self):
        """Validates configuration for consistency."""
        assert self.context_length > 0,        "context_length must be > 0"
        assert self.prediction_length > 0,     "prediction_length must be > 0"
        if self.loss_type in PROBABILISTIC_LOSSES:
            assert all(0 < q < 1 for q in self.quantiles), (
                "All quantile values must be in the open interval (0, 1)"
            )


# ----------------------------------------------------------------
# AUTO-REGISTER all top-level fields in attribute_map
# ----------------------------------------------------------------
# instantiate a dummy config to discover its to_dict() keys
_dummy = BaseTimeSeriesConfig(_name_or_path="base_time_series")
BaseTimeSeriesConfig.attribute_map.update({k: k for k in _dummy.to_dict().keys()})
del _dummy
