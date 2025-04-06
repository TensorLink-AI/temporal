from transformers import PretrainedConfig

PROBABILISTIC_LOSSES = ["quantile", "mq"]  # Constant set, cannot be modified dynamically

class BaseTimeSeriesConfig(PretrainedConfig):
    """
    Configuration class for time-series transformer models.

    This config holds model hyperparameters related to:
      - Window lengths for time-series inputs and outputs
      - Choice of loss function (standard vs. probabilistic)
      - Transformer architecture details (hidden size, number of layers, etc.)
      - Feature usage (dynamic/static) and optional advanced settings

    NOTE:
      - If your model runs in a strict encoder–decoder style, you may not need
        the `autoregressive` or `is_decoder` flags (they can be superseded by
        your actual forward logic). Evaluate whether your code still uses these
        or if they're effectively deprecated.

    Args:
        feature_size (int, optional, defaults to 1):
            Number of features in each time-step (i.e., the dimensionality of
            each “token” or “value”).
        context_length (int, optional, defaults to 128):
            Number of historical steps used as input (the “lookback window”).
        prediction_length (int, optional, defaults to 12):
            Number of future steps to predict (the “forecast horizon”).
        quantiles (List[float], optional, defaults to [0.1, 0.5, 0.9]):
            Quantile values for probabilistic losses (e.g. "quantile" or "mq").
            Must be between 0.0 and 1.0.
        output_token_lengths (int, optional, defaults to 1):
            Number of positions or tokens the model outputs. Typically 1 if
            predicting a single step at once, or more if the model outputs
            multiple “heads” for multi-dimensional forecasts. Also used for
            custom “head aggregation”.
        head_aggregation_method (str, optional, defaults to "mean"):
            Strategy for aggregating multiple output heads. Options include
            {"mean", "gated", "attention", "fusion", "stacked", "weighted_mean"}.
        loss_type (str, optional, defaults to "quantile"):
            Which loss function to use:
              - "mse", "mae", "rmse" for standard regression losses
              - "quantile", "mq" for quantile-based probabilistic losses
              - other custom options if integrated in your code
        use_dynamic_features (bool, optional, defaults to False):
            Whether your model expects time-varying auxiliary inputs
            (e.g. holiday flags, known future variables).
        use_static_features (bool, optional, defaults to False):
            Whether your model uses any static (time-invariant) categorical or
            real features. If True, additional embeddings or projections might
            be used.
        autoregressive (bool, optional, defaults to True):
            If True, the model is intended to run step-by-step in an AR loop.
            If you rely on multi-step teacher forcing in a single forward pass,
            you might set this to False. (Potentially deprecated if your code
            no longer uses single-step loops.)
        is_decoder (bool, optional, defaults to False):
            A flag that some internals (like multi-head attention) might check
            to treat this module as a decoder. If you use a strict
            encoder–decoder approach, you may rely on other ways to designate
            a “decoder” sub-model. Potentially deprecated in modern usage.

        # Transformer-specific hyperparameters:

        hidden_size (int, optional, defaults to 128):
            Dimensionality of the model’s embeddings and hidden states.
            Must be divisible by `num_attention_heads`.
        num_layers (int, optional, defaults to 4):
            Number of transformer layers (stacked encoders or decoders).
        num_attention_heads (int, optional, defaults to 8):
            Number of attention heads per layer.
        ffn_hidden_size (int, optional, defaults to 4*hidden_size):
            Dimensionality of the “intermediate” (feed-forward) layer inside
            each transformer block. If None, defaults to 4 * hidden_size.
        dropout (float, optional, defaults to 0.1):
            Dropout rate applied to embeddings and residual connections.
        attention_dropout (float, optional, defaults to 0.1):
            Dropout rate applied inside the attention mechanism.

        # Architectural choices:

        use_layer_norm (bool, optional, defaults to True):
            Whether to apply LayerNorm in key places (e.g. after self-attention
            or feed-forward).
        use_positional_encoding (bool, optional, defaults to True):
            Whether to add positional encodings to each token/time-step.
        use_skip_connections (bool, optional, defaults to True):
            Whether each sub-block uses a residual (skip) connection
            around the self-attention and feed-forward blocks.

        kwargs (dict, optional):
            Additional keyword arguments passed to PretrainedConfig. Usually
            includes identifiers or special options from Hugging Face’s config
            ecosystem.

    Attributes:
        num_quantiles (int):
            Computed automatically as `len(quantiles)` if `loss_type` is among
            PROBABILISTIC_LOSSES (i.e., "quantile" or "mq"), else 1.

    Raises:
        AssertionError: If `context_length <= 0`, `prediction_length <= 0`, or
                        `hidden_size % num_attention_heads != 0`.
        ValueError: If any quantile is not in (0,1) for "quantile"/"mq" losses,
                    or if `head_aggregation_method` is not recognized.
    """

    model_type = "base_time_series"

    def __init__(
        self,
        feature_size: int = 1,
        context_length: int = 128,
        prediction_length: int = 12,
        quantiles: list = [0.1, 0.5, 0.9],
        output_token_lengths: int = 1,
        head_aggregation_method: str = "head2head",
        loss_type: str = "quantile",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        autoregressive: bool = True,   # Possibly deprecated if you fully rely on multi-step
        is_decoder: bool = False,      # Possibly deprecated in a strict encoder–decoder approach

        # Transformer hyperparams
        hidden_size: int = 128,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_hidden_size: int = None,   # Defaults to 4 * hidden_size
        dropout: float = 0.1,
        attention_dropout: float = 0.1,

        # Architectural options
        use_layer_norm: bool = True,
        use_positional_encoding: bool = True,
        use_skip_connections: bool = True,
        
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Model architecture parameters
        self.feature_size = feature_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.output_token_lengths = output_token_lengths
        self.head_aggregation_method = head_aggregation_method
        self.loss_type = loss_type
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.autoregressive = autoregressive
        self.is_decoder = is_decoder

        # Transformer-specific parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size or 4 * hidden_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # Architectural options
        self.use_layer_norm = use_layer_norm
        self.use_positional_encoding = use_positional_encoding
        self.use_skip_connections = use_skip_connections

        # Number of output quantiles
        from . import PROBABILISTIC_LOSSES  # or define at top
        self.num_quantiles = len(self.quantiles) if self.loss_type in PROBABILISTIC_LOSSES else 1

        # Validation checks
        self.validate_config()

    def validate_config(self):
        assert self.context_length > 0, "context_length must be > 0"
        assert self.prediction_length > 0, "prediction_length must be > 0"
        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )

        if self.loss_type in PROBABILISTIC_LOSSES:
            assert all(0 < q < 1 for q in self.quantiles), (
                "All quantiles must be between 0 and 1 for probabilistic losses."
            )

        allowed_aggregation_methods = {"mean", "gated", "attention", "fusion", "stacked", "weighted_mean"}
        assert self.head_aggregation_method in allowed_aggregation_methods, (
            f"Invalid head_aggregation_method. Choose from {allowed_aggregation_methods}"
        )
