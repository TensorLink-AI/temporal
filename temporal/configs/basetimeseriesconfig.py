from transformers import PretrainedConfig


from transformers import PretrainedConfig
PROBABILISTIC_LOSSES = ["quantile", "mq"]  # Constant set, cannot be modified dynamically

class BaseTimeSeriesConfig(PretrainedConfig):
    
    """
    Base configuration for time series models.
    """

    model_type = "base_time_series"

    def __init__(
        self,
        feature_size: int = 1,
        context_length: int = 128,
        prediction_length: int = 12,
        quantiles: list = [0.1, 0.5, 0.9],
        output_token_lengths: int = 1,
        head_aggregation_method: str = "mean",
        loss_type: str = "quantile",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        autoregressive: bool = True,
        is_decoder: bool = False,

        # Transformer-specific hyperparameters
        hidden_size: int = 128,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_hidden_size: int = None,  # Defaults to 4 * hidden_size
        dropout: float = 0.1,
        attention_dropout: float = 0.1,

        # Architectural choices
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
        self.ffn_hidden_size = ffn_hidden_size or 4 * hidden_size  # Defaults to 4 * hidden_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # Architectural options
        self.use_layer_norm = use_layer_norm
        self.use_positional_encoding = use_positional_encoding
        self.use_skip_connections = use_skip_connections

        # Number of output quantiles
        self.num_quantiles = len(self.quantiles) if self.loss_type in PROBABILISTIC_LOSSES else 1

        # Validation checks
        self.validate_config()

    def validate_config(self):
        """Validate the configuration parameters."""
        assert self.context_length > 0, "context_length must be > 0"
        assert self.prediction_length > 0, "prediction_length must be > 0"
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        
        if self.loss_type in PROBABILISTIC_LOSSES:
            assert all(0 < q < 1 for q in self.quantiles), \
                "All quantiles must be between 0 and 1 when using probabilistic losses."
        
        allowed_aggregation_methods = {"mean", "gated", "attention", "fusion", "stacked", "weighted_mean"}
        assert self.head_aggregation_method in allowed_aggregation_methods, \
            f"Invalid head_aggregation_method. Choose from {allowed_aggregation_methods}"