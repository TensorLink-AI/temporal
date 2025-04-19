from transformers import PretrainedConfig

#: List of loss types treated as probabilistic. Each requires valid quantile values in (0, 1).
PROBABILISTIC_LOSSES = ["quantile", "mq"]

class BaseTimeSeriesConfig(PretrainedConfig):
    """Configuration for a base time series transformer model.

    Inherits from :class:`transformers.PretrainedConfig` to enable saving/loading
    via the Hugging Face ecosystem.

    Attributes:
        feature_size (int): Number of input features per time step.
        context_length (int): Length of the historical context window.
        prediction_length (int): Forecast horizon length.
        quantiles (List[float]): Quantile levels for probabilistic forecasting.
        output_token_lengths (int): Number of output tokens per prediction step.
        loss_type (str): Loss function type; must be one of PROBABILISTIC_LOSSES or a point-wise loss.
        use_dynamic_features (bool): Whether to include dynamic covariates.
        use_static_features (bool): Whether to include static covariates.
        autoregressive (bool): If True, model predicts one step at a time autoregressively.
        is_decoder (bool): If True, configures transformer blocks for decoding.
        hidden_size (int): Dimensionality of hidden representations.
        num_layers (int): Number of transformer encoder/decoder layers.
        num_attention_heads (int): Number of attention heads.
        ffn_hidden_size (int): Hidden size of the feed-forward sublayers.
        dropout (float): Dropout probability for feed-forward layers.
        attention_dropout (float): Dropout probability for attention weights.
        use_layer_norm (bool): Whether to apply layer normalization.
        use_positional_encoding (bool): Whether to add positional encodings.
        use_skip_connections (bool): Whether to use residual skip connections.
        value_embedding_type (str): Type of embedding for input values.
        pos_embedding_type (str): Type of positional embedding to use.
    """

    model_type = "base_time_series"

    def __init__(
        self,
        feature_size: int = 1,
        context_length: int = 128,
        prediction_length: int = 12,
        quantiles: list = [0.1, 0.5, 0.9],
        output_token_lengths: int = 1,
        loss_type: str = "quantile",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        autoregressive: bool = True,
        is_decoder: bool = False,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_hidden_size: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_positional_encoding: bool = True,
        use_skip_connections: bool = True,
        value_embedding_type: str = "value",
        pos_embedding_type: str = "positional_sinusoidal",
        **kwargs,
    ):
        """Initializes the base time series model configuration.

        Args:
            feature_size (int): Number of input features per time step.
            context_length (int): Length of the historical context window.
            prediction_length (int): Forecast horizon length.
            quantiles (List[float]): Quantile levels for probabilistic forecasting.
            output_token_lengths (int): Number of output tokens per prediction step.
            loss_type (str): Loss function type; must be one of PROBABILISTIC_LOSSES or a point-wise loss.
            use_dynamic_features (bool): Whether to include dynamic covariates.
            use_static_features (bool): Whether to include static covariates.
            autoregressive (bool): If True, model predicts one step at a time autoregressively.
            is_decoder (bool): If True, configures transformer blocks for decoding.
            hidden_size (int): Dimensionality of hidden representations.
            num_layers (int): Number of transformer encoder/decoder layers.
            num_attention_heads (int): Number of attention heads.
            ffn_hidden_size (Optional[int]): Hidden size of the feed-forward sublayers.
            dropout (float): Dropout probability for feed-forward layers.
            attention_dropout (float): Dropout probability for attention weights.
            use_layer_norm (bool): Whether to apply layer normalization.
            use_positional_encoding (bool): Whether to add positional encodings.
            use_skip_connections (bool): Whether to use residual skip connections.
            value_embedding_type (str): Type of embedding for input values.
            pos_embedding_type (str): Type of positional embedding to use.
            **kwargs: Additional keyword arguments passed to :class:`PretrainedConfig`.
        """
        super().__init__(**kwargs)

        self.feature_size = feature_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.output_token_lengths = output_token_lengths
        self.loss_type = loss_type
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.autoregressive = autoregressive
        self.is_decoder = is_decoder

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size or 4 * hidden_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.use_layer_norm = use_layer_norm
        self.use_positional_encoding = use_positional_encoding
        self.use_skip_connections = use_skip_connections
        self.value_embedding_type = value_embedding_type
        self.pos_embedding_type = pos_embedding_type

        self.validate_config()

    def validate_config(self):
        """Validates the configuration attributes for consistency and correctness.

        Raises:
            AssertionError: If any of the following conditions are violated:
                - `context_length` and `prediction_length` must be positive.
                - `hidden_size` must be divisible by `num_attention_heads`.
                - For probabilistic losses, all `quantiles` must be in the interval (0, 1).
        """
        assert self.context_length > 0, "context_length must be > 0"
        assert self.prediction_length > 0, "prediction_length must be > 0"
        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )
        if self.loss_type in PROBABILISTIC_LOSSES:
            assert all(0 < q < 1 for q in self.quantiles), (
                "All quantile values must be in the open interval (0, 1)"
            )
